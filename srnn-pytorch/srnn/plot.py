import os
import pickle
import os
import pickle
import argparse
import time

import torch
from torch.autograd import Variable
from helper import *
import numpy as np
from helper import getCoef, sample_gaussian_2d, compute_edges, get_mean_error, get_final_error
from criterion import Gaussian2DLikelihood, Gaussian2DLikelihoodInference
from sample import sample
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
from utils_mouse import DataLoaderMouse as DM


def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]

    return absolute_x_seq


def sample(x_seq, Pedlist, net, true_x_seq, true_Pedlist, saved_args, dimensions, dataloader, look_up, num_pedlist, is_gru, grid = None):
    '''
    The sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    true_x_seq: True positions
    true_Pedlist: The true peds present in each frame
    saved_args: Training arguments
    dimensions: The dimensions of the dataset
    target_id: ped_id number that try to predict in this sequence
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        hidden_states = hidden_states.cuda()
        cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        cell_states = cell_states.cuda()

        ret_x_seq = Variable(torch.zeros(saved_args.obs_length+saved_args.pred_length, numx_seq, 2))
        ret_x_seq = ret_x_seq.cuda()


        # For the observed part of the trajectory
        for tstep in range(saved_args.obs_length-1):

            out_obs, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), None,hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
           
        ret_x_seq[:saved_args.obs_length, :, :] = x_seq.clone()

        # For the predicted part of the trajectory
        for tstep in range(saved_args.obs_length-1, saved_args.pred_length + saved_args.obs_length-1):
            # Do a forward prop
            outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, 2), None,hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
           
            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs)
            # Sample from the bivariate Gaussian
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)

            # Store the predicted position
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y

            # List of x_seq at the last time-step (assuming they exist until the end)
            true_Pedlist[tstep+1] = [int(_x_seq) for _x_seq in true_Pedlist[tstep+1]]
            next_ped_list = true_Pedlist[tstep+1].copy()
            converted_pedlist = [look_up[_x_seq] for _x_seq in next_ped_list]
            list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

#             if args.use_cuda:
            list_of_x_seq = list_of_x_seq.cuda()
           
            #Get their predicted positions
            # current_x_seq = torch.index_select(ret_x_seq[tstep+1], 0, list_of_x_seq)

        return ret_x_seq


class Visualisation:

    def single_keypoint_visualisation(self,nodes,ret_nodes,observed_length,ret_attn,name,color_dict=None,peds=0):
        '''
        Function to visualize human and mice trajectory along with their respective attention weights supported only for single 
        keypoint representation. The attention weights plotting is similar to the one followed
        in the social attention model: https://github.com/cmubig/socialAttention/tree/master/social-attention/srnn-pytorch
        
        '''
        print('Visualization imminent')
        if color_dict is None: color_dict = {k:np.random.rand(3) for k in range(self.nodes.shape[1])}
        
        ## Ground truth frame
        
        for frames in range(nodes.shape[0]+1): 
            fig,ax = plt.subplots()
            marker,linestyle = 'o','solid'
            marker_pred,linestyle_pred = '+','dashed'

            for frame_id in range(frames):
                
                ## Preparation for plotting the ground truth frames
                current_frame = self.nodes[frame_id,:,:]
                current_peds = np.array([i for i,(x,y) in enumerate(current_frame) if (x != 0) and (y != 0)])
                plt.scatter((current_frame[current_peds][:,0]+1)/2,(current_frame[current_peds][:,1]+1)/2,c = [color_dict[c] for c in current_peds],\
                            marker=marker,linestyle=linestyle,edgecolors=None)
    
                ## Preparation for plotting the predicted frames
                current_frame_pred = self.ret_nodes[frame_id,:,:]
                current_peds_pred = np.array([i for i,(x,y) in enumerate(current_frame) if (x != 0) and (y != 0)])
                try:
                    plt.scatter((current_frame_pred[current_peds_pred][:,0]+1)/2,(current_frame_pred[current_peds_pred][:,1]+1)/2,c = [color_dict[c] for c in current_peds_pred],\
                            marker=marker_pred,linestyle=linestyle_pred,edgecolors=None)
                    
                except:pass
        
                if frame_id>0:

                    ## Ground truth trajectories

                    prev_frame = nodes[frame_id-1,:,:]
                    prev_peds  = np.array([i for i,(x,y) in enumerate(prev_frame) if (x != 0) and (y != 0)])
                    common = np.array(list(set(prev_peds) & set(current_peds)))

                    ## Predicted trajectories

                    prev_frame_pred = ret_nodes[frame_id-1,:,:]
                    prev_peds_pred  = np.array([i for i,(x,y) in enumerate(prev_frame_pred) if (x != 0) and (y != 0)])
                    common_pred = np.array(list(set(prev_peds_pred) & set(current_peds_pred)))

                    for i in range(len(common)):
                        plt.plot([(prev_frame[common[i], 0]+1)/2, (current_frame[common[i], 0]+1)/2],[(prev_frame[common[i], 1]+1)/2, (current_frame[common[i], 1]+1)/2],\
                                    c = color_dict[common[i]],linestyle=linestyle,marker=marker)
                        
                        try:
                            plt.plot([(prev_frame_pred[common_pred[i], 0]+1)/2, (current_frame_pred[common_pred[i], 0]+1)/2],[(prev_frame_pred[common_pred[i], 1]+1)/2, (current_frame_pred[common_pred[i], 1]+1)/2],\
                                    c = color_dict[common_pred[i]],linestyle=linestyle_pred,marker=marker_pred,alpha=0.3)
                        except:pass
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.text(-0.75,-0.6,'AW for Mouse 1: {:.2f}'.format(float("inf")),fontsize=10,color='blue',bbox=props)
                plt.text(-0.75,-0.8,'AW for Mouse 2: {:.2f}'.format(float("inf")),fontsize=10,color='green',bbox=props)
            # if frames!=0:
            #     plt.scatter((current_frame[current_peds][:,0]+1)/2,(current_frame[current_peds][:,1]+1)/2,c ='k',marker='D',linewidths=3)

            current_frame = ret_nodes[frames-1,:,:]
            if frames>=observed_length:
                frame = frames-observed_length-1
                peds_other = ret_attn[frame].get(peds)
                if peds_other is None:
                    pedold = peds
                    peds = list(ret_attn[frame].keys())[0]
                    peds_other = ret_attn[frame].get(peds)[1]
                    print(f'Key Changed from {pedold} to {peds}')
                else:peds_other = peds_other[1]
                print(f'Key requested: {peds} is available.')
                common_attn_peds = list((set(current_peds)^set([peds]))&set(peds_other))
                attn_w     = dict(zip(ret_attn[frame].get(peds)[1],self.ret_attn[frame].get(peds)[0]))
                props = dict(boxstyle='round', facecolor='wheat', alpha=1)
                plt.text(-0.75,-0.6,'AW for Mouse 1: {:.2f}'.format(attn_w[1]),fontsize=10,color='blue',bbox=props)
                plt.text(-0.75,-0.8,'AW for Mouse 2: {:.2f}'.format(attn_w[2]),fontsize=10,color='green',bbox=props)
                traj_ped_observed = current_frame[peds]
                circle = plt.Circle(((traj_ped_observed[0]+1)/2, (traj_ped_observed[1]+1)/2), 0.01, fill=True, color='k', linewidth=2)
                ax.add_artist(circle)
                for other_ped in common_attn_peds:
                    traj_ped_observed = current_frame[other_ped]
                    weight = attn_w[other_ped]
                    circle = plt.Circle(((traj_ped_observed[0]+1)/2, (traj_ped_observed[1]+1)/2), weight*0.1, fill=False, color='b', linewidth=2)
                    # plt.text()
                    ax.add_artist(circle)
                
            if frames!=0:
                plt.title(f'Sequence number: {name//2}')
                plt.ylim((-1, 1))
                plt.xlim((-1, 1))
                plt.grid(True)
                plt.savefig(f'./plots/{name}_{frames}.png',bbox_inches='tight')
        return fig,ax
    
    ############################################################
    ## Mouse plotting function based on the keypoint location ##
    ############################################################
    
    def plot_mouse_with_bodyline(self,pose, color,linestyle = None):
        from criterion import joint_xy
        PLOT_MOUSE_START_END = [(0, 1), (1, 3), (3, 2), (2, 0),        # head
                                (3, 6), (6, 9),                        # midline
                                (9, 10), (10, 11),                     # tail
                                (4, 5), (5, 8), (8, 9), (9, 7), (7, 4) # legs
                                ]

        # Draw each keypoint
        # for j in range(10):
        #     plt.plot(pose[j, 0], pose[j, 1], 'o', color=color, markersize=3)

        # Draw a line for each point pair to form the shape of the mouse
        # for pair in PLOT_MOUSE_START_END:
        #     line_to_plot = pose[pair, :]
        #     plt.plot(line_to_plot[:, 0], line_to_plot[
        #             :, 1], color=color, linewidth=1,linestyle = linestyle)
        bodyline_nodes = joint_xy(pose[None,None,...]).squeeze()
        # Draw a line for each point pair to form the shape of the mouse
        plt.scatter(bodyline_nodes[:,0],bodyline_nodes[:,1],color='b')
        plt.plot(bodyline_nodes[:,0],bodyline_nodes[:,1],color='k',linestyle=linestyle)
     
    
    def plot_mouse(self,pose,keypoints,color,linestyle):
        '''
        Draw a line for each point pair to form the skeleton of the mouse
        '''
        
        for i,pair in enumerate(keypoints):
            line_to_plot = pose[pair, :]
            if i == 0 or i == len(keypoints)-1:plt.plot(line_to_plot[0,0], line_to_plot[0,1], color=color, linewidth=1,marker='s')
            else:plt.plot(line_to_plot[0,0], line_to_plot[0,1], color=color, linewidth=1,marker='o')
            plt.plot(line_to_plot[1,0], line_to_plot[1,1], color=color, linewidth=1,marker='o')
            plt.plot(line_to_plot[:, 0], line_to_plot[
                    :, 1], color=color, linewidth=1,linestyle=linestyle)

    def createTable(self,attn_w,keypoints):
        ''''
        Creates a table consisting of the attention weights marked against the keypoints
        '''
        df = pd.DataFrame(data=attn_w)
        df.insert(0,'keypoints',keypoints)
        data = np.asarray(df)
        formatted_data = []
        for row in data:
            formatted_row = []
            for cell in row:
                if isinstance(cell, float):
                    formatted_cell = f'{cell:.3f}'  
                else:
                    formatted_cell = cell
                formatted_row.append(formatted_cell)
            formatted_data.append(formatted_row)
        return formatted_data
    
    def miceVisualisation(self,nodes,ret_nodes,observed_length,ret_attn,name,save_dir,keypoints,color_dict=None,attn_color=None,peds = 0,\
        table = False):
        '''
        Function to visualize mice trajectory along with their respective attention weights. The attention weights plotting is similar to the one followed
        in the social attention model: https://github.com/cmubig/socialAttention/tree/master/social-attention/srnn-pytorch

        '''

        if color_dict is None: color_dict = {k:np.random.rand(3) for k in range(nodes.shape[1])}
        if attn_color is None: attn_color = {k:np.random.rand(3) for k in range(nodes.shape[2])}
        print('Mice Visualisation imminent')
        if nodes.shape[1] > 3: 
            print('in')
            nodes = nodes.reshape(nodes.shape[0],3,12,2)
            ret_nodes = ret_nodes.reshape(ret_nodes.shape[0],3,12,2)

        for frames in range(nodes.shape[0]):
            fig = plt.figure(figsize=(8, 8)) 
            img = np.ones((850, 850, 3))
            ax = fig.add_subplot(111)
            
            ax.imshow(img)
            
            # fig,ax = plt.subplots(figsize=(10,10))
            _,linestyle = 'o','solid'
            _,linestyle_pred = '+','dashed'

            ## Preparation for plotting the ground truth frames
            current_frame = nodes[frames,:,:]
            current_peds = np.array([i for i,(x,y) in enumerate(current_frame[:,0,:]) if (x != 0) and (y != 0)])
            for i in range(current_frame.shape[0]):
                if keypoints is not None:
                    self.plot_mouse(current_frame[i],[(0,1),(1,2)],color=color_dict[i],linestyle=linestyle)
                else: 
                    self.plot_mouse_with_bodyline(current_frame[i],color=color_dict[i],linestyle=linestyle)

            ## Preparation for plotting the predicted frames
            current_frame_pred = ret_nodes[frames,:,:]
            #current_peds_pred = np.array([i for i,(x,y) in enumerate(current_frame[:,0,:]) if (x != 0) and (y != 0)])
            try:
                for i in range(current_frame.shape[0]):
                    if keypoints is not None:
                        self.plot_mouse(current_frame_pred[i],[(0,1),(1,2)],color=color_dict[i],linestyle=linestyle_pred)
                    else:
                        self.plot_mouse_with_bodyline(current_frame_pred[i],color = color_dict[i],linestyle=linestyle_pred)
                    
            except:pass

            if table:
                temp = {1:[float("inf")]*len(keypoints),2:[float("inf")]*len(keypoints)}
                data = self.createTable(temp,keypoints)
                table = ax.table(cellText=data,loc='upper left',colLabels=['kp']+[f'AW{k+1}' for k in range(len(keypoints))],colColours=\
                        ['yellow',color_dict[1],color_dict[2]],cellColours=[[attn_color[k],"w","w"] for k in range(nodes.shape[2])])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(0.8, 1)  #

            current_frame = ret_nodes[frames-1,:,:]
            if frames>=observed_length:
                frame = frames-observed_length-1
                peds_other = ret_attn[frame].get(0)
                if peds_other is None:
                    pedold = peds
                    peds = list(ret_attn[frame].keys())[0]
                    peds_other = ret_attn[frame].get(peds)[1]
                    print(f'Key Changed from {pedold} to {peds}')
                else:peds_other = peds_other[1]
                print(f'Key requested: {peds} is available.')
                common_attn_peds = list((set(current_peds)^set([peds]))&set(peds_other))
                attn_w     = dict(zip(ret_attn[frame].get(peds)[1],ret_attn[frame].get(peds)[0]))
                if table:
                    data = self.createTable(attn_w,keypoints)
                    table = ax.table(cellText=data,loc='upper left',colLabels=['kp']+[f'AW{k+1}' for k in range(len(keypoints))],colColours=\
                        ['yellow',color_dict[1],color_dict[2]],cellColours=[[attn_color[k],"w","w"] for k in range(nodes.shape[2])])
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(0.8, 1)  #
                traj_ped_observed = current_frame[peds]    
                circle = plt.Circle((traj_ped_observed[0,0], traj_ped_observed[0,1]), 0.01, fill=True, color='k', linewidth=2)
                ax.add_artist(circle)
                
                # for other_ped in common_attn_peds:
                #     traj_ped_observed = current_frame[other_ped]
                #     weight = attn_w[other_ped]
        
                #     for i in range(traj_ped_observed.shape[0]): 
                #         circle = plt.Circle((traj_ped_observed[i,0], traj_ped_observed[i,1]), weight[i]*0.1, fill=False, color=attn_color[i], linewidth=2)
                #         ax.add_artist(circle)

            if frames!=0:   
                plt.title(f'Sequence number: {name//100}')
                # plt.ylim((-1, 1))
                # plt.xlim((-1, 1))
                if not os.path.isdir(save_dir):
                    print(f'Directory does not exist... Creating the directory at {os.getcwd()+save_dir}')
                    os.mkdir(save_dir)
                
                plt.savefig(os.path.join(save_dir,f'{name}_{frames}.png'),bbox_inches='tight',dpi=100)
        print(f'Saved the image as: {os.path.join(os.getcwd(),save_dir,str(name))}__.png')        
        return fig,ax
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,default='results',help='Provide the filepath to the result.pkl file where the trajectory tensors are stored')
    parser.add_argument('--skip',type=int,default=5,help='Number of frames to be skipped without storing')
    parser.add_argument('--keypoint_id',type=str,default=None,help='Keypoint ID for each mice to be displayed in the table')
    parser.add_argument('--save_dir',type=str,default='\plots',help='Directory to store the saved .png files post visualisation')
    parser.add_argument('--load_dir',type=str,default="../save/",help="Directory to load the results.pkl file from")
    args = parser.parse_args()
    if args.keypoint_id is None: body_keypoint = None
    else:body_keypoint = [int(kp) for kp in args.keypoint_id.split(',')]
    save_directory = args.load_dir
    save_directory += str(0) + '/save_attention/'
    f = open(save_directory+args.filename+'.pkl', 'rb')
    results = pickle.load(f)
    color_dict = {0: 'red', 1: 'blue', 2: 'green'}
    # attn_color = {0:'sandybrown',1:'magenta','2':'blue'}
    attn_color = None
    visualise = Visualisation()
    for i in range(len(results)):
        if i%args.skip !=0 and i!=0:continue
        visualise.miceVisualisation(DM.revert_seq((results[i][0].squeeze())), DM.revert_seq(results[i][1].squeeze()),results[i][3], results[i][4],i,args.save_dir,body_keypoint,color_dict=color_dict,attn_color=attn_color)
        # train_visualization(results[i][0].squeeze(), results[i][1].squeeze(),results[i][3], results[i][4],i,color_dict=color_dict)
if __name__ == "__main__":
    main()
























# def train_visualization(nodes,ret_nodes,observed_length,ret_attn,name,color_dict = None,peds=0):
#     print('Visualization imminent')
#     if color_dict is None: color_dict = {k:np.random.rand(3) for k in range(nodes.shape[1])}
        
#     ## Ground truth frame
    
#     for frames in range(nodes.shape[0]+1): 
#         fig,ax = plt.subplots()
#         marker,linestyle = 'o','solid'
#         marker_pred,linestyle_pred = '+','dashed'

#         for frame_id in range(frames):
            
#             ## Preparation for plotting the ground truth frames
#             current_frame = nodes[frame_id,:,:]
#             current_peds = np.array([i for i,(x,y) in enumerate(current_frame) if (x != 0) and (y != 0)])
#             plt.scatter((current_frame[current_peds][:,0]+1)/2,(current_frame[current_peds][:,1]+1)/2,c = [color_dict[c] for c in current_peds],\
#                         marker=marker,linestyle=linestyle,edgecolors=None)
   
#             ## Preparation for plotting the predicted frames
#             current_frame_pred = ret_nodes[frame_id,:,:]
#             current_peds_pred = np.array([i for i,(x,y) in enumerate(current_frame) if (x != 0) and (y != 0)])
#             try:
#                 plt.scatter((current_frame_pred[current_peds_pred][:,0]+1)/2,(current_frame_pred[current_peds_pred][:,1]+1)/2,c = [color_dict[c] for c in current_peds_pred],\
#                         marker=marker_pred,linestyle=linestyle_pred,edgecolors=None)
                
#             except:pass
    
#             if frame_id>0:

#                 ## Ground truth trajectories

#                 prev_frame = nodes[frame_id-1,:,:]
#                 prev_peds  = np.array([i for i,(x,y) in enumerate(prev_frame) if (x != 0) and (y != 0)])
#                 common = np.array(list(set(prev_peds) & set(current_peds)))

#                 ## Predicted trajectories

#                 prev_frame_pred = ret_nodes[frame_id-1,:,:]
#                 prev_peds_pred  = np.array([i for i,(x,y) in enumerate(prev_frame_pred) if (x != 0) and (y != 0)])
#                 common_pred = np.array(list(set(prev_peds_pred) & set(current_peds_pred)))

#                 for i in range(len(common)):
#                     plt.plot([(prev_frame[common[i], 0]+1)/2, (current_frame[common[i], 0]+1)/2],[(prev_frame[common[i], 1]+1)/2, (current_frame[common[i], 1]+1)/2],\
#                                 c = color_dict[common[i]],linestyle=linestyle,marker=marker)
                    
#                     try:
#                         plt.plot([(prev_frame_pred[common_pred[i], 0]+1)/2, (current_frame_pred[common_pred[i], 0]+1)/2],[(prev_frame_pred[common_pred[i], 1]+1)/2, (current_frame_pred[common_pred[i], 1]+1)/2],\
#                                 c = color_dict[common_pred[i]],linestyle=linestyle_pred,marker=marker_pred,alpha=0.3)
#                     except:pass
#             props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#             plt.text(-0.75,-0.6,'AW for Mouse 1: {:.2f}'.format(float("inf")),fontsize=10,color='blue',bbox=props)
#             plt.text(-0.75,-0.8,'AW for Mouse 2: {:.2f}'.format(float("inf")),fontsize=10,color='green',bbox=props)
#         # if frames!=0:
#         #     plt.scatter((current_frame[current_peds][:,0]+1)/2,(current_frame[current_peds][:,1]+1)/2,c ='k',marker='D',linewidths=3)

#         current_frame = ret_nodes[frames-1,:,:]
#         if frames>=observed_length:
#             frame = frames-observed_length-1
#             peds_other = ret_attn[frame].get(peds)
#             if peds_other is None:
#                 pedold = peds
#                 peds = list(ret_attn[frame].keys())[0]
#                 peds_other = ret_attn[frame].get(peds)[1]
#                 print(f'Key Changed from {pedold} to {peds}')
#             else:peds_other = peds_other[1]
#             print(f'Key requested: {peds} is available.')
#             common_attn_peds = list((set(current_peds)^set([peds]))&set(peds_other))
#             attn_w     = dict(zip(ret_attn[frame].get(peds)[1],ret_attn[frame].get(peds)[0]))
#             props = dict(boxstyle='round', facecolor='wheat', alpha=1)
#             plt.text(-0.75,-0.6,'AW for Mouse 1: {:.2f}'.format(attn_w[1]),fontsize=10,color='blue',bbox=props)
#             plt.text(-0.75,-0.8,'AW for Mouse 2: {:.2f}'.format(attn_w[2]),fontsize=10,color='green',bbox=props)
#             traj_ped_observed = current_frame[peds]
#             circle = plt.Circle(((traj_ped_observed[0]+1)/2, (traj_ped_observed[1]+1)/2), 0.01, fill=True, color='k', linewidth=2)
#             ax.add_artist(circle)
#             for other_ped in common_attn_peds:
#                 traj_ped_observed = current_frame[other_ped]
#                 weight = attn_w[other_ped]
#                 circle = plt.Circle(((traj_ped_observed[0]+1)/2, (traj_ped_observed[1]+1)/2), weight*0.1, fill=False, color='b', linewidth=2)
#                 # plt.text()
#                 ax.add_artist(circle)
            
#         if frames!=0:
    
#             plt.title(f'Sequence number: {name//2}')
#             plt.ylim((-1, 1))
#             plt.xlim((-1, 1))
#             plt.grid(True)
#             plt.savefig(f'./plots/{name}_{frames}.png',bbox_inches='tight')
#     return fig,ax