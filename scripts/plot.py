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



def train_visualization(nodes,ret_nodes,observed_length,ret_attn,name,color_dict = None,peds=4):
    print('Visualization imminent')
    if color_dict is None: color_dict = {k:np.random.rand(3) for k in range(nodes.shape[1])}
        
    ## Ground truth frame
    for frames in range(nodes.shape[0]+1): 
        fig,ax = plt.subplots()
        marker,linestyle = 'o','solid'
        marker_pred,linestyle_pred = '+','dashed'
        
        for frame_id in range(frames):

            ## Preparation for plotting the ground truth frames
            current_frame = nodes[frame_id,:,:]
            current_peds = np.array([i for i,(x,y) in enumerate(current_frame) if (x != 0) and (y != 0)])
            plt.scatter((current_frame[current_peds][:,0]+1)/2,(current_frame[current_peds][:,1]+1)/2,c = [color_dict[c] for c in current_peds],\
                        marker=marker,linestyle=linestyle)
            
            ## Preparation for plotting the predicted frames
            current_frame_pred = ret_nodes[frame_id,:,:]
            current_peds_pred = np.array([i for i,(x,y) in enumerate(current_frame) if (x != 0) and (y != 0)])
            try:
                plt.scatter((current_frame_pred[current_peds_pred][:,0]+1)/2,(current_frame_pred[current_peds_pred][:,1]+1)/2,c = [color_dict[c] for c in current_peds_pred],\
                        marker=marker_pred,linestyle=linestyle_pred)
                
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
                                c = color_dict[common[i]],linestyle=linestyle,marker=marker,alpha=0.3)
                    
                    try:
                        plt.plot([(prev_frame_pred[common_pred[i], 0]+1)/2, (current_frame_pred[common_pred[i], 0]+1)/2],[(prev_frame_pred[common_pred[i], 1]+1)/2, (current_frame_pred[common_pred[i], 1]+1)/2],\
                                c = color_dict[common_pred[i]],linestyle=linestyle_pred,marker=marker_pred,alpha=0.3)
                    except:pass
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
            attn_w     = dict(zip(ret_attn[frame].get(peds)[1],ret_attn[frame].get(peds)[0]))
            traj_ped_observed = current_frame[peds]
            circle = plt.Circle(((traj_ped_observed[0]+1)/2, (traj_ped_observed[1]+1)/2), 0.01, fill=True, color='k', linewidth=10)
            ax.add_artist(circle)
            for other_ped in common_attn_peds:
                traj_ped_observed = current_frame[other_ped]
                weight = attn_w[other_ped]
                circle = plt.Circle(((traj_ped_observed[0]+1)/2, (traj_ped_observed[1]+1)/2), weight*0.2, fill=False, color='b', linewidth=2)
                ax.add_artist(circle)

        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.savefig(f'./plot/{name} {frames}.png',bbox_inches='tight')
    return fig,ax


def main():
    save_directory = '../save/'
    save_directory += str(0) + '/save_attention/'
    f = open(save_directory+'results.pkl', 'rb')
    results = pickle.load(f)
    for i in range(len(results)):
        print (i)

        if i%207 !=0 and i!=0:continue
        train_visualization(results[i][0].squeeze(), results[i][1].squeeze(),results[i][3], results[i][4],i)
if __name__ == "__main__":
    main()