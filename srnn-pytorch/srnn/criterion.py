'''
Criterion for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 30th March 2017
'''


import torch
import numpy as np
from helper import getCoef
from torch.autograd import Variable
from helper import getCoef, sample_gaussian_2d
from tqdm.auto import tqdm

def Gaussian2DLikelihood(outputs, targets, nodesPresent, pred_length):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''

    # Get the sequence length
    seq_length = outputs.size()[0]
    # Get the observed length
    obs_length = seq_length - pred_length
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)
  
    # Compute factors

    normx = targets[:, :,:, 0] - mux
    normy = targets[:, :,:, 1] - muy
    sxsy = sx * sy
    z = torch.pow((normx/sx), 2) + torch.pow((normy/sy), 2) - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - torch.pow(corr, 2)

    # Numerator
    result = torch.exp(-z/(2*negRho))

    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss across all frames and all nodes
    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]
        for nodeID in nodeIDs:
            for bodyID in range(outputs.shape[2]):
                loss = loss + result[framenum, nodeID,bodyID]
                counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss


def Gaussian2DLikelihoodInference(outputs, targets, assumedNodesPresent, nodesPresent):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    '''
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss
    loss = Variable(torch.zeros(1).cuda())
    counter = 0

    for framenum in range(outputs.size()[0]):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:
            if nodeID not in assumedNodesPresent:
                # If the node wasn't assumed to be present, don't compute loss for it
                continue
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss

def find_angle(v1,v2):
    # v1,v2 -> NX1 vector
    numerator  = torch.clip(torch.dot(v1,v2),-1,1)
    denominator = torch.linalg.norm(v1)*torch.linalg.norm(v2)
    cos = numerator/denominator
    return torch.rad2deg(torch.atan2((1-cos**2)**0.5,cos))

def boneloss(outputs,targets,nodesPresent,args,tolerance = 1e-3):
    '''
    outputs -> (N x 3 x 3 x 5) -> sample_2d_gaussian -> (N x 3 x 3 x 2)
    
    '''
    numNodes = outputs.shape[1]
    
    ret_nodes = Variable(torch.zeros(args.obs_length + args.pred_length, numNodes, outputs.shape[2],2), volatile=True).cuda()
    ret_nodes[:args.obs_length, :, :,:] = targets[:args.obs_length].clone()
    for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
   
        mux, muy, sx, sy, corr = getCoef(outputs[tstep][None,...])
        
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent[args.obs_length-1])
        ret_nodes[tstep + 1, :,:, 0] = next_x
        ret_nodes[tstep + 1, :,:, 1] = next_y
        
    
    ### Mouse bone length and joint angle information for 3 keypoints 
    ## seg_1 -> Euclidean distance between the nose (0) and centerback (6)
    ## seg_2 -> Euclidean distance between the centerback (6) and tailbase (9)
    ## angle -> joint angle in degrees representing the angle between the vectors connecting the keypoints (0,6) and (6,9) 

    mouseinfo = {'mouse_0':{'seg_1':(0.10969,0.10769), 'seg_2':(0.07123,0.07323),'angle':(0.81,0.809)},\
                 'mouse_1':{'seg_1':(0.08504,0.08404), 'seg_2':(0.069836,0.067836),'angle':(0.81,0.809)},\
                 'mouse_2':{'seg_1':(0.11827,0.11627), 'seg_2':(0.07001,0.06801),'angle':(0.81,0.809)}}
    
    NUM_MICE,NUM_FRAMES = ret_nodes.shape[1], ret_nodes.shape[0]
    counter,loss = 0,0
    for mouse in range(NUM_MICE):
        lengthloss_1,lengthloss_2,angleloss = 0.0,0.0,0.0
        for frame in range(NUM_FRAMES):
            arr_1 = ret_nodes[frame][mouse]
            arr_2 = torch.cat((ret_nodes[frame][mouse][1:],torch.zeros((1,2)).cuda()))
            diff  = arr_2-arr_1
            
            lengths = torch.linalg.norm(diff,axis=-1)[:2] ## -> (1x2) vector 
            angle = find_angle(diff[0,:],diff[1,:]) ## -> scalar (1x1)
            ## Length 1 loss
            uc,lc = mouseinfo[f'mouse_{mouse}']['seg_1']
            if lc<=lengths[0]<=uc:pass
            else:
                # if lengths[0]>uc:lengthloss_1+=uc-lengths[0]
                # else:lengthloss_1-=lc-lengths[0]
                lengthloss_1 += min(abs(mouseinfo[f'mouse_{mouse}']['seg_1'][0]-lengths[0]),\
                           abs(mouseinfo[f'mouse_{mouse}']['seg_1'][1]-lengths[0]))
                
            ## Length 2 loss
            uc,lc = mouseinfo[f'mouse_{mouse}']['seg_2']
            if lc<=lengths[1]<=uc:pass
            else:lengthloss_2 += min(abs(mouseinfo[f'mouse_{mouse}']['seg_2'][0]-lengths[1]),\
                           abs(mouseinfo[f'mouse_{mouse}']['seg_2'][1]-lengths[1]))
            
            ## Angle loss
            uc,lc = mouseinfo[f'mouse_{mouse}']['angle']
            print("Angle",angle)
            if torch.isnan(angle):angle = 0 
            if lc<=angle<=uc:pass
            else:angleloss += min(abs(mouseinfo[f'mouse_{mouse}']['angle'][0]-angle),\
                             abs(mouseinfo[f'mouse_{mouse}']['angle'][1]-angle))
            counter+=1
        loss += lengthloss_1 + lengthloss_2 + angleloss
        
    return loss/counter if counter!=0 else loss



######################################################################
def geodesic_loss(outputs,targets,nodesPresent,args) -> float:
    '''
    Computes the geodesic loss between the predicted and the ground truth frame
    
    Args:
    predicted - (N X 3 X 12 X 2) tensor 
    ground_truth - (N X 3 X 12 X 2) tensor
    
    Returns:
    loss -> float (1X1)
    '''
    ## Computation of the output by sampling the coordinates from the bi-variate gaussian distribution
    numNodes = outputs.shape[1]
    ret_nodes = torch.zeros((args.obs_length + args.pred_length, numNodes, outputs.shape[2],2),requires_grad = True).cuda()
    
    if targets.shape[-1]>2:targets = targets[:,:,:,:2]
    ret_nodes[:args.obs_length, :, :,:] = targets[:args.obs_length].clone()
    for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
        mux, muy, sx, sy, corr = getCoef(outputs[tstep][None,...])
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent[args.obs_length-1])
        # next_x,next_y = torch.zeros((3,12),requires_grad=True),torch.zeros((3,12),requires_grad=True)
        ret_nodes[tstep + 1, :,:, 0] = next_x
        ret_nodes[tstep + 1, :,:, 1] = next_y

    if ret_nodes.shape[1] > 3:
        ret_nodes = ret_nodes.squeeze().reshape(ret_nodes.shape[0],3,12,ret_nodes.shape[-1])
        targets   = targets.squeeze().reshape(targets.shape[0],3,12,targets.shape[-1])
    # Get the Lie-algebra space transformation of both the vectors
    if args.bodyline_only:
        (transformed_pred,_),(transformed_gt,_) = lieSpace(joint_xy(ret_nodes),True),lieSpace(joint_xy(targets),True)
    else:
        (transformed_pred,_),(transformed_gt,_) = lieSpace(ret_nodes),lieSpace(targets)

    nframes,nmice,njoints,_ = transformed_pred.shape

    total_loss = 0
    for frame in range(nframes):
        for mice in range(nmice):
            twist_loss = torch.linalg.norm(transformed_pred[frame,mice,:,:]-transformed_gt[frame,mice,:,:],axis=-1)
            bone_lengths = transformed_gt[frame,mice,:,3]
            coefficients = torch.arange(njoints,0,step=-1).cuda()
            total_loss += torch.sum(twist_loss*bone_lengths*coefficients)
    return total_loss/(nframes*nmice)

#################
# Forward Lie #
##################

def joint_xy(_input,
             joint_list:list = None) -> "tensor":
    '''
    Slice the tensor on the penultimate index based on 6 different keypoints
    Args:
    _input: (N X 3 X 12 X 2)
    
    Returns:
    output: (N X 3 X 6 X 2)
    '''
    if not joint_list: joint_list = [0,3,6,9,10,11]
    return _input[:,:,joint_list,:]

def lieSpace(og_input,
             bodyline_only = False,
             root_joint:list=[3],tolerance = 1e-10) -> "tensor":
    '''
    Map the given input manifold into the lie algebra space
    Args:
    _input -> (N X 3 X 6 X 2) when bodyline_only is set to True
    _input -> (N X 3 X 12 X 2) otherwise
    
    Returns:
    lieTensor -> (N X 3 X 6 X 6) # Screw representation ==> Column Vector: [wx,wy,wz,vx,vy,vz]
    '''
    if not bodyline_only: root_joint = root_joint[0]
    else: root_joint = 3 # Represents the tail base
 
    uncentered_data = og_input.clone()
    _input = og_input.clone()
    _input = _input - _input[:,:,root_joint:root_joint+1,:] # Centering the data with respect to the root joint
    nframes,nmice,njoints,_ = _input.shape
    lieTensor = torch.zeros((nframes,nmice,njoints,6),requires_grad = True).cuda()
    axis_of_rotation = torch.Tensor([0,0,1]).T # z-axis is the axis of rotation
    vhat,uhat = torch.zeros(3).cuda(),torch.zeros(3).cuda()

    for frame in range(nframes):
        for mice in range(nmice):
            ## Add the root joint
            lieTensor[frame,mice,0,3:5] = _input[frame,mice,root_joint,:]     
            ## Calculate the bone length
            if bodyline_only:lengths = _input[frame,mice,1:,:] - _input[frame,mice,:-1,:]
            else: lengths = _input[frame,mice,1:,:] - tensorConcat(_input[frame,mice,:,:]) 
            lieTensor[frame,mice,1:,3] = torch.linalg.norm(lengths,axis=-1)
            ## Axis angle representation
            for joint in range(njoints-2,0,-1):
                vhat[:2] = lengths[joint]/(torch.linalg.norm(lengths[joint],dim=-1)+tolerance)
                if joint == 1: uhat = axis_of_rotation.cuda()
                else:uhat[:2] = lengths[joint-1]/(torch.linalg.norm(lengths[joint-1],dim=-1)+tolerance)                
                lieTensor[frame,mice,joint,:3] = SE2_se2(rotationMatrix(uhat.clone(),vhat.clone()))
    return lieTensor,uncentered_data

def tensorConcat(_input,parent_joints:list = [0,3,6]):
    size = _input[parent_joints,:].shape
    tensortile = torch.tile(_input[parent_joints,:],(3,3)).reshape(-1,*size)[:3,:,:].reshape(-1,size[-1])
    tail_joints = _input[[-3,-2],:]
    return torch.vstack((tensortile,tail_joints))
    
def rotationMatrix(v,u,tolerance = 1e-10):
    '''
    Computes the rotation matrix, SO(3), between given the two vectors connected in a head-tail fashion
    Args:
    v,u -> (1 X 3) vectors
    
    Returns:
    rotmat -> ( 3 X 3 ) matrix
    
    '''
    #Cross product Computation and finding the axis-angle
    w = torch.cross(v,u)
    what = w/(torch.linalg.norm(w)+tolerance)
    dot = torch.clip(torch.dot(u,v),-1,1)
    angle = torch.arccos(dot)

    # Convert aa representation into rotation matrix
    k = skew(what)
    rotmat = torch.eye(3).cuda() + torch.sin(angle)*k + (1-torch.cos(angle))*(k@k) # Rodrigues formula
    return rotmat

def skew(vector)->"skew symmetric matrix":
    '''
    Returns the skew symmetric matrix given a vector 
    '''
    kx,ky,kz = vector
    skew = torch.zeros((3,3)).cuda()
    '''
    || 0   | -kz | ky ||  
    || kz  |  0  |-kx ||
    || -ky |  kx |  0 ||
    '''
    skew[0][1], skew[1][0] = -kz,kz
    skew[0][2], skew[2][0] =  ky,-ky
    skew[1][2], skew[2][1] = -kx,kx
    return skew

def SE2_se2(matrix):
    '''
    Lie algebra space - logarithm mapping
    '''
    
    solution = torch.zeros(3).cuda()
    r21,r11 = matrix[1,0],matrix[0,0]
    solution[2] = torch.atan2(r21,r11)
    return solution

#################
# Inverse Lie #
##################

def inverse(matrix):
    '''
    Map the vector from the lie algebra space into the original manifold (Exp mapping)
    Args:
    matrix -> (N X 3 X 6 X 6)
    
    Returns:
    output -> (N X 3 X 6 X 2)
    '''
    nframes,nmice,njoints,_ = matrix.shape
    output = np.zeros((nframes,nmice,njoints,2))
    
    for frame in range(nframes):
        for mice in range(nmice):
            output[frame,mice,:,:] = computeInverse(matrix[frame,mice,:,:])[...,:-1]
    return output

def computeInverse(matrix):
    njoints,_ = matrix.shape
    joints,prev = np.zeros((njoints,3)),np.zeros((njoints,3,3))
    for joint in range(njoints):
        if joint == 0:prev[joint,:,:] = getRotationMatrix(matrix[joint])
        else:prev[joint,:,:] = prev[joint-1,:,:]@getRotationMatrix(matrix[joint])       
    for joint in range(njoints):
        joints[joint,:] = prev[joint,:,:]@(np.array([0,0,1]))
    return joints

def getRotationMatrix(vector):
    from scipy.spatial.transform import Rotation as R
    
    rotation,trans = vector[:3],vector[3:]
    rotmat = R.from_rotvec(rotation).as_matrix()
    rotmat[:,-1] = trans
    rotmat[-1,-1] = 1
    return rotmat


######################################################################