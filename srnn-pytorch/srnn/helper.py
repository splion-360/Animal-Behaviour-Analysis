'''
Helper functions for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 3rd April 2017
'''
import numpy as np
import torch
from torch.autograd import Variable

def getVector(pos_list,body=True):
    '''
    Gets the vector pointing from second element to first element
    params:
    pos_list : A list of size two containing two (x, y) positions
    '''
    if body:
        pos_i = pos_list[0]
        pos_j = pos_list[1]
        return np.array(pos_i) - np.array(pos_j)
    else:
        pos_i = np.array(pos_list[0]).transpose((1,0))
        pos_j = np.array(pos_list[1]).transpose((1,0))

        lieTensor_i,_ = lieSpace(torch.from_numpy(pos_i[None,None,:,:]),False)
        lieTensor_j,_ = lieSpace(torch.from_numpy(pos_j[None,None,:,:]),False)
        
        angle_i = lieTensor_i.squeeze()[:,2].detach().cpu().numpy()
        angle_j = lieTensor_j.squeeze()[:,2].detach().cpu().numpy()
        angle = np.vstack((angle_i,angle_j)).T
        
        return np.hstack((np.array(pos_i) - np.array(pos_j),angle))
        
        #return np.array(pos_i) - np.array(pos_j)
        # '''
        # New technique
        
        # '''
        # # return (pos_i[:,None,:]-pos_j[None,:,:]).reshape(-1,2)
        
        


def getMagnitudeAndDirection(*args):
    '''
    Gets the magnitude and direction of the vector corresponding to positions
    params:
    args: Can be a list of two positions or the two positions themselves (variable-length argument)
    '''
    if len(args) == 1:
        pos_list = args[0]
        pos_i = pos_list[0]
        pos_j = pos_list[1]

        vector = np.array(pos_i) - np.array(pos_j)
        magnitude = np.linalg.norm(vector)
        if abs(magnitude) > 1e-4:
            direction = vector / magnitude
        else:
            direction = vector
        return [magnitude] + direction.tolist()

    elif len(args) == 2:
        pos_i = args[0]
        pos_j = args[1]

        ret = torch.zeros(3)
        vector = pos_i - pos_j
        magnitude = torch.norm(vector)
        if abs(magnitude) > 1e-4:
            direction = vector / magnitude
        else:
            direction = vector

        ret[0] = magnitude
        ret[1:3] = direction
        return ret

    else:
        raise NotImplementedError('getMagnitudeAndDirection: Function signature incorrect')


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, :,0], outputs[:, :, :,1], outputs[:, :, :,2], outputs[:, :, :,3], outputs[:, :, :,4]

    # Exponential to get a positive value for std dev
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    # tanh to get a value between [-1, 1] for correlation
    corr = torch.tanh(corr)

    return mux, muy, sx, sy, corr

def reparametrise(mean,cov,nsamples=2):
    mean,cov = np.array(mean),np.array(cov)
    epsilon = np.random.randn(nsamples)
    coords  = (cov@epsilon)+mean
    return coords

def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent):
    '''
    Returns samples from 2D Gaussian defined by the parameters
    params:
    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation
    nodesPresent : a list of nodeIDs present in the frame

    returns:
    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :].cpu(), muy[0, :].cpu(), sx[0, :].cpu(), sy[0, :].cpu(), corr[0, :].cpu()
    numNodes = mux.size()[1]

    next_x = torch.zeros((numNodes,o_mux.shape[1]))
    next_y = torch.zeros((numNodes,o_mux.shape[1]))

    for node in range(numNodes):
        for keypoint in range(o_mux.shape[1]):
            if node not in nodesPresent:
                continue
            mean = [o_mux[node,keypoint], o_muy[node,keypoint]]
            cov = [[o_sx[node,keypoint]*o_sx[node,keypoint], o_corr[node,keypoint]*o_sx[node,keypoint]*o_sy[node,keypoint]], \
                   [o_corr[node,keypoint]*o_sx[node,keypoint]*o_sy[node,keypoint], o_sy[node,keypoint]*o_sy[node,keypoint]]]

            # next_values = np.random.multivariate_normal(mean, cov, 1)
            # next_x[node,keypoint] = next_values[0][0]
            # next_y[node,keypoint] = next_values[0][1]
            
            next_values = reparametrise(mean, cov)
            next_x[node,keypoint] = next_values[0]
            next_y[node,keypoint] = next_values[1]

    return next_x, next_y


def compute_edges(nodes, tstep, edgesPresent):
    '''
    Computes new edgeFeatures at test time
    params:
    nodes : A tensor of shape seq_length x numNodes x 2
    Contains the x, y positions of the nodes (might be incomplete for later time steps)
    tstep : The time-step at which we need to compute edges
    edgesPresent : A list of tuples
    Each tuple has the (nodeID_a, nodeID_b) pair that represents the edge
    (Will have both temporal and spatial edges)

    returns:
    edges : A tensor of shape numNodes x numNodes x 2
    Contains vectors representing the edges
    '''
    numNodes = nodes.size()[1]
    edges = (torch.zeros(numNodes * numNodes, 2)).cuda()
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]

        if nodeID_a == nodeID_b:
            # Temporal edge
            pos_a = nodes[tstep - 1, nodeID_a, :]
            pos_b = nodes[tstep, nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            pos_a = nodes[tstep, nodeID_a, :]
            pos_b = nodes[tstep, nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent):
    '''
    Computes average displacement error
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length).cuda()
    counter = 0

    for tstep in range(pred_length):
        counter = 0
        for nodeID in assumedNodesPresent:

            if nodeID not in trueNodesPresent[tstep]:
                continue

            pred_pos = ret_nodes[tstep, nodeID, :,:]
            true_pos = nodes[tstep, nodeID, :,:]
            
            error[tstep] += torch.norm(pred_pos - true_pos,p=2)/true_pos.shape[0]
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)


def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent):
    '''
    Computes final displacement error
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent:
     
        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID,:, :]
        true_pos = nodes[tstep, nodeID, :,:]

        error += torch.norm(pred_pos - true_pos, p=2)/true_pos.shape[0]
        counter += 1

    if counter != 0:
        error = error / counter

    return error


def sample_gaussian_2d_batch(outputs, nodesPresent, edgesPresent, nodes_prev_tstep):
    mux, muy, sx, sy, corr = getCoef_train(outputs)

    next_x, next_y = sample_gaussian_2d_train(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent)

    nodes = torch.zeros(outputs.size()[0], 2)
    nodes[:, 0] = next_x
    nodes[:, 1] = next_y

    nodes = Variable(nodes.cuda())

    edges = compute_edges_train(nodes, edgesPresent, nodes_prev_tstep)

    return nodes, edges


def compute_edges_train(nodes, edgesPresent, nodes_prev_tstep):
    numNodes = nodes.size()[0]
    edges = Variable((torch.zeros(numNodes * numNodes, 2)).cuda())
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]

        if nodeID_a == nodeID_b:
            # Temporal edge
            pos_a = nodes_prev_tstep[nodeID_a, :]
            pos_b = nodes[nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            pos_a = nodes[nodeID_a, :]
            pos_b = nodes[nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges


def getCoef_train(outputs):
    mux, muy, sx, sy, corr = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d_train(mux, muy, sx, sy, corr, nodesPresent):
    o_mux, o_muy, o_sx, o_sy, o_corr = mux, muy, sx, sy, corr

    numNodes = mux.size()[0]

    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    for node in range(numNodes):
        if node not in nodesPresent:
            continue
        mean = [o_mux[node], o_muy[node]]

        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y

#################
# Forward Lie #
##################
def lieSpace(og_input,
             bodyline_only = True,
             root_joint:list=[3],tolerance = 1e-10) -> "tensor":
    '''
    Map the given input manifold into the lie algebra space
    Args:
    _input -> (N X 3 X 6 X 2) when bodyline_only is set to True
    _input -> (N X 3 X 12 X 2) otherwise
    
    Returns:
    lieTensor -> (N X 3 X 6 X 6) # Screw representation ==> Column Vector: [wx,wy,wz,vx,vy,vz]
    '''
    if bodyline_only: root_joint = root_joint[0]
    else: root_joint = 9 # Represents the tail base

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