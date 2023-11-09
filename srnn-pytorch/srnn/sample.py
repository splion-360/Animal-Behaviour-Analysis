'''
Test script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 2nd April 2017
'''


import os
import pickle
import argparse
import time

import torch
from torch.autograd import Variable

import numpy as np
from utils_mouse import DataLoaderMouse
from st_graph import ST_GRAPH
from model import SRNN
from helper import getCoef, sample_gaussian_2d, compute_edges, get_mean_error, get_final_error
from criterion import Gaussian2DLikelihood, Gaussian2DLikelihoodInference


def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=15,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=5,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Dataset to be tested on')

    parser.add_argument('--save_dir', type=str, default="save/",
                        help='Directory where the trained model files are saved')
    
    parser.add_argument(
        "--all_kp", action="store_true", help="Boolean variable indicating the necessity of a utilising all the keypoints for ST-graph construction", default=False,
    )
    

    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=99,
                        help='Epoch of model to be loaded')
    parser.add_argument('--body_keypoint', type=str, default=None,
                        help='Body keypoint to be used')
    parser.add_argument('--train', type=bool, default=False,
                        help='Flag to indicate the usage of a slice of training data for testing')

    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    save_directory = sample_args.save_dir
    save_directory += str("0")+'/'
    save_directory += 'save_attention'
    # save_directory = './save/2/mouse 200 slice/'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(save_directory,'./indices.pkl'),'rb') as f:
        index = pickle.load(f)

    # Initialize net
    net = SRNN(saved_args, True)
    net.cuda()

    checkpoint_path = os.path.join(save_directory, 'srnn_model_'+str(sample_args.epoch)+'.tar')

    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at {}'.format(model_epoch))

    dataloader = DataLoaderMouse(1, saved_args.seq_length + 1, forcePreProcess=True,body_keypoint=sample_args.body_keypoint,\
                                 train_frac=saved_args.train_frac,train_data = sample_args.train,oob_indices=index,allkp=saved_args.all_kp)

    dataloader.reset_batch_pointer()

    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, sample_args.pred_length + sample_args.obs_length,body_keypoint=sample_args.body_keypoint)

    results = []

    # Variable to maintain total error
    total_error = 0
    final_error = 0

    for batch in range(dataloader.num_batches):
        start = time.time()

        # Get the next batch
        x, _, frameIDs, d = dataloader.next_batch(randomUpdate=False)

        # Construct ST graph
        stgraph.readGraph(x)

        nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

        # Convert to cuda variables
        nodes = Variable(torch.from_numpy(nodes).float(), volatile=True).cuda()
        edges = Variable(torch.from_numpy(edges).float(), volatile=True).cuda()

        # Separate out the observed part of the trajectory
        obs_nodes, obs_edges, obs_nodesPresent, obs_edgesPresent = nodes[:sample_args.obs_length], edges[:sample_args.obs_length], nodesPresent[:sample_args.obs_length], edgesPresent[:sample_args.obs_length]

        # Sample function
        ret_nodes, ret_attn = sample(obs_nodes, obs_edges, obs_nodesPresent, obs_edgesPresent, sample_args, net, nodes, edges, nodesPresent)

        # Compute mean and final displacement error
        total_error += get_mean_error(ret_nodes[sample_args.obs_length:].data, nodes[sample_args.obs_length:].data, nodesPresent[sample_args.obs_length-1], nodesPresent[sample_args.obs_length:])
        final_error += get_final_error(ret_nodes[sample_args.obs_length:].data, nodes[sample_args.obs_length:].data, nodesPresent[sample_args.obs_length-1], nodesPresent[sample_args.obs_length:])

        end = time.time()

        print('Processed trajectory number : ', batch, 'out of', dataloader.num_batches, 'trajectories in time', end - start)
        # print(ret_nodes[-1])
        # Store results
        results.append((nodes.data.cpu().numpy(), ret_nodes.data.cpu().numpy(), nodesPresent, sample_args.obs_length, ret_attn, frameIDs))

        # Reset the ST graph
        stgraph.reset()

    print('Total mean error of the model is ', total_error / dataloader.num_batches)
    print('Total final error of the model is ', final_error / dataloader.num_batches)

    print('Saving results')
    with open(os.path.join(save_directory, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)


def sample(nodes, edges, nodesPresent, edgesPresent, args, net, true_nodes, true_edges, true_nodesPresent):
    '''
    Sample function
    Parameters
    ==========

    nodes : A tensor of shape obs_length x numNodes x 2
    Each row contains (x, y)

    edges : A tensor of shape obs_length x numNodes x numNodes x 2
    Each row contains the vector representing the edge
    If edge doesn't exist, then the row contains zeros

    nodesPresent : A list of lists, of size obs_length
    Each list contains the nodeIDs that are present in the frame

    edgesPresent : A list of lists, of size obs_length
    Each list contains tuples of nodeIDs that have edges in the frame

    args : Sampling Arguments

    net : The network

    Returns
    =======

    ret_nodes : A tensor of shape (obs_length + pred_length) x numNodes x 2
    Contains the true and predicted positions of all the nodes
    '''
    # Number of nodes
    numNodes = nodes.size()[1]

    # Initialize hidden states for the nodes
    h_nodes = Variable(torch.zeros(numNodes,nodes.shape[2], net.args.human_node_rnn_size), volatile=True).cuda()
    h_edges = Variable(torch.zeros(numNodes * numNodes,nodes.shape[2], net.args.human_human_edge_rnn_size), volatile=True).cuda()
    c_nodes = Variable(torch.zeros(numNodes, nodes.shape[2],net.args.human_node_rnn_size), volatile=True).cuda()
    c_edges = Variable(torch.zeros(numNodes * numNodes, nodes.shape[2],net.args.human_human_edge_rnn_size), volatile=True).cuda()

    # Propagate the observed length of the trajectory
    for tstep in range(args.obs_length-1):
        # Forward prop

        out_obs, h_nodes, h_edges, c_nodes, c_edges, _ = net(nodes[tstep].view(1, numNodes, nodes.shape[2],2), edges[tstep].view(1, numNodes*numNodes,nodes.shape[2], 2), [nodesPresent[tstep]], [edgesPresent[tstep]], h_nodes, h_edges, c_nodes, c_edges)
        # loss_obs = Gaussian2DLikelihood(out_obs, nodes[tstep+1].view(1, numNodes, 2), [nodesPresent[tstep+1]])

    # Initialize the return data structures
    ret_nodes = Variable(torch.zeros(args.obs_length + args.pred_length, numNodes, nodes.shape[2],2), volatile=True).cuda()
    ret_nodes[:args.obs_length, :, :,:] = nodes.clone()

    ret_edges = Variable(torch.zeros((args.obs_length + args.pred_length), numNodes * numNodes,nodes.shape[2], 2), volatile=True).cuda()
    ret_edges[:args.obs_length, :, :,:] = edges.clone()

    ret_attn = []

    # Propagate the predicted length of trajectory (sampling from previous prediction)
    for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
        # TODO Not keeping track of nodes leaving the frame (or new nodes entering the frame, which I don't think we can do anyway)
        # Forward prop
        outputs, h_nodes, h_edges, c_nodes, c_edges, attn_w = net(ret_nodes[tstep].view(1, numNodes,nodes.shape[2], 2), ret_edges[tstep].view(1, numNodes*numNodes,nodes.shape[2], 2),
                                                                  [nodesPresent[args.obs_length-1]], [edgesPresent[args.obs_length-1]], h_nodes, h_edges, c_nodes, c_edges)
        # print(attn_w)
        # loss_pred = Gaussian2DLikelihoodInference(outputs, true_nodes[tstep + 1].view(1, numNodes, 2), nodesPresent[args.obs_length-1], [true_nodesPresent[tstep + 1]])

        # Sample from o
        # mux, ... are tensors of shape 1 x numNodes
        mux, muy, sx, sy, corr = getCoef(outputs)
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent[args.obs_length-1])
        ret_nodes[tstep + 1, :,:, 0] = next_x
        ret_nodes[tstep + 1, :,:, 1] = next_y
        ret_attn.append(attn_w[0])
    return ret_nodes, ret_attn


if __name__ == '__main__':
    main()
