'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import time

import torch
from torch.autograd import Variable

from utils_mouse import DataLoaderMouseOld, DataLoaderMouseNew
from st_graph import ST_GRAPH
from model import SRNN
import numpy as np
from criterion import Gaussian2DLikelihood, geodesic_loss
import logging
from torch.utils.tensorboard import SummaryWriter
import warnings 
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=2,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=5,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=20,
                        help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted sequence length')
    
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=0,
                        help='The dataset index to be left out in training')
    
    parser.add_argument(
        "--body_keypoint", type=str, help="Selector variable for keypoints of the mouse", default=None,
    )
    parser.add_argument(
        "--train_frac", type=float, help="Fraction of the total sequences to be used for training", default=0.0625,
    )

    parser.add_argument('--train', type=bool, default=False,
                        help='Flag to indicate the usage of a slice of training data for testing')

    parser.add_argument(
        "--save_dir", type=str, help="Specify the directory to save the models", default=None,
    )
    parser.add_argument(
        "--gl", action="store_true", help="Boolean variable indicating the necessity of a Geodesic loss implementation", default=False,
    )
    parser.add_argument(
        "--bodyline_only", action="store_true", help="Boolean variable indicating the necessity of a utilising all the keypoints for a geodesic loss implementation", default=False,
    )
    parser.add_argument(
        "--all_kp", action="store_true", help="Boolean variable indicating the necessity of a utilising all the keypoints for ST-graph construction", default=False,
    )
    parser.add_argument(
        "--exp_name", type=str, help="Name of the run", default="G1_old_gl6",
    )

    args = parser.parse_args()
    train(args)

def train(args):
    if args.exp_name[3:6] == "new":
        dataloader = DataLoaderMouseNew(args.batch_size, args.seq_length + 1, forcePreProcess=True, body_keypoint=args.body_keypoint,\
                                        train_frac=args.train_frac,train_data=args.train,frame_reduction_mode = "Slice", \
                                        allkp=args.all_kp, frames = 120)
    else: 
        dataloader = DataLoaderMouseOld(args.batch_size, args.seq_length + 1, forcePreProcess=True, body_keypoint=args.body_keypoint,\
                                        train_frac=args.train_frac, train_data = args.train, frame_reduction_mode = "Slice", \
                                        allkp = args.all_kp, frames = 150)
    

    ## Tensorboard Callback for data logging
    writer = SummaryWriter(f'./logs/{args.exp_name}')

    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, args.seq_length + 1,body_keypoint=args.body_keypoint)
    save_directory = os.path.join(args.save_dir,str(args.leaveDataset),'save_attention')

    if not os.path.isdir(save_directory):
        print("Creating directory")
        os.makedirs(save_directory)
        
    # Open the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(save_directory, 'indices.pkl'), 'wb') as f:
        print('Dumping OOB Indices')
        pickle.dump(dataloader.oob_indices,f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        print(os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar'))
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    # Initialize net
    net = SRNN(args)
    net.cuda()
    print(torch.cuda.get_device_name())
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)
    obs_length = args.seq_length - args.pred_length
    learning_rate = args.learning_rate

    logging.basicConfig(level=logging.INFO,\
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    logger.info('Training the Network with the following parameters')
    logger.info('--------------------------------')
    logger.info(f'Body Keypoints Used: {args.body_keypoint if args.body_keypoint is not None else 12}')
    logger.info('--------------------------------')
    logger.info(f'Number of frames skipped: {dataloader.skip}')
    logger.info('--------------------------------')
    logger.info('Network Info')
    logger.info('--------------------------------')
    logger.info(f'Using Geodesic Loss') if args.gl == True else logger.info(f'No Geodesic Loss')
    logger.info('--------------------------------')
    logger.info(net)
    logger.info({"experiment":args.exp_name,
                                    "learning_rate": args.learning_rate,
                                    "epochs": args.num_epochs,
                                    "batch size":args.batch_size,
                                    "sequence length":args.seq_length,
                                    "prediction length": args.pred_length,
                                    "dropout":args.dropout,
                                    "keypoints":args.body_keypoint,
                                    "fraction":args.train_frac,
                                    "GL Type 6/12": (args.bodyline_only * 6 + (1-args.bodyline_only) * 12),
                                    "Graph Type":args.exp_name[0:2],
                                    "Dataset Type": args.exp_name[3:6] 
                                     })

    # writer.add_hparams(hparam_dict={"experiment":args.exp_name,
    #                                 "learning_rate": args.learning_rate,
    #                                 "epochs": args.num_epochs,
    #                                 "batch size":args.batch_size,
    #                                 "sequence length":args.seq_length,
    #                                 "prediction length": args.pred_length,
    #                                 "dropout":args.dropout,
    #                                 "keypoints":args.body_keypoint,
    #                                 "fraction":args.train_frac,       
    #                                 "GL Type 6/12": (args.bodyline_only * 6 + (1-args.bodyline_only) * 12),
    #                                 "Graph Type":args.exp_name[0:2],
    #                                 "Dataset Type": args.exp_name[3:6] 
    #                                  },
    #                     metric_dict={}
    #                                  )
    best_val_loss = 100
    best_epoch = 0
    # Training
    for epoch in range(args.num_epochs):
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch,loss_geo,loss_gauss  = 0,0,0

        # For each batch
        print(dataloader.num_batches)
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, _, _, d = dataloader.next_batch(randomUpdate=False)
            # Loss for this batch
            loss_batch = 0
            loss_gauss_batch,loss_joint_batch = 0,0
            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                # Construct the graph for the current sequence
                stgraph.readGraph([x[sequence]])
                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]

                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, nodes.shape[2],args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, nodes.shape[2],args.human_human_edge_rnn_size)).cuda()

                cell_states_node_RNNs = Variable(torch.zeros(numNodes, nodes.shape[2],args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, nodes.shape[2],args.human_human_edge_rnn_size)).cuda()
                
                # Zero out the gradients
                net.zero_grad()
                optimizer.zero_grad()
                outputs, _, _, _, _, attn_weights = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)
 
                gaussian_loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                joint_loss = geodesic_loss(outputs,nodes[1:],nodesPresent[1:],args) if args.gl else torch.tensor(0.0)
                loss = gaussian_loss + joint_loss
                loss_gauss_batch+=gaussian_loss
                loss_joint_batch+=joint_loss
                loss_batch += loss.data
                # Compute gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

                # Reset the stgraph
                stgraph.reset()
            end = time.time()

            loss_batch = loss_batch / dataloader.batch_size
            loss_gauss_batch,loss_joint_batch = loss_gauss_batch/ dataloader.batch_size, loss_joint_batch/ dataloader.batch_size

            loss_epoch += loss_batch
            loss_gauss += loss_gauss_batch
            loss_geo   += loss_joint_batch
            print("Overall Training Loss per batch",loss_batch,batch)
        
        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches
        loss_geo /= dataloader.num_batches
        loss_gauss /= dataloader.num_batches
        # Log it: Tensorboard Logging
        # writer.add_scalar("Overall Training Loss per epoch",loss_epoch,epoch)
        # writer.add_scalar("Training Geodesic Loss per epoch",loss_geo,epoch)
        # writer.add_scalar("Training Gaussian Loss per epoch", loss_gauss,epoch)
        
        print("Overall Training Loss per epoch",loss_epoch,epoch)
        print("Training Geodesic Loss per epoch",loss_geo,epoch)
        print("Training Gaussian Loss per epoch", loss_gauss,epoch)
        
        ## Log attention
        # tensor = torch.from_numpy(attn_weights[-1][0][0].reshape(2,12,12))
        # writer.add_tensor('Attention values with respect to mice 0 per epoch',tensor,epoch)

        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch,loss_gauss,loss_geo = 0, 0, 0

        for batch in range(dataloader.valid_num_batches):
            # Get batch data
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

            # Loss for this batch
            loss_batch = 0
            loss_gauss_batch,loss_joint_batch = 0,0

            for sequence in range(dataloader.batch_size):
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()
                

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, nodes.shape[2],args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, nodes.shape[2],args.human_human_edge_rnn_size)).cuda()
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, nodes.shape[2],args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, nodes.shape[2],args.human_human_edge_rnn_size)).cuda()


                outputs, _, _, _, _, attn_weights = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                             hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                             cell_states_node_RNNs, cell_states_edge_RNNs)


                # Compute loss
                gaussian_loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                joint_loss = geodesic_loss(outputs,nodes[1:],nodesPresent[1:],args) if args.gl else torch.tensor(0.0)
                loss = gaussian_loss + joint_loss
                loss_gauss_batch+=gaussian_loss
                loss_joint_batch+=joint_loss
                
                loss = gaussian_loss + joint_loss
                loss_batch += loss.data

                # Reset the stgraph
                stgraph.reset()

            loss_batch = loss_batch / dataloader.batch_size
            loss_gauss_batch,loss_joint_batch = loss_gauss_batch/ dataloader.batch_size, loss_joint_batch/ dataloader.batch_size
            loss_epoch += loss_batch
            loss_gauss += loss_gauss_batch
            loss_geo += loss_joint_batch

        loss_epoch = loss_epoch / dataloader.valid_num_batches
        loss_geo = loss_geo / dataloader.valid_num_batches
        loss_gauss = loss_gauss / dataloader.valid_num_batches

        # if args.wandb:wandb.log({"Validation Loss":loss_batch,"Gaussian loss": loss_gauss_batch, "Joint loss": loss_joint_batch})
        # if args.wandb and epoch%20 == 0:wandb.log({"Table":wandb.Table(dataframe=df)})
        # writer.add_scalar("Overall Validation Loss per epoch",loss_epoch,epoch)
        # writer.add_scalar("Validation Geodesic Loss per epoch",loss_geo,epoch)
        # writer.add_scalar("Validation Gaussian Loss per epoch", loss_gauss,epoch)
        print("Overall Validation Loss per epoch",loss_epoch,epoch)
        print("Validation Geodesic Loss per epoch",loss_geo,epoch)
        print("Validation Gaussian Loss per epoch", loss_gauss,epoch)

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch

        # Save the model after each epoch
        if epoch%5==0:
            logger.info(f'Saving model @ {checkpoint_path(epoch)}')
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch))

if __name__ == '__main__':
    main()
