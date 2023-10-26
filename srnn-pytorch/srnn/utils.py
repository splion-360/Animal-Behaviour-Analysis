'''
Utils script for the structural RNN implementation
Handles processing the input and target data in batches and sequences

Author : Anirudh Vemula
Date : 15th March 2017
'''

import os
import pickle
import numpy as np
import random
from tqdm.auto import tqdm
import copy

class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0, 1, 2, 3, 4,5], forcePreProcess=False, infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                          './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                          './data/ucy/univ']
        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.test_data_dirs = [self.data_dirs[x] for x in range(5) if x not in datasets]
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Validation arguments
        self.val_fraction = 0.2

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)


    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for ind_directory, directory in enumerate(data_dirs):
            # define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')
            
            file_path = os.path.join(directory, 'pixel_pos_interpolate.csv')

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            numFrames = len(frameList)
            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])

            # if directory == './data/eth/univ':
            #    skip = 6
            # else:
            #    skip = 10
            skip = 10

            for ind, frame in enumerate(frameList):

                ## NOTE CHANGE
                if ind % skip != 0:
                    # Skip every n frames
                    # print(counter)
                    continue

                
                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()
           

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y])

                if (ind > numFrames * self.val_fraction) or (self.infer):
                    # At inference time, no validation data
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index].append(np.array(pedsWithPos))
                    # print('Visited..',times)
                    # print('Index',ind)
                    # print('Counter',counter)
                    # all_frame.append(ind)
                    # times+=1
                else:
                    valid_frame_data[dataset_index].append(np.array(pedsWithPos))

            # print('All frame',all_frame)
            # print('Valid frame',valid_frame)
            dataset_index += 1

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        
        # print(len(self.data))
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print('Training data from dataset {} : {}'.format(dataset, len(all_frame_data)))
            print('Validation data from dataset {} : {}'.format(dataset, len(valid_frame_data)))
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        print('Total number of training batches: {}'.format(self.num_batches * 2))
        print('Total number of validation batches: {}'.format(self.valid_num_batches))
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2
        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Frame data
        frame_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            

            frame_data = self.data[self.dataset_pointer]

            frame_ids = self.frameList[self.dataset_pointer]

            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]
                seq_frame_ids = frame_ids[idx:idx+self.seq_length]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                frame_batch.append(seq_frame_ids)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, frame_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0    

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

class DataLoaderMouse():

    def __init__(self, batch_size=50, seq_length=5,forcePreProcess=False, infer=False, body_keypoint:list = None,size=2,val_fraction=0.2):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        #Read the mouse triplets data
        # print('Mouse data is being processed...')
        # self.data = np.load('./mousedata/user_train.npy',allow_pickle=True).tolist()
        # sequences = list(self.data['sequences'].keys())
        # # self.sequences = list(self.data['sequences'].keys())
        # indices = np.random.choice(len(sequences),size,replace=False)
        # self.oob_indices = list(set(range(len(self.data['sequences'].keys()))) - set(indices))
        # self.sequences = [sequences[cur] for cur in indices]
        # if body_keypoint is not None:
        #     self.body_keypoint = [int(kp) for kp in body_keypoint.split(',')]
        # else: self.body_keypoint = body_keypoint
        # self.infer = infer

        self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                          './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                          './data/ucy/univ']
        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.test_data_dirs = [self.data_dirs[x] for x in range(5) if x not in datasets]
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Validation arguments
        self.val_fraction = 0.2

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Validation arguments
        self.val_fraction = val_fraction
        
        self.data_dir = './mousedata'
        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)
    

    def frame_preprocess(self,data_file,thresh=100):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        all_frame_data = []
        valid_frame_data = []
        frameList_data = []
        numMouse_data = []
        sequence_index = 0 #dataset_index = 0
        skip = 2

        # For each sequence
        for index, seq_id in enumerate(tqdm(self.sequences,desc='Processing Sequence ......')):
            current_seq_data = self.data['sequences'][seq_id]['keypoints']
            frameList = np.arange(0,current_seq_data.shape[0])
            numFrames = len(frameList)
            frameList_data.append(frameList)
            numMouse_data.append([])
            all_frame_data.append([])
            valid_frame_data.append([])

            curr_frame = np.zeros((3,12,2))
            for ind, frame in enumerate(frameList):
                miceWithPos = []
                if ind % skip != 0:
                    continue
                if self.body_keypoint is not None:
                    miceInFrame = current_seq_data[frame,:,self.body_keypoint,:].transpose((1,0,2))
                else:
                    miceInFrame = current_seq_data[frame,:,:]

                # Normalise the x and y coordinates between -1 and 1
                # NOTE: These lines are not to be changed (data normalisation code)
                # coords = np.max(miceInFrame,0)
                # if np.prod(coords) == 0:coords = np.where(coords==0,1,coords)
                # miceInFrame = ((miceInFrame/coords)*2)- 1

                # if not (abs(miceInFrame-curr_frame)>thresh).any():
                #     print('In')
                #     continue
                # curr_frame = miceInFrame

                numMouse_data[sequence_index].append(current_seq_data.shape[0])
                miceWithPos = []

                # For each ped in the current frame
                for mice in range(miceInFrame.shape[0]):
                    if len(miceInFrame.shape)>2:
                        current_x = miceInFrame[mice,:,1].tolist()
                        current_y = miceInFrame[mice,:,0].tolist()
                        mousearr = np.ones(len(current_x))*mice
                    else:
                        current_x = miceInFrame[mice,1]
                        current_y = miceInFrame[mice,0]
                        mousearr = mice
                    miceWithPos.append([mousearr, current_x, current_y])
                if len(miceInFrame.shape)>2:
                    miceWithPos = np.array(miceWithPos).transpose((0,2,1))
                else:miceWithPos = np.array(miceWithPos)

                if (ind > numFrames * self.val_fraction) or (self.infer):
                    # At inference time, no validation data
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[sequence_index].append(miceWithPos)
                else:
                    valid_frame_data[sequence_index].append(miceWithPos)
                
            sequence_index += 1
            
        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numMouse_data, valid_frame_data,self.oob_indices), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        self.oob_indices = self.raw_data[4]
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            # print('Training data from dataset {} : {}'.format(dataset, len(all_frame_data)))
            # print('Validation data from dataset {} : {}'.format(dataset, len(valid_frame_data)))
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        print('Total number of training batches: {}'.format(self.num_batches * 2))
        print('Total number of validation batches: {}'.format(self.valid_num_batches))
        print('Total Out of Box indices: {}'.format(len(self.oob_indices)))
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2
        # self.valid_num_batches = self.valid_num_batches * 2

                
    def fill_holes(self,data):
        clean_data = copy.deepcopy(data)
        clean_data = np.array(clean_data)
        for m in range(3):
            holes = np.where(clean_data[0,m,:,1]==0)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:,m,h,1]!=0)
                if(sub and sub[0].size > 0):
                    clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
                else:
                    return np.empty((0))

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,1]==0)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        return [arr for arr in clean_data]

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Frame data
        frame_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            frame_ids = self.frameList[self.dataset_pointer]

            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = self.fill_holes(frame_data[idx:idx+self.seq_length])
                seq_target_frame_data = self.fill_holes(frame_data[idx+1:idx+self.seq_length+1])
                seq_frame_ids = frame_ids[idx:idx+self.seq_length]

                # Number of unique peds in this sequence of frames
            
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                frame_batch.append(seq_frame_ids)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, frame_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0    

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0