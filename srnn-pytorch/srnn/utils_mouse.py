
import os
import pickle
import numpy as np
import random
from tqdm.auto import tqdm
import copy

max_x,max_y = 850,850

class DataLoaderMouseNew():
    
    def __init__(self, batch_size=50, seq_length=5,forcePreProcess=False, infer=False, body_keypoint:list = None,\
                 train_frac = 0.0625,val_fraction=0.5,frame_reduction_mode = 'Slice',normalise = True,frames = 150,\
                 train_data = False, oob_indices=None,seed = 903809718,allkp = False):
        
        '''
        Initialiser function for the New DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        body_keypoint: The indices of the keypoints to be used
        train_frac: Fraction of the total sequences to be used for training
        val_fraction : Fraction of sequences to be used for validation
        '''
        
        #Read the mouse triplets data
        self.seed = seed
        self.allkp = allkp
        np.random.seed(self.seed)

        self.data = np.load('./mousedata/newdataset.npy',allow_pickle=True).tolist()
        sequences = list(self.data['sequences'].keys())

        size = int(len(sequences)*train_frac)
        indices = np.random.choice(len(sequences),size,replace=False)
        self.oob_indices = list(set(range(len(self.data['sequences'].keys()))) - set(indices))
        self.sequences = [sequences[cur] for cur in indices]
        
        if not train_data:
            self.sequences = [sequences[cur] for i,cur in enumerate(range(len(sequences))) if cur in oob_indices and i<=50]

        if body_keypoint is not None:
            self.body_keypoint = [int(kp) for kp in body_keypoint.split(',')]
        else: 
            self.body_keypoint = body_keypoint
        
        self.infer = infer
        self.normalise = normalise
        self.skip_frames_by_static = frame_reduction_mode

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.frame = frames
        # Validation arguments
        self.val_fraction = val_fraction
        
        self.data_dir = './mousedata'
        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "new_trajectories.cpkl")

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

    @staticmethod
    def revert_seq(sequence):
        sequence[...,0],sequence[...,1] = ((sequence[...,0]+1)*max_x)//2, ((sequence[...,1]+1)*max_y)//2
        return sequence
    def convert_seq(self,sequence):
        sequence[...,1],sequence[...,2] = (sequence[...,1]/max_x)*2 - 1, (sequence[...,2]/max_y)*2 - 1
        return sequence
    
    def frame_preprocess(self,data_file,thresh=100,skip = 1):
        
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
        self.thresh = thresh
        self.skip = skip

        # For each sequence
        seed = [int(f"903809{i}") for i in range(len(self.sequences))]

        for index, seq_id in enumerate(tqdm(self.sequences,desc='Processing Sequence ......')):
            
            np.random.seed(seed[index])
            current_seq_data = self.data['sequences'][seq_id]['keypoints']
            if len(current_seq_data) < self.frame:
                continue
            
            indices = np.arange(0,current_seq_data.shape[0]-self.frame,20)

            randomidx = np.random.choice(indices)
            current_seq_data = current_seq_data[index:index+self.frame]
            frameList = np.arange(0,current_seq_data.shape[0])


            if len(frameList)<self.frame:
                continue
           
            numFrames = len(frameList) if self.skip_frames_by_static != 'Slice' else self.frame
            
            frameList_data.append(frameList)
            numMouse_data.append([])
            all_frame_data.append([])
            valid_frame_data.append([])

            slice_ind = 0

            for ind, frame in enumerate(frameList):
                miceWithPos = []

                if self.body_keypoint is not None:miceInFrame = current_seq_data[frame,:,self.body_keypoint,:].transpose((1,0,2))

                else:miceInFrame = current_seq_data[frame,:,:]           


                if self.skip_frames_by_static == 'Slice':
                    if ind < randomidx:
                        continue
                    else:
                        if slice_ind%self.frame == 0 and slice_ind != 0:
                            break
                        slice_ind+=1

                elif self.skip_frames_by_static == 'Skip':
                    if ind % skip != 0:
                        continue

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

            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)

        if self.skip_frames_by_static == 'Static':print(f'Skip Frame Mode: Static,  Threshold: {self.thresh}')
        elif self.skip_frames_by_static == 'Slice':print(f'Skip Frame Mode: Slice, Max Frames: {self.frame}')
        else:print(f'Skip Frame Mode: Naive Skip, Frequency: {self.skip} frames')
        
        if self.normalise:print('Normalisation is enabled')
        else:print('Normalisation is disabled')

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

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,1]==0)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        
        if self.normalise:
            if self.allkp:
                return [self.convert_seq(arr).reshape(-1,arr.shape[-1])[:,None,:] for arr in clean_data] # Considering all the interactions that happen spatially (Reduntant information 
            #are also considered)
            else:return [self.convert_seq(arr) for arr in clean_data]
        
        else:return [arr for arr in clean_data]

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



##########################################################################################################################


class DataLoaderMouseOld():
    
    def __init__(self, batch_size=50, seq_length=5,forcePreProcess=False, infer=False, body_keypoint:list = None,\
                 train_frac = 0.0625,val_fraction=0.3,frame_reduction_mode = 'Slice',normalise = True,frames = 150,\
                 train_data = False, oob_indices=None,seed = 903809718, allkp = False):
        
        '''
        Initialiser function for the Old DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        body_keypoint: The indices of the keypoints to be used
        train_frac: Fraction of the total sequences to be used for training
        val_fraction : Fraction of sequences to be used for validation
        '''
        
        #Read the mouse triplets data
        self.seed = seed
        self.allkp = allkp
        np.random.seed(self.seed)

        self.data = np.load('./mousedata/user_train.npy',allow_pickle=True).tolist()
        sequences = list(self.data['sequences'].keys())

        size = int(len(sequences)*train_frac)
        indices = np.random.choice(len(sequences),size,replace=False)
        self.oob_indices = list(set(range(len(self.data['sequences'].keys()))) - set(indices))
        self.sequences = [sequences[cur] for cur in indices]
        
        if not train_data:
            self.sequences = [sequences[cur] for i,cur in enumerate(range(len(sequences))) if cur in oob_indices and i<=50]

        if body_keypoint is not None:
            self.body_keypoint = [int(kp) for kp in body_keypoint.split(',')]
        else: 
            self.body_keypoint = body_keypoint
        
        self.infer = infer
        self.normalise = normalise
        self.skip_frames_by_static = frame_reduction_mode

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.frame = frames
        # Validation arguments
        self.val_fraction = val_fraction
        
        self.data_dir = './mousedata'
        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "old_trajectories.cpkl")

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

    @staticmethod
    def revert_seq(sequence):
        sequence[...,0],sequence[...,1] = ((sequence[...,0]+1)*max_x)//2, ((sequence[...,1]+1)*max_y)//2
        return sequence
    def convert_seq(self,sequence):
        sequence[...,1],sequence[...,2] = (sequence[...,1]/max_x)*2 - 1, (sequence[...,2]/max_y)*2 - 1
        return sequence
    
    def frame_preprocess(self,data_file,thresh=100,skip = 1):
        
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
        self.thresh = thresh
        self.skip = skip

        # For each sequence
        seed = [int(f"903809{i}") for i in range(len(self.sequences))]

        for index, seq_id in enumerate(tqdm(self.sequences,desc='Processing Sequence ......')):
            np.random.seed(seed[index])
            current_seq_data = self.data['sequences'][seq_id]['keypoints']
            frameList = np.arange(0,current_seq_data.shape[0])
            numFrames = len(frameList) if self.skip_frames_by_static != 'Slice' else self.frame
            frameList_data.append(frameList)
            numMouse_data.append([])
            all_frame_data.append([])
            valid_frame_data.append([])


            for ind, frame in enumerate(frameList):
                miceWithPos = []

                if self.body_keypoint is not None:
                    miceInFrame = current_seq_data[frame,:,self.body_keypoint,:].transpose((1,0,2))

                else:
                    miceInFrame = current_seq_data[frame,:,:]           

                if self.skip_frames_by_static == 'Slice':
                    if ind%self.frame == 0 and ind != 0:
                        break
   

                elif self.skip_frames_by_static == 'Skip':
                    if ind % skip != 0:
                        continue

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
                else:
                    miceWithPos = np.array(miceWithPos)
                
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

            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)

        if self.skip_frames_by_static == 'Static':print(f'Skip Frame Mode: Static,  Threshold: {self.thresh}')
        elif self.skip_frames_by_static == 'Slice':print(f'Skip Frame Mode: Slice, Max Frames: {self.frame}')
        else:print(f'Skip Frame Mode: Naive Skip, Frequency: {self.skip} frames')
        
        if self.normalise:print('Normalisation is enabled')
        else:print('Normalisation is disabled')

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

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,1]==0)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        
        if self.normalise:
            if self.allkp:
                return [self.convert_seq(arr).reshape(-1,arr.shape[-1])[:,None,:] for arr in clean_data] # Considering all the interactions that happen spatially (Reduntant information 
            #are also considered)
            else:return [self.convert_seq(arr) for arr in clean_data]
        
        else:return [arr for arr in clean_data]

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