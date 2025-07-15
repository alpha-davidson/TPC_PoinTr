"""
Processes and cuts .h5 files of simulated AT-TPC data
into one numpy array per event for track completion

Author: Ben Wagner
Date Created: 12 Feb 2025
Date Edited:  30 Jun 2025 (Hakan Bora Yavuzkara)
"""


import numpy as np
import os
import h5py
import random
import sys
import mg22_cutting_functions as cf
sys.path.append("/data")
sys.path.append("../")


def scale_data(event, ranges):
    """
    Min/Max scales data based on ranges (logs qs first)

    Parameters:
        data: numpy.ndarray - data to scale
        ranges: dict - contains min and max values of x,y,z, and ln(q)

    Returns:
        scaled: numpy.ndarray - scaled data
    """

    scaled = np.ndarray(event.shape)

    #xs, ys, zs, qs = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
    xs, ys, zs= event[:, 0], event[:, 1], event[:, 2]

    dxs = (xs - ranges['MIN_X']) / (ranges['MAX_X'] - ranges['MIN_X'])
    dys = (ys - ranges['MIN_Y']) / (ranges['MAX_Y'] - ranges['MIN_Y'])
    dzs = (zs - ranges['MIN_Z']) / (ranges['MAX_Z'] - ranges['MIN_Z'])
    #dqs = (np.log(qs) - ranges['MIN_LNQ']) / (ranges['MAX_LNQ'] - ranges['MIN_LNQ'])

    scaled[:, 0] = dxs
    scaled[:, 1] = dys
    scaled[:, 2] = dzs
    #scaled[:, 3] = dqs

    return scaled




def process_file(file_path, save_path, min_len, max_len, upsampleN,RNG,ranges):
    '''
    Turns .h5 file into individual numpy arrays for each event. 
    Hard coded for 4 dimensional output point cloud and 
    input point cloud to be [x, y, z, t, q_a, ...]

    Parameters:
        file_path: str - Path to .h5 file
        save_path: str - Path to folder for saving .npy files for each event
        min_len: int - Minimum number of unique points allowed in an event
        max_len: int - Maximum number of unique points allowed in an event

    Returns:
        None
    '''

    file = h5py.File(file_path, 'r')
    keys = list(file.keys())

    lengths = np.ndarray((len(keys)), dtype=int)
    for i, k in enumerate(keys):
        lengths[i] = len(file[k])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i, k in enumerate(file):

        if lengths[i] < min_len or lengths[i] > max_len:
            continue

        event = np.ndarray((lengths[i], 4))
        for idx, p in enumerate(file[k]):
            event[idx, 0] = p[0]
            event[idx, 1] = p[1]
            event[idx, 2] = p[2]
            #event[idx, 3] = p[4]

        # Upsample
        if event.shape[0] < upsampleN:
            extra = RNG.choice(event.shape[0], 2048 - event.shape[0], replace=True)
            event = np.concatenate([event, event[extra]], axis=0)
        
        # Save each event with a random hash
        # Scale event also
        scaled_event = scale_data(event,ranges)
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, scaled_event)


def get_ttv_split(mg_path, train_split=0.6, val_split=0.2):
    '''
    Splits events into train, val, and test sets

    Parameters:
        mg_path: str - path to 22Mg events
        o_path: str - path to 16O events
        train_split: float - percentage of events to put in train set
        val_split: float - percentage of events to put in val set

    Returns:
        train: np.ndarray - hahses that are part of the train set 
                            (and which isotope they are)
        val: np.ndarray - hashes that are part of the val set
                          (and which isotope they are)
        test: np.ndarray - hashes that are part of the test set
                           (and which isotope they are)
    '''
    
    mg_hashes = os.listdir(mg_path)

    name_arr = np.ndarray(len(mg_hashes),
                          dtype=[('hash', 'object'), ('experiment', 'object')])
    i = 0
    for h in mg_hashes:
        name_arr[i] = (h.split('.')[0], '22Mg')
        i += 1

    rng = np.random.default_rng()
    rng.shuffle(name_arr)

    num_train = int(train_split * i)
    num_val = int(val_split * i)

    train = name_arr[:num_train]
    val = name_arr[num_train:num_train+num_val]
    test = name_arr[num_train+num_val:]

    return train, val, test


def make_category_file(train, val, test, path):
    '''
    Creates .json category file for dataset.

    Parameters:
        train: np.ndarray - hahses that are part of the train set 
                            (and which isotope they are)
        val: np.ndarray - hashes that are part of the val set
                          (and which isotope they are)
        test: np.ndarray - hashes that are part of the test set
                           (and which isotope they are)
        path: str - path to category file

    Returns:
        None
    '''

    with open(path, 'w') as jason:

        jason.write("[\n")

        jason.write("\t{\n")
        jason.write("\t\t\"experiment\": \"22Mg\",\n")
        jason.write("\t\t\"train\": [\n")

        for event in train[:-1]:
            if event['experiment'] != "22Mg":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")

        jason.write(f"\t\t\t\"{train[-1]['hash']}\"\n")
        jason.write("\t\t],\n")

        jason.write("\t\t\"val\": [\n")
        for event in val[:-1]:
            if event['experiment'] != "22Mg":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        jason.write(f"\t\t\t\"{val[-1]['hash']}\"\n")
        jason.write("\t\t],\n")

        jason.write("\t\t\"test\": [\n")
        for event in test[:-1]:
            if event['experiment'] != "22Mg":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        jason.write(f"\t\t\t\"{test[-1]['hash']}\"\n")
        jason.write("\t\t]\n")

        jason.write("\t}\n]")
        return
    

def sort_files(mg_path, save_path, train, val, test):
    '''
    Reorganizes files based on their dataset.
    Raises a NameError if an unknown hash is encountered

    Parameters:
        mg_path: str - path to where 22Mg files were initially stored
        o_path: str - path to where 16O files were initially stored
        save_path: str - path to where files will be stored properly
        train: np.ndarray - hahses that are part of the train set 
                            (and which isotope they are)
        val: np.ndarray - hahses that are part of the val set 
                            (and which isotope they are)
        test: np.ndarray - hahses that are part of the test set 
                            (and which isotope they are)

    Returns:
        None
    '''

    # Ensure folders exist / create if they don't
    # To not create them where I run it from
    """
    if not os.path.exists(f"./train/"):
        os.mkdir("./train")
    if not os.path.exists("./train/complete"):
        os.mkdir("./train/complete")
    if not os.path.exists(f"./val/"):
        os.mkdir("./val")
    if not os.path.exists("./val/complete"):
        os.mkdir("./val/complete")
    if not os.path.exists(f"./test/"):
        os.mkdir("./test")
    if not os.path.exists("./test/complete"):
        os.mkdir("./test/complete")
    """

    for split in ('train', 'val', 'test'):
        os.makedirs(f"{save_path}/{split}/complete", exist_ok=True)

    # Reorganize files
    for file in os.listdir(mg_path):

        hsh = file.split(".")[0]

        if hsh in train['hash']:
            os.rename(os.path.join(mg_path, file), os.path.join(save_path, 'train', 'complete', file))
        elif hsh in val['hash']:
            os.rename(os.path.join(mg_path, file), os.path.join(save_path, 'val', 'complete', file))
        elif hsh in test['hash']:
            os.rename(os.path.join(mg_path, file), os.path.join(save_path, 'test', 'complete', file))
        else:
            raise NameError(f"Hash {hsh} not found with 22Mg files")

        
    # Remove old folders
    os.rmdir(mg_path)

    return


def sample(data, lengths, n_complete, rng):
    """
    Samples events to a set number of total points

    Parameters:
        data: numpy.ndarray - data to sample
        lengths: numpy.ndarray - number of unique points in each event
        n_complete: int - number of points to sample to
        rng: numpy.random._generator.Generator - random number generator 
            (usually an instance of numpy.random.default_rng())

    Returns:
        sampled: numpy.ndarray - sampled events
    """

    sampled = np.ndarray((len(lengths), n_complete, 4))
    ZERO = np.array([0.0, 0.0, 0.0])

    idx = 0
    for i, ev in enumerate(data):

        if lengths[i] == 0: # Discard
            continue

        elif lengths[i] < n_complete: # Up sample

            # Get complete event
            og_l, l = lengths[i], lengths[i]
            sampled[idx, :og_l] = ev[:og_l]

            # Randomly select non zero points to upsample with
            while l < n_complete:
                chosen = rng.choice(ev[:og_l])
                while np.array_equal(chosen, ZERO):
                    chosen = rng.choice(ev[:og_l])
                sampled[idx, l] = chosen
                l += 1

        elif lengths[i] > n_complete: # Down sample

            # Randomly select n_complete points
            whole_ev = ev[:lengths[i]]
            rng.shuffle(whole_ev, axis=0)
            sampled[idx] = whole_ev[:n_complete]

        else: # No sampling necessary
            sampled[idx] = ev[:n_complete]
        
        idx += 1

    return sampled



def create_partial_clouds(path, percentage_cut=0.25):
    '''
    Cuts complete cloud into 3 partial clouds: center, random, and downsampled

    Parameters:
        path: str - path to where files will be stored properly
        percentage_cut: float - percentage of points cut from each event

    Returns:
        None
    '''

    # Ensure folders exist / create if they don't
    # To not create them where I run it from
    """
    if not os.path.exists("./train/partial"):
        os.makedirs(os.path.join(path,"train","partial"),exist_ok=True)
    if not os.path.exists("./val/partial"):
        os.makedirs(os.path.join(path,"val","partial"),exist_ok=True)
    if not os.path.exists("./test/partial"):
        os.makedirs(os.path.join(path,"test","partial"),exist_ok=True)
    """

    for split in ('train', 'val', 'test'):
        os.makedirs(f"{path}/{split}/partial", exist_ok=True)

    rng = np.random.default_rng()

    for split in ('/train', '/val', '/test'):
        for file in os.listdir(path+split+'/complete'):
            event = np.load(path+split+'/complete/'+file)
            k = int(len(event) * percentage_cut)

            center = cf.center_cut(event, k)
            #rand = cf.rand_cut(event, k, rng)
            #down = cf.down_sample(event, k)

            hsh = file.split('.')[0]
            if not os.path.exists(path+split+f'/partial/{hsh}'):
                os.mkdir(path+split+f'/partial/{hsh}')

            np.save(path+split+f'/partial/{hsh}/center.npy', center)
            #np.save(path+split+f'/partial/{hsh}/rand.npy', rand)
            #np.save(path+split+f'/partial/{hsh}/down.npy', down)

    return


if __name__ == '__main__':

    MG_FILE_PATH = '/data/22Mg/point_clouds/simulated/output_digi_HDF_Mg22_Ne20pp_8MeV.h5'

    # Make sure to edit these paths accordingly, for some reason it doesn't
    # like it when ~ is used instead of /home/DAVIDSON/username
    MG_SAVE_PATH = '/home/DAVIDSON/hayavuzkara/Data/Recycle/mg22'

    FINAL_PATH = "/home/DAVIDSON/hayavuzkara/Data/22Mg"

    MIN_N_POINTS = 80
    MAX_N_POINTS = 1500

    CATEGORY_FILE_PATH = "/home/DAVIDSON/hayavuzkara/Data/22Mg/category.json"

    #RANGE

    
    N_COMPLETE = 2048
    N_PARTIAL = 1024 + 512
    MIN_N_UNIQUE = 128

    RANGES = {
        'MIN_X' : -270.0,
        'MAX_X' :  270.0,
        'MIN_Y' : -270.0,
        'MAX_Y' :  270.0,
        'MIN_Z' : -185.0,
        'MAX_Z' : 1185.0,
        'MIN_LNQ' :  1.0,
        'MAX_LNQ' : 10.2
    }

    RNG = np.random.default_rng()


    # Process
    process_file(MG_FILE_PATH, MG_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS,N_COMPLETE,RNG,RANGES)

    # Split and sort
    train, val, test = get_ttv_split(MG_SAVE_PATH)
    train = np.sort(train, order='experiment')
    val = np.sort(val, order='experiment')
    test = np.sort(test, order='experiment')
    make_category_file(train, val, test, CATEGORY_FILE_PATH)
    sort_files(MG_SAVE_PATH, FINAL_PATH, train, val, test)

    # Cut
    create_partial_clouds(FINAL_PATH)