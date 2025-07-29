"""
Processes and cuts .h5 files of simulated AT-TPC data
into one numpy array per event for track completion

Used for experimental data, for built models to be tested on.

Author: Ben Wagner
Date Created: 12 Feb 2025
Date Edited:  30 Jun 2025 (Hakan Bora Yavuzkara)
"""


import numpy as np
import os
import h5py
import random
import sys
import cutting_functions as cf
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

    xs, ys, zs, qs = event[:, 0], event[:, 1], event[:, 2], event[:, 3]


    dxs = (xs - ranges['MIN_X']) / (ranges['MAX_X'] - ranges['MIN_X'])
    dys = (ys - ranges['MIN_Y']) / (ranges['MAX_Y'] - ranges['MIN_Y'])
    dzs = (zs - ranges['MIN_Z']) / (ranges['MAX_Z'] - ranges['MIN_Z'])
    dqs = (np.log(qs) - ranges['MIN_LNQ']) / (ranges['MAX_LNQ'] - ranges['MIN_LNQ'])

    scaled[:, 0] = dxs
    scaled[:, 1] = dys
    scaled[:, 2] = dzs
    scaled[:, 3] = dqs

    return scaled


def process_group(group, save_path, min_len, max_len, upsampleN,RNG,ranges):
    '''
    Turns .h5 file into individual numpy arrays for each event. Considering that h5 is a group.
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
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for k in group:
        dataset = group[k]
        if not isinstance(dataset, h5py.Dataset):
            continue

        if len(dataset) < min_len or len(dataset) > max_len:
            continue

        event = np.ndarray((len(dataset), 4))
        for idx, p in enumerate(dataset):
            event[idx, 0] = p[0]
            event[idx, 1] = p[1]
            event[idx, 2] = p[2]
            event[idx, 3] = p[4]

        if event.shape[0] < upsampleN:
            extra = RNG.choice(event.shape[0], upsampleN - event.shape[0], replace=True)
            event = np.concatenate([event, event[extra]], axis=0)

        scaled_event = scale_data(event, ranges)
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, scaled_event)

    

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
            event[idx, 3] = p[4]

        # Upsample
        if event.shape[0] < upsampleN:
            extra = RNG.choice(event.shape[0], 2048 - event.shape[0], replace=True)
            event = np.concatenate([event, event[extra]], axis=0)
        
        # Save each event with a random hash
        # Scale event also
        scaled_event = scale_data(event,ranges)
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, scaled_event)

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



if __name__ == '__main__':

    """
    EXP_FILE_PATH = '/data/46Ar/point_clouds/experimental/run_0210_peaks.h5'  #CHANGE TO RELEVANT
    EXP_SAVE_PATH = '/home/DAVIDSON/hayavuzkara/Data/46Ar/all' #CHANGE TO RELEVANT
    """

    EXP_FILE_PATH = '/data/22Mg/point_clouds/experimental/22Mg_alpha_exp.h5'  #CHANGE TO RELEVANT
    EXP_SAVE_PATH = '/home/DAVIDSON/hayavuzkara/Data/22MgExp/all' #CHANGE TO RELEVANT

    MIN_N_POINTS = 50
    MAX_N_POINTS = 1000


    #RANGE

    
    N_COMPLETE = 2048
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

    with h5py.File(EXP_FILE_PATH, 'r') as f:
        first_key = next(iter(f.keys()))
        if isinstance(f[first_key], h5py.Group): #If h5 is a group
            process_group(f[first_key], EXP_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS, N_COMPLETE, RNG, RANGES)
        else:
            process_file(EXP_FILE_PATH, EXP_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS, N_COMPLETE, RNG, RANGES)
            
    