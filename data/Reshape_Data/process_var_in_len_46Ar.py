"""
Processes and cuts .h5 files of simulated AT-TPC data
into one numpy array per event for track completion for 46Ar

Author: Hakan Bora
Date Created: 11 Jul 2025
Code heavily borrowed from Ben Wagner 
"""


import numpy as np
import os
import h5py
import random
import sys
import cutting_functions as cf
sys.path.append("/data")
sys.path.append("../")



def scale_data(event):
    """
    x,y (-1,1)
    z (0,4)
    
    """
    scaled = np.empty(event.shape, dtype=np.float32)
    # xs, ys, zs, qs = event[:,0], event[:,1], event[:,2], event[:,3] For 4D
    xs, ys, zs = event[:,0], event[:,1], event[:,2]

    scaled[:,0] = (xs)/255.0
    scaled[:,1] = (ys)/255.0
    scaled[:,2] = (zs)/255.0
    #scaled[:,3] = qs # Charge can be gotten rid of if wanted, event[:,3] and qs should be deleted from above in that case.
    return scaled



def process_file(file_path, save_path, min_len, max_len):
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

        # Save each event with a random hash
        # Scale event also
        scaled_event = scale_data(event)
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, scaled_event)


def get_ttv_split(ar_path, train_split=0.6, val_split=0.2):
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
    
    ar_hashes = os.listdir(ar_path)

    name_arr = np.ndarray(len(ar_hashes),
                          dtype=[('hash', 'object'), ('experiment', 'object')])
    i = 0
    for h in ar_hashes:
        name_arr[i] = (h.split('.')[0], '46Ar')
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
    """
    Minimal fix: generate a single-experiment (46Ar) category file
    without risking IndexError when a split is empty.
    """

    # --- keep only the 46Ar events ----------------------------------
    train = [e for e in train if e['experiment'] == "46Ar"]
    val   = [e for e in val   if e['experiment'] == "46Ar"]
    test  = [e for e in test  if e['experiment'] == "46Ar"]

    with open(path, 'w') as jason:
        jason.write("[\n")
        jason.write("\t{\n")
        jason.write("\t\t\"experiment\": \"46Ar\",\n")

        def dump_split(name, split, trailing_comma):
            jason.write(f'\t\t"{name}": [\n')
            for i, ev in enumerate(split):
                comma = "," if i < len(split) - 1 else ""
                jason.write(f'\t\t\t"{ev["hash"]}"{comma}\n')
            jason.write("\t\t]" + trailing_comma + "\n")

        dump_split("train", train, ",")
        dump_split("val",   val,   ",")
        dump_split("test",  test,  "")

        jason.write("\t}\n]")
        return
    

def sort_files(ar_path, save_path, train, val, test):
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
    for file in os.listdir(ar_path):

        hsh = file.split(".")[0]

        if hsh in train['hash']:
            os.rename(os.path.join(ar_path, file), os.path.join(save_path, 'train', 'complete', file))
        elif hsh in val['hash']:
            os.rename(os.path.join(ar_path, file), os.path.join(save_path, 'val', 'complete', file))
        elif hsh in test['hash']:
            os.rename(os.path.join(ar_path, file), os.path.join(save_path, 'test', 'complete', file))
        else:
            raise NameError(f"Hash {hsh} not found with 46Ar files")

    # Remove old folders
    os.rmdir(ar_path)


    return


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
            rand = cf.rand_cut(event, k, rng)
            down = cf.down_sample(event, k)

            hsh = file.split('.')[0]
            if not os.path.exists(path+split+f'/partial/{hsh}'):
                os.mkdir(path+split+f'/partial/{hsh}')

            np.save(path+split+f'/partial/{hsh}/center.npy', center)
            np.save(path+split+f'/partial/{hsh}/rand.npy', rand)
            np.save(path+split+f'/partial/{hsh}/down.npy', down)

    return


if __name__ == '__main__':

    AR_FILE_PATH = '/data/46Ar/point_clouds/experimental/clean_run_0130.h5'
    
    # Make sure to edit these paths accordingly, for some reason it doesn't
    # like it when ~ is used instead of /home/DAVIDSON/username
    AR_SAVE_PATH = '/home/DAVIDSON/hayavuzkara/Data/Recycle/ar46'


    FINAL_PATH = "/home/DAVIDSON/hayavuzkara/Data/46Ar"

    MIN_N_POINTS = 50
    MAX_N_POINTS = 1500

    CATEGORY_FILE_PATH = "/home/DAVIDSON/hayavuzkara/Data/46Ar/category.json"


    # Process
    process_file(AR_FILE_PATH, AR_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS)

    # Split and sort
    train, val, test = get_ttv_split(AR_SAVE_PATH)
    train = np.sort(train, order='experiment')
    val = np.sort(val, order='experiment')
    test = np.sort(test, order='experiment')
    make_category_file(train, val, test, CATEGORY_FILE_PATH)
    sort_files(AR_SAVE_PATH, FINAL_PATH, train, val, test)
    """
    All paths referenced are on the alphalogin.rc.davidson.edu cluster
    
    Experimental
    
    * original point clouds (unsampled): /data/46Ar/point_clouds/experimental/ 
        * run_0130_peaks.h5 
        * run_0210_peaks.h5
        * clean_run_0130.h5  has two additional columns for nearest neighbor distance and hough distance (I think)
    * /data/46Ar/images/experimental/
        * clean_run_0210.h5 has two additional columns for nearest neighbor distance and hough distance (I think)
        * images/x-y/proton-carbon-junk-noise.h5 128x128x3 (duplicated fro RGB) images. 2151 training examples, 538 test examples
    """


    # Cut
    create_partial_clouds(FINAL_PATH)