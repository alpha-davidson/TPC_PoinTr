"""
Author: Hakan Bora Yavuzkara
Date Created: 26 June 2025
Heavily borrowed from PCNDataset.py
"""

import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class ALPHA(data.Dataset):
    
    def __init__(self,config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        #As data is normalised already
        self.already_norm = getattr(config, 'ALREADY_NORMALISED', False)

        
        # Following is possibly wrong:
        self.variant = getattr(config, 'VARIANT', 'rand')

        
        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
    
        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self,subset):

        #Following idea is added later:
        """
        common = [
        {
            'callback': 'UpSamplePoints',
            'parameters': {'n_points': self.npoints},
            'objects': ['partial', 'gt']
        }
        ]

        if subset == 'train':
            return data_transforms.Compose(
                common + [
                    {
                        'callback': 'RandomMirrorPoints',
                        'objects': ['partial', 'gt']
                    },
                    {
                        'callback': 'ToTensor',
                        'objects': ['partial', 'gt']
                    }
                ]
            )
        else:   # val / test
            return data_transforms.Compose(
                common + [
                    {
                        'callback': 'ToTensor',
                        'objects': ['partial', 'gt']
                    }
                ]
            )
        """
            

        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048 
                },
                'objects': ['partial'] # When gt, error is tensor size mismatch
            },{
                
        # ADDITION FOR GT TO REACH 16384
                
        'callback': 'UpSamplePoints',
        'parameters': {'n_points': self.npoints},
        'objects': ['gt']
                
            },{
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            },{
                
        # ADDITION FOR GT TO REACH 16384
                
        'callback': 'UpSamplePoints',
        'parameters': {'n_points': self.npoints},
        'objects': ['gt']
                
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])


            
    def _get_file_list(self,subset,n_renderings=1):
        file_list=[]

        for dc in self.dataset_categories:
            experiment = dc['experiment']
            print_log(f"Collecting files of experiment [{dc}]", logger='ALPHA')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id': experiment,
                    'experiment': experiment,
                    'model_id': s,
                    'partial_path': self.partial_points_path % (subset, s, self.variant),
                    'gt_path': self.complete_points_path % (subset, s)
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='ALPHA')
        return file_list

    def __getitem__(self,idx):
        """        
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial','gt']:
            file_path = sample['%s_path' % ri]
            
            if type(file_path) == list:
                file_path = file_path[rand_idx]
                
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['experiment'], sample['model_id'], (data['partial'], data['gt'])
        """

        sample = self.file_list[idx]
        data = {}

        
        # [:,:3] is a temporary fix for now to make the channels line up, the other option of making it train on 4-D will also be explored.
        
        # data['partial'] = IO.get(sample['partial_path']).astype(np.float32)[:,:3]
        # data['gt'] = IO.get(sample['gt_path']).astype(np.float32)[:,:3]

        data['partial'] = IO.get(sample['partial_path']).astype(np.float32)
        data['gt'] = IO.get(sample['gt_path']).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)
        
        assert data['gt'].shape[0] == self.npoints, (
            f"GT point count {data['gt'].shape[0]} != expected {self.npoints}")

        
        
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])


    def __len__(self):
        return len(self.file_list)
    