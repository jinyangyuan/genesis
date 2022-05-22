# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import os
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from forge import flags
from forge.experiment_tools import fprint


flags.DEFINE_string('path_data', '', 'Path to data file.')
flags.DEFINE_integer('num_workers', 1, 'Number of threads for loading data.')
flags.DEFINE_integer('img_size', 0, 'Dimension of images. Images are square.')
flags.DEFINE_integer('K_steps', 0, 'Number of object slots.')
flags.DEFINE_integer('K_steps_general', 0, 'Number of object slots.')
flags.DEFINE_integer('num_tests', 5, 'Number of test runs.')


def load(cfg, **unused_kwargs):
    del unused_kwargs
    fprint(f"Using {cfg.num_workers} data workers.")
    assert cfg.img_size != 0
    assert cfg.K_steps != 0
    assert cfg.K_steps_general != 0
    data_loaders = {}
    with h5py.File(cfg.path_data, 'r') as f:
        for phase in f:
            data = f[phase]['image'][()]
            data_loaders[phase] = DataLoader(
                CustomizedDataset(data),
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                shuffle=(phase == 'train'),
                drop_last=(phase == 'train'),
                pin_memory=True,
            )
    return data_loaders


class CustomizedDataset(Dataset):

    def __init__(self, images):
        self.images = torch.from_numpy(np.rollaxis(images, -1, -3).astype(np.float32) / 255)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return {'input': self.images[idx]}
