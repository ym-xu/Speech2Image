import argparse
import os
import pickle
import sys
import time
import torch
import random
import datetime
import pprint
import dateutil.tz
import numpy as np
from PIL import Image
import sys
import torchvision.transforms as transforms 
sys.path.append("..")
from utils.config import cfg

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data.datasets import SpeechDataset, pad_collate 
from model import AudioModels, ImageModels

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', type = str, default='./../data/Flickr8k')
parser.add_argument('--img_size',type = int, default=256, help = 'the size of image')
parser.add_argument('--gpu_id',type = int, default= -1)

args = parser.parse_args()

cfg.DATA_DIR = args.data_path

if args.gpu_id != -1:
    cfg.GPU_ID = args.gpu_id
else:
    cfg.CUDA = False

def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(args.manualSeed + worker_id)

imsize = args.img_size
image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

dataset = SpeechDataset(cfg.DATA_DIR, 'train', 
                        img_size = imsize, 
                        transform=image_transform)
dataset_test = SpeechDataset(cfg.DATA_DIR, 'test',
                            img_size = imsize,
                            transform=image_transform)

assert dataset

train_loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        drop_last=True, 
                                        shuffle=True,
                                        num_workers=cfg.WORKERS,
                                        collate_fn=pad_collate,
                                        worker_init_fn=worker_init_fn)
val_loader = torch.utils.data.DataLoader(dataset_test, 
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        drop_last=False, 
                                        shuffle=False,
                                        num_workers=cfg.WORKERS,
                                        collate_fn=pad_collate,
                                        worker_init_fn=worker_init_fn)

# audio_model = AudioModels.CNN_RNN_ENCODER()
# image_cnn = ImageModels.Inception_v3()
# image_model = ImageModels.LINEAR_ENCODER()

# MODELS = [audio_model, image_cnn,image_model]

for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(train_loader):
    print(i)