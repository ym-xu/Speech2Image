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
from options import cfg, cfg_from_file
from utils import *

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data.datasets import SpeechDataset, pad_collate 
from model import AudioModels, ImageModels

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', type = str, default='./../data/Flickr8k')
parser.add_argument('--img_size',type = int, default=256, help = 'the size of image')
parser.add_argument('--manualSeed',type=int,default= 200, help='manual seed')
parser.add_argument('--cfg_file',type = str, default='config/flickr_train.yml',help='optional config file')
parser.add_argument('--gpu_id',type = int, default= -1)

args = parser.parse_args()

cfg.DATA_DIR = args.data_path

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

if args.gpu_id != -1:
    cfg.GPU_ID = args.gpu_id
else:
    cfg.CUDA = False

def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(args.manualSeed + worker_id)

def train(Models,train_loader, test_loader, args):
    audio_model, image_cnn,image_model = Models[0],Models[1],Models[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_root
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_cnn, torch.nn.DataParallel):
        image_cnn = nn.DataParallel(image_cnn)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)


if __name__ == '__main__':

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

