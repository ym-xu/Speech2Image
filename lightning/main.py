## Standard libraries
import os
import numpy as np
import random
import math
import json
from omegaconf import DictConfig, OmegaConf
from functools import partial
from PIL import Image
import argparse
import logging
from collections import OrderedDict

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')

# %matplotlib inline

from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import tensorboard

# datasett model
from data import SpeechDataset
from model import VisionTransformer, CNN_RNN_ENCODER, ESResNeXtFBSP
from utils import *

torch.set_default_dtype(torch.float64)

print(pl.__version__, torch.__version__, torchvision.__version__)
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial15"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


class MM_Matching(pl.LightningModule):
    
    def __init__(self, img_kwargs, audio_kwargs, lr):
        super().__init__()

#         self.trainset = trainset
#         self.datatrain, self.dataval= \
#         torch.utils.data.random_split(self.trainset,
#                                       [round(int(len(trainset)* 0.8)),
#                                        round(len(trainset) - int(len(trainset)* 0.8))])

#         self.datatest = testset

        self.save_hyperparameters()
        self.img_model = VisionTransformer(**img_kwargs)
        self.audio_model = CNN_RNN_ENCODER()
        # self.audio_model = ESResNeXtFBSP(**audio_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]

        self.__build_model()
    
    def __build_model(self):
        pass

    def forward(self, x1, x2, len):
        img_encode = self.img_model.forward(x1)
        audio_encode = self.audio_model.forward(x2, len)

        return img_encode , audio_encode
    #     #return audio_encode

    # def forward(self, x):
    #     return self.img_model(x)

    def _calculate_loss(self, batch, mode="train"):
        #imgs, labels = batch
        imgs, caps, cls_id, key, input_length, labels = batch

        # ---------------------
        # MM Model
        # ---------------------
        img_encode, audio_encode = self(imgs, caps, input_length)
        lossb1, lossb2 = batch_loss(img_encode, audio_encode, cls_id)
        loss_batch = lossb1 + lossb2
        loss = loss_batch * cfg.Loss.gamma_batch
        labels = labels.type(torch.LongTensor).cuda()
        loss = F.cross_entropy(img_encode, labels)  + F.cross_entropy(audio_encode, labels)
        loss += loss * cfg.Loss.gamma_clss

        # ---------------------
        # Audio Model
        # ---------------------
        # labels = labels.type(torch.LongTensor)
        # preds = self.forward(caps)
        # loss = F.cross_entropy(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        # ---------------------
        # Image Model
        # ---------------------
        # labels = labels.type(torch.LongTensor)
        # preds = self.forward(imgs)
        # loss = F.cross_entropy(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(f'{mode}_loss', loss)
        # self.log(f'{mode}_acc', acc)
        return loss
  
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

#     def _epoch_end(self, outputs, name):
#         avg_loss = torch.stack([x[name] for x in outputs]).mean()
#         tqdm_dict = {name: avg_loss}
#         result = OrderedDict({name: avg_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
#         return result

#     def validation_epoch_end(self, outputs):
#         return self._epoch_end(outputs, name="val_loss")
#     def test_epoch_end(self, outputs):
#         return self._epoch_end(outputs, name="test_loss")

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]   

#     def __dataloader(self, train, dataset):
#         # when using multi-node (ddp) we need to add the  datasampler
#         train_sampler = None
#         batch_size = cfg.TREE.BASE_SIZE

#         should_shuffle = train and train_sampler is None
#         should_drop = train and train_sampler is None
#         should_pin = train and train_sampler is None
#         loader = DataLoader(dataset, 
#             batch_size=128, 
#             shuffle=should_shuffle, 
#             drop_last=should_drop, 
#             pin_memory=should_pin, 
#             num_workers=0, 
#             collate_fn=pad_collate
#         )

#         return loader

#     def train_dataloader(self):
#         logging.info('training data loader called')
#         return self.__dataloader(train=True, dataset=self.datatrain)

#     def val_dataloader(self):
#         logging.info('val data loader called')
#         return self.__dataloader(train=False, dataset=self.dataval)

#     def test_dataloader(self):
#         logging.info('val data loader called')
#         return self.__dataloader(train=False, dataset=self.datatest)

def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MM_Matching"), 
                         gpus=1 if str(device)=="cuda:0" else 0,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1)
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MM_Matching-birds.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = MM_Matching.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        model = MM_Matching(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = MM_Matching.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Siamese Network - Face Recognition', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cfg_file',type = str, default='./config/birds_train.yml',help='optional config file')
    parser.add_argument('--imsize', default=256, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--root',type = str, default='./../../data/birds')

    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # config = OmegaConf.load('./../config/birds_train.yml')
    # config.SPEECH.CAPTIONS_PER_IMAGE

    test_transform = transforms.Compose([
                                    transforms.Resize(int(args.imsize * 76 / 64)),
                                    transforms.RandomResizedCrop((256,256),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                    ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize(int(args.imsize * 76 / 64)),
                                    transforms.RandomResizedCrop((256,256),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                    ])

    image_transform = transforms.Compose([
        transforms.Resize(int(args.imsize * 76 / 64)),
        transforms.RandomCrop(args.imsize),
        transforms.RandomHorizontalFlip()])
                                     
    train_dataset = SpeechDataset(root=args.root, train=True, transform=train_transform)
    val_dataset = SpeechDataset(root=args.root, train=True, transform=test_transform)
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) -  int(len(train_dataset) * 0.9)])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) -  int(len(train_dataset) * 0.9)])

    test_set = SpeechDataset(root=args.root, train=False, transform=test_transform)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=0, collate_fn=pad_collate)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0, collate_fn=pad_collate)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0, collate_fn=pad_collate)
    
    model, results = train_model(img_kwargs={
                            'embed_dim': 256,
                            'hidden_dim': 512,
                            'num_heads': 8,
                            'num_layers': 6,
                            'patch_size': 32,
                            'num_channels': 3,
                            'num_patches': 512,
                            'num_classes': 200,
                            'dropout': 0.2},
                            audio_kwargs={
                            'n_fft': 2048,
                            'hop_length': 561,
                            'win_length': 1654,
                            'window': 'blackmanharris',
                            'normalized': True,
                            'onesided': True,
                            'spec_height': -1,
                            'spec_width': -1,
                            'num_classes': 200,
                            'apply_attention': True,
                            'pretrained': False },  
                            lr=3e-4)

    print("ViT results", results)