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
from model import VisionTransformer, CNN_RNN_ENCODER
from utils import *

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
    
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.img_model = VisionTransformer(**model_kwargs)
        self.audio_model = CNN_RNN_ENCODER()
        self.example_input_array = next(iter(train_loader))[0]

        self.__build_model()
    
    def __build_model(self):
        pass

    def forward(self, x1, x2, len):
        img_encode = self.img_model.forward(x1)
        audio_encode = self.audio_model.forward(x2, len)

        return img_encode , audio_encode
        #return audio_encode

    # def forward(self, x):
    #     return self.img_model(x)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]   

    def _calculate_loss(self, batch, mode="train"):
        #imgs, labels = batch
        imgs, caps, cls_id, key, input_length, labels = batch
    
        img_encode, audio_encode = self.forward(imgs, caps, input_length)
        # img_encode = self.img_model.forward(imgs)
        # audio_encode = self.audio_model.forward(caps, input_length)
        # img_encode, audio_encode = self.forward(caps, input_length)
        lossb1, lossb2 = self.batch_loss(img_encode, audio_encode, cls_id)
        loss_batch = lossb1 + lossb2
        loss += loss_batch * cfg.Loss.gamma_batch
        labels = labels.type(torch.LongTensor)
        loss = F.cross_entropy(img_encode, labels)  + F.cross_entropy(audio_encode, labels)
        # loss = F.cross_entropy(audio_encode, labels)
        loss += loss * cfg.Loss.gamma_clss
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


    def batch_loss(cnn_code, rnn_code, class_ids,eps=1e-8):
        # ### Mask mis-match samples  ###
        # that come from the same class as the real sample ###
        batch_size = cfg.TRAIN.BATCH_SIZE
        labels = Variable(torch.LongTensor(range(batch_size)))
        labels = labels.cuda()   
    
        masks = []
        if class_ids is not None:
            class_ids =  class_ids.data.cpu().numpy()
            for i in range(batch_size):
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            masks = np.concatenate(masks, 0)
            # masks: batch_size x batch_size
            masks = torch.ByteTensor(masks)
            masks = masks.to(torch.bool)
            if cfg.CUDA:
                masks = masks.cuda()

        # --> seq_len x batch_size x nef
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
        # scores* / norm*: seq_len x batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        if class_ids is not None:
            scores0.data.masked_fill_(masks, -float('inf'))
        scores1 = scores0.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(scores0, labels)
            loss1 = nn.CrossEntropyLoss()(scores1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1

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
    parser.add_argument('--batch_size', default=64, type=int)


    # parser.add_argument('--pretrain_epochs', default=5000, type=int)
    # parser.add_argument('--margin', default=1.0, type=float)
    # parser.add_argument('--should_invert', default=False)
    # parser.add_argument('--imageFolderTrain', default=None)
    # parser.add_argument('--imageFolderTest', default=None)
    # parser.add_argument('--learning_rate', default=2e-2, type=float)
    # parser.add_argument('--resize', default=100, type=int)

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
                                     
    train_dataset = SpeechDataset(root='./../../data/birds', train=True, transform=train_transform)
    val_dataset = SpeechDataset(root='./../../data/birds', train=True, transform=test_transform)
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1)])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1)])

    test_set = SpeechDataset(root='./../../data/birds', train=False, transform=test_transform)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=0, collate_fn=pad_collate)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0, collate_fn=pad_collate)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=0, collate_fn=pad_collate)
    
    model, results = train_model(model_kwargs={
                            'embed_dim': 256,
                            'hidden_dim': 512,
                            'num_heads': 8,
                            'num_layers': 6,
                            'patch_size': 32,
                            'num_channels': 3,
                            'num_patches': 512,
                            'num_classes': 200,
                            'dropout': 0.2
                            }
                            , lr=3e-4)

    print("ViT results", results)