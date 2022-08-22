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
from model import VisionTransformer, CNN_RNN_ENCODER, ESResNeXtFBSP, ESResNet, ESResNeXt, ESResNetFBSP, MMModels, classification
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

def batch_loss(cnn_code, rnn_code, class_ids,eps=1e-8):
        # ### Mask mis-match samples  ###
        # that come from the same class as the real sample ###
        # print(class_ids)
        batch_size = args.batch_size
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
    
def calc_mAP(image_output,audio_output,cls_id):

    value,idx = cls_id.sort()
    image_output = image_output[idx]
    audio_output = audio_output[idx]
    cls_id = cls_id[idx]
    cls_f = -1
    new_cls = []      # classes of the sampled audio
    cls_num = []      #number of each classes of sampled audio
    sampled_audio = []
    i = 0
    j = 0
   
    # with this code, the query is speech
    # only one speech for one class images
    for cls_i in cls_id:
        if cls_i!= cls_f:
            new_cls.append(cls_i.unsqueeze(0))  
            sampled_audio.append(audio_output[i].unsqueeze(0))         
            cls_f = cls_i            
            if i!=0:
                cls_num.append(j)    #             
            j = 1 
        else:
            j += 1
        i += 1   
    cls_num.append(j)   


    new_cls = torch.cat(new_cls)
    sampled_audio = torch.cat(sampled_audio)
       
    # using consine similarity
    if cfg.EVALUATE.dist == 'cosine':
        img_f = normalizeFeature(image_output)
        aud_f = normalizeFeature(sampled_audio) 
        S = aud_f.mm(img_f.t()) 
        value, indx = torch.sort(S,dim=1,descending=True)
    elif cf.EVALUATE.dist == 'L2':
        img_f = image_output / image_output.norm(dim=1,keepdim=True)
        aud_f = sampled_audio / sampled_audio.norm(dim=1,keepdim=True)
        img_ex = (img_f.unsqueeze(0)).repeat(aud_f.shape[0],1,1)
        aud_ex = (aud_f.unsqueeze(1)).repeat(1,img_f.shape[0],1)
        diff = aud_ex - img_ex
        squareDiff = diff**2
        squareDist = squareDiff.sum(-1)
        S = squareDist**0.5
        value, indx = torch.sort(S,dim=1)
    
    
    class_sorted = cls_id[indx]
    clss_m2 = new_cls.unsqueeze(-1).repeat(1,S.shape[1])
    
    mask = (class_sorted==clss_m2).bool()
    class_sorted_filed = class_sorted.data.masked_fill_(mask,-10e5)   

    v, index = torch.sort(class_sorted_filed,dim=1)
    index = index +1
    sc = 0.0
    ap = 0.0


    for i in range(index.shape[0]):
        sc = 0.0
        num = cls_num[i]
        for k in range(num):    
            position =  index[i][:num]  
            position = sorted(position)     
            sc += (k+1.0)/(position[k]).float()
        ap += sc/cls_num[i]
    
    mAP = ap/(mask.shape[0])

    return mAP
    
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
        # self.audio_model = CNN_RNN_ENCODER()
        self.audio_model = ESResNet(**audio_kwargs)
        self.class_model = classification.CLASSIFIER()
        self.mm_model = MMModels.MM_MultiHeadAttention()
        # self.example_input_array = next(iter(train_loader))[0]

        self.__build_model()
    
    def __build_model(self):
        pass

    def forward(self, x1, x2, len):
        image_output = self.img_model.forward(x1)
        audio_output = self.audio_model.forward(x2, len)
        
        image_class_output = self.class_model(image_output) 
        audio_class_output = self.class_model(audio_output)
        
        mm_image_output = self.mm_model(image_output, audio_output)
        mm_audio_output = self.mm_model(audio_output, image_output)
        
        mm_image_class_output = self.class_model(mm_image_output)
        mm_audio_class_output = self.class_model(mm_audio_output)

        return image_output , audio_output, image_class_output, audio_class_output, mm_image_output, mm_audio_output, mm_image_class_output, mm_audio_class_output

#     def forward(self, x):
#         return self.img_model(x)

    def _calculate_loss(self, batch, mode="train"):
        #imgs, labels = batch
        imgs, caps, cls_id, key, input_length, labels = batch

        # ---------------------
        # MM Model
        # ---------------------
        image_output , audio_output, image_class_output, audio_class_output, mm_image_output, mm_audio_output, mm_image_class_output, mm_audio_class_output = self(imgs, caps, input_length)
        # print("img_encode, audio_encode: ", img_encode.type(), img_encode.shape, audio_encode.type(), audio_encode.shape)
        lossb1, lossb2 = batch_loss(image_output, audio_output, cls_id)
        loss_batch = lossb1 + lossb2
        loss = loss_batch * cfg.Loss.gamma_batch
        
        lossm1,lossm2 = batch_loss(mm_image_output,mm_audio_output,cls_id)
        loss_batch_m = lossm1 + lossm2
        loss += loss_batch_m*cfg.Loss.gamma_batch_m
            
        labels = labels.type(torch.LongTensor).cuda()
        loss = F.cross_entropy(image_class_output, labels)  + F.cross_entropy(audio_class_output, labels) + F.cross_entropy(mm_image_class_output, labels)  + F.cross_entropy(mm_audio_class_output, labels)
        loss += loss * cfg.Loss.gamma_clss

        # ---------------------
        # Audio Model
        # ---------------------
        # labels = labels.type(torch.LongTensor).cuda()
        # preds = self.forward(caps)
        # loss = F.cross_entropy(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        # self.log(f'{mode}_acc', acc)
        
        # ---------------------
        # Image Model
        # ---------------------
        # labels = labels.type(torch.LongTensor).cuda()
        # preds = self.forward(imgs)
        # loss = F.cross_entropy(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        # self.log(f'{mode}_acc', acc)
        
        self.log(f'{mode}_loss', loss)
        
        if mode != "train":
            # I_embeddings = [] 
            # A_embeddings = [] 
            # class_ids = []
            # I_embeddings.append(img_encode.to('cpu').detach())
            # A_embeddings.append(audio_encode.to('cpu').detach()) 
            # class_ids.append(cls_id.to('cpu'))
            recalls = calc_mAP(image_output.detach(),audio_output.detach(),cls_id)
            self.log(f'{mode}_acc', recalls) 
            
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

def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MM_Matching"), 
                         gpus=1 if str(device)=="cuda:0" else 0,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1, accumulate_grad_batches=8)
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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--root',type = str, default='./../autodl-tmp/data/birds')

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
    
    audioconfig = json.load(open('./config/audio_config.json'))
    audiotransforms = audioconfig['Transforms']

    transforms_tr_audio = list()
    transforms_te_audio = list()

    for idx, transform in enumerate(audiotransforms):
        use_train = transform.get('train', True)
        use_test = transform.get('test', True)

        transform = load_class(transform['class'])(**transform['args'])

        if use_train:
            transforms_tr_audio.append(transform)
        if use_test:
            transforms_te_audio.append(transform)

        audiotransforms[idx]['train'] = use_train
        audiotransforms[idx]['test'] = use_test

    transforms_tr_audio = torchvision.transforms.Compose(transforms_tr_audio)
    transforms_te_audio = torchvision.transforms.Compose(transforms_te_audio)

                                     
    train_dataset = SpeechDataset(root=args.root, train=True, transform=train_transform, audiotransform = transforms_tr_audio)
    val_dataset = SpeechDataset(root=args.root, train=True, transform=test_transform, audiotransform = transforms_tr_audio)
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) -  int(len(train_dataset) * 0.9)])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) -  int(len(train_dataset) * 0.9)])

    test_set = SpeechDataset(root=args.root, train=False, transform=test_transform, audiotransform = transforms_te_audio)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=12, collate_fn=pad_collate)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=12, collate_fn=pad_collate)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=12, collate_fn=pad_collate)
    
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
                            'num_classes': 1024,
                            'apply_attention': True,
                            'pretrained': False },  
                            lr=3e-4)

    print("ViT results", results)