from torch.utils.data.dataloader import default_collate

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import pickle
from options import cfg

def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')
    if cfg.TRAIN.MODAL != 'extraction':
        for elem in batch:
            if cfg.TRAIN.MODAL != 'extraction':
                imgs, caps, cls_id, key, label = elem
            max_input_len = max_input_len if max_input_len > caps.shape[0] else caps.shape[0]       

        for i, elem in enumerate(batch):
            imgs, caps, cls_id, key,label = elem
            input_length = caps.shape[0]
            input_dim = caps.shape[1]
            # print('f.shape: ' + str(f.shape))
            feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            feature[:caps.shape[0], :caps.shape[1]] = caps       
            
            batch[i] = (imgs, feature, cls_id, key, input_length, label)
            # print('feature.shape: ' + str(feature.shape))
            # print('trn.shape: ' + str(trn.shape))

        batch.sort(key=lambda x: x[-2], reverse=True)

    return default_collate(batch)

def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    return normalize(img)

class SpeechDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 img_size=64,
                 transform=None, target_transform=None):
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.SPEECH.CAPTIONS_PER_IMAGE   
        self.imsize = img_size
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        print(self.class_id)

    def load_filenames(self, data_dir, split):      
        filepath = '%s/%s/filenames_byimage.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def __getitem__(self, index):

        key = self.filenames[index]
        cls_id = self.class_id[index] 
        label = cls_id

        if cfg.TRAIN.MODAL != 'extraction':
            if self.data_dir.find('Flickr8k') != -1: 
                bbox = None
                data_dir = self.data_dir

            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)

        if cfg.SPEECH.style == 'mel':
            if self.data_dir.find('Flickr8k') != -1:
                audio_file = '%s/audio2/audio_mel/%s.npy' % (data_dir, key) 
            
            if self.split=='train':
                audio_ix = random.randint(0, self.embeddings_num)
            else:
                audio_ix = 0 

            audios = np.load(audio_file,allow_pickle=True)

            if len(audios.shape)==2:
                audios = audios[np.newaxis,:,:]  

            if cfg.TRAIN.MODAL != 'extraction':
                caps = audios[audio_ix] 
            else:
                caps = audios

        if cfg.TRAIN.MODAL =='extraction':
            return caps
        else:
            return imgs, caps, cls_id, key, label   

    def __len__(self):
        return len(self.filenames)