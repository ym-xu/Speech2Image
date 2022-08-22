import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from model.transforms import *

import os
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
from torch.utils.data.dataloader import default_collate
from typing import Any, Callable, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

config = OmegaConf.load('./config/birds_train.yml')
config.SPEECH.CAPTIONS_PER_IMAGE

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

    return img # normalize(img) # 
    
class SpeechDataset(data.Dataset):

    def __init__(
        self, 
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        audiotransform: Optional[Callable] = None,
        img_size: int = 64,
        target_transform: Optional[Callable] = None,
        ) -> None:

        #super().__init__(root, transform=transform, target_transform=target_transform)
        super().__init__()

        self.train = train
        self.transform = transform
        self.audiotransform = audiotransform

        self.norm = transforms.Compose([
            transforms.Resize(int(256 * 76 / 64)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.target_transform = target_transform

        self.embeddings_num = config.SPEECH.CAPTIONS_PER_IMAGE   

        self.imsize = img_size

        self.root = root

        if root.find('birds') != -1 or root.find('FFHQ-Text') != -1:   
            self.bbox = self.load_bbox()

        if self.train:
            split_dir = os.path.join(root, 'train-s')
            self.filenames = self.load_filenames(root, 'train-s')
        else:
            split_dir = os.path.join(root, 'test-s')
            self.filenames = self.load_filenames(root, 'test-s')

        self.class_id = self.load_class_id(split_dir, len(self.filenames))

        if config.DATASET_NAME == 'birds' or config.DATASET_NAME == 'flowers':
            if self.train:
                unique_id = np.unique(self.class_id)
                seq_labels = np.zeros(config.DATASET_ALL_CLSS_NUM)
                for i in range(config.DATASET_TRAIN_CLSS_NUM):
                    seq_labels[unique_id[i]-1]=i
                
                self.labels = seq_labels[np.array(self.class_id)-1]    

    def load_bbox(self):
        data_dir = self.root
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
    
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        
        return filename_bbox

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):      
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames
    
    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index] 

        if self.train:
            if config.DATASET_NAME == 'birds' or config.DATASET_NAME == 'flowers':
                label = self.labels[index]
            else:
                label = cls_id 
        else:
            label = cls_id

        if config.TRAIN.MODAL != 'extraction':
            if self.root.find('birds') != -1: 
                bbox = self.bbox[key]
                data_dir = '%s/CUB_200_2011' % self.root   

            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)  

        if config.SPEECH.style == 'mel':
            if  self.root.find('Flickr8k') != -1:
                audio_file = '%s/audio/audio_mel/%s.npy' % (data_dir, key) 
            elif self.root.find('places') != -1:
                audio_file = '%s/audio/mel/%s.npy' % (data_dir, key) 
            else:
                audio_file = '%s/audio_mel-64/%s.npy' % (data_dir, key) 
            
            if self.train:
                audio_ix = random.randint(0, self.embeddings_num - 1 )
            else:
                audio_ix = 0            
            audios = np.load(audio_file,allow_pickle=True)
            if len(audios.shape)==2:
                audios = audios[np.newaxis,:,:]            
            
            if config.TRAIN.MODAL != 'extraction':
                caps = audios[audio_ix]
                #caps = audios[0]
            else:
                caps = audios
        elif config.SPEECH.style == 'esresnet':
            if self.train:
                audio_ix = random.randint(1, self.embeddings_num - 1)
                audio_file = '%s/audio/%s_%s.mp3' % (data_dir, key, str(audio_ix))
            else:
                audio_ix = 0
                audio_file = '%s/audio/%s_%s.mp3' % (data_dir, key, str(audio_ix))
            wav, sample_rate = librosa.load(audio_file, sr=22050, mono=True)
            if wav.ndim == 1:
                wav = wav[:, np.newaxis]
                
            if np.abs(wav.max()) > 1.0:
                wav = scale(wav, wav.min(), wav.max(), -1.0, 1.0)
                
            wav = wav.T * 32768.0
            wav = wav.astype(np.float32)
            
            caps = self.audiotransform(wav)

        if config.TRAIN.MODAL =='extraction':
            return caps
        else:
            return imgs, caps, cls_id, key, label   
            #return imgs, label

    def __len__(self) -> int:
        return len(self.filenames)

