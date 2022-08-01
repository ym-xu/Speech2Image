import os
import numpy as np
import librosa
import opensmile
import torch
from towhee import pipeline 
from torchvggish import vggish, vggish_input

import torch.nn as nn

from shutil import copyfile

file_ads = './../data/Flickr8k/audio/wavs'
out_ads = './../data/Flickr8k/audio2/wavs'

def list_wavs(file_ads, out_ads):
    audio_names = os.listdir(file_ads)

    for audio_name in sorted(audio_names):
        class_name = audio_name[:-6]
        source_path = os.path.join(file_ads, audio_name)
        target_path = os.path.join(out_ads, class_name, audio_name)
        if not os.path.exists(os.path.join(out_ads, class_name)):
            os.makedirs(os.path.join(out_ads, class_name))
        copyfile(source_path, target_path)

#list_wavs(file_ads, out_ads)

def wavs2npy(file_ads, out_ads):

    clss_names = os.listdir(file_ads)
    #clss_names = ['wavs']

    for clss_name in sorted(clss_names):
        clss_path = os.path.join(file_ads,clss_name)
        img_names= os.listdir(clss_path)
        for img_name in sorted(img_names):
            img_path =  os.path.join(clss_path,img_name)
            audio_names = os.listdir(img_path)
            audio = []
            for audio_name in sorted(audio_names):
                audio_path = os.path.join(img_path,audio_name)
                y,sr = librosa.load(audio_path)
                audio.append(y)
            save_path = out_ads + '/'+ clss_name
        
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path +'/' + img_name + '.npy'
            np.save(save_name,audio)
            break


# wavs_ads = './../data/Flickr8k/'
# out_ads = './../data/Flickr8k/audio2/audio_npy'

wavs_ads = './../data/birds/CUB_200_2011_audio'
out_ads = './../data/birds/CUB_200_2011_audio/audio_npy'

#wavs2npy(wavs_ads, out_ads)

def audio_processing(input_file):
    
    y = input_file
    sr = 22050
    window_size = 25
    stride = 10
    input_dim = 40
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]
    cmvn = True

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')  

def wavs2mel(file_ads, out_ads):
    clss_names = os.listdir(file_ads)
    for clss_name in sorted(clss_names):
        clss_path = os.path.join(file_ads,clss_name)
        img_names = os.listdir(clss_path)    
        for img_name in sorted(img_names):
            name = img_name.split('.')[0]
            audio_path = os.path.join(clss_path,img_name)        
            audios =np.load(audio_path,allow_pickle=True)
            mels = []
            for audio in audios:            
                mel = audio_processing(audio)   
                print(mel)    
                mels.append(mel)        
            save_dir = os.path.join(out_ads,clss_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = save_dir + '/' + name +'.npy'
            np.save(save_path,mels)

file_ads = './../data/Flickr8k/audio2/audio_npy'
out_ads = './../data/Flickr8k/audio2/audio_mel'

#wavs2mel(file_ads, out_ads)

def opensmile_features(file_ads, out_ads):

    #clss_names = os.listdir(file_ads)
    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,)

    clss_names = ['audio']

    for clss_name in sorted(clss_names):
        clss_path = os.path.join(file_ads,clss_name)
        img_names= os.listdir(clss_path)
        for img_name in sorted(img_names):
            img_path =  os.path.join(clss_path,img_name)
            audio_names = os.listdir(img_path)
            audio = []
            for audio_name in sorted(audio_names):
                audio_path = os.path.join(img_path,audio_name)
                y = smile.process_file(audio_path)
                audio.append(np.array(y).ravel())
            save_path = out_ads + '/'+ clss_name
        
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path +'/' + img_name + '.npy'
            np.save(save_name,audio)


wavs_ads = './../data/birds/CUB_200_2011_audio'
out_ads = './../data/birds/CUB_200_2011_audio/audio_opensmile'

opensmile_features(wavs_ads, out_ads)

# model = torch.hub.load('harritaylor/torchvggish', 'vggish')
# model.eval()

# process_Data = model._preprocess('./../data/Flickr8k/audio/wavs/3273969811_42e9fa8f63_4.wav',fs = 44100)

# print(process_Data.size())
 

class VGG(nn.Module): 
    def __init__(self): 
        super(VGG, self).__init__() 
        self.features = nn.Sequential( 
            nn.Conv2d(1, 64, 3, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(256, 512, 3, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2)) 
        self.embeddings = nn.Sequential( 
            nn.Linear(512 * 24, 4096), 
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 4096), 
            nn.ReLU(inplace=True), 
            nn.Linear(4096, 128), 
            #nn.ReLU(inplace=True) 
        )

def vggish_features(file_ads, out_ads):
    #embedding_pipeline = pipeline('towhee/audio-embedding-vggish') 
    # model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    # model.eval().to('cpu')
    embedding_model = vggish()
    embedding_model.eval()
    # model = VGG() 
    # saved_state_dict = torch.load('./../data/vggish.pth', map_location=torch.device('cpu')) 
    # model.load_state_dict(saved_state_dict) 
    # model.eval()

    clss_names = ['wavs']

    for clss_name in sorted(clss_names):
        clss_path = os.path.join(file_ads,clss_name)
        img_names= os.listdir(clss_path)
        for img_name in sorted(img_names):
            img_path =  os.path.join(clss_path,img_name)
            audio_names = os.listdir(img_path)
            audio = []
            for audio_name in sorted(audio_names):
                audio_path = os.path.join(img_path,audio_name)
                #embeddings = embedding_pipeline(audio_path) 
                #embeddings = model.forward(audio_path)
                mel_features = vggish_input.wavfile_to_examples(audio_path)
                if mel_features.shape[0] == 0:
                    continue
                embeddings = embedding_model.forward(mel_features)
                audio.append(embeddings)
            if len(audio) != 5:
                for i in range(5 - len(audio)):
                    audio.append(audio[-1])
            save_path = out_ads + '/'+ clss_name
        
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path +'/' + img_name + '.npy'
            np.save(save_name,audio)

wavs_ads = './../data/Flickr8k/audio2'
out_ads = './../data/Flickr8k/audio2/audio_vggish'
#vggish_features(wavs_ads, out_ads)


# embedding_model = vggish()
# embedding_model.eval()

# mel_features = vggish_input.wavfile_to_examples('./../data/Flickr8k/audio/wavs/1402640441_81978e32a9_0.wav')
# name = './../data/Flickr8k/audio/wavs/1402640441_81978e32a9_0.wav'
# print(str(name[:-5]) + str(int(name[-5:-4]) + 1) + '.wav')
# print(mel_features.shape)
# if mel_features.shape[0] == 0:
#     mel_features = mel_features.reshape(mel_features.shape[1], mel_features.shape[2], mel_features.shape[3])

# print(mel_features.shape)