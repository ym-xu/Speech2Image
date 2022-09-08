import numpy as np
import librosa
import os
#import opensmile
import torch
#from torchvggish import vggish, vggish_input
import soundfile as sf
#import wav2clip
import warnings
import json
import torchvision
warnings.filterwarnings('ignore')
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Optional
import importlib

path = './../data/flowers/Oxford-102/Oxfords_102_audio/audio'
clss_names = os.listdir(path)
save_root = './../data/flowers/Oxford-102/Oxfords_102_audio/audio_'

def load_class(package_name: str, class_name: Optional[str] = None) -> Type:
    if class_name is None:
        package_name, class_name = package_name.rsplit('.', 1)

    importlib.invalidate_caches()

    package = importlib.import_module(package_name)
    cls = getattr(package, class_name)

    return cls

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

def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def audio_processing(input_file):
    
    y = input_file
    sr = 22050
    window_size = 25
    stride = 10
    input_dim = 128
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

def wav2features(path, save_root, f_type = 'mel'):
    save_root = save_root + f_type
    for clss_name in sorted(clss_names):
        print(clss_name)
        clss_path = os.path.join(path,clss_name)
        audio_names= os.listdir(clss_path)
        i = 0
        audio = []
        for audio_name in sorted(audio_names):
            audio_path =  os.path.join(clss_path,audio_name)
            if f_type == 'npy':
                y,sr = librosa.load(audio_path)
                audio.append(y)
            elif f_type == 'mel-64':
                i = i+1
                y,sr = librosa.load(audio_path)
                mel = audio_processing(y)   
                audio.append(mel)
            elif f_type == 'opensmile':
                opens = smile.process_file(audio_path)
                audio.append(opens)
                print(opens.shape)
            elif f_type == 'vggish2':
                i = i+1
                wav_data, sr = librosa.load(audio_path, dtype='int16')
                mel_features = vggish_input.waveform_to_examples(wav_data / 32768.0, sr)
                if mel_features.shape[0] == 0:
                    # wav_data2 = np.hstack((wav_data, wav_data))
                    # mel_features = vggish_input.waveform_to_examples(wav_data2 / 32768.0, sr)
                    continue
                embeddings = embedding_model.forward(mel_features).detach().numpy().reshape(-1,64)
                audio.append(embeddings)
            elif f_type == 'wav2clip':
                embeddings = wav2clip.embed_audio(audio, w2cmodel)
                audio.append(embeddings)
            elif f_type == "esresnet":
                i = i+1
                wav, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
                if wav.ndim == 1:
                    wav = wav[:, np.newaxis]

                if np.abs(wav.max()) > 1.0:
                    wav = scale(wav, wav.min(), wav.max(), -1.0, 1.0)

                wav = wav.T * 32768.0
                wav = wav.astype(np.float32)
                if int(clss_name[:3]) > 50 :
                    wav = transforms_tr_audio(wav)
                else:
                    wav = transforms_te_audio(wav)  
                audio.append(wav)


            if i >= 9:
                if len(audio) != 9:
                    print("len(audio): ", len(audio))
                    for i in range(10 - len(audio)):
                        audio.append(audio[i])
                save_path = save_root + '/'+ clss_name
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = save_path +'/' + audio_name[:-6] + '.npy'
                np.save(save_name,audio)
                i = 0
                audio = []
                continue
        # for audio_name in sorted(audio_names):
        #     audio_path = os.path.join(img_path,audio_name)
        #     y,sr = librosa.load(audio_path)
        #     audio.append(y)
        # save_path = save_root + '/'+ clss_name
        
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # save_name = save_path +'/' + img_name + '.npy'
        # np.save(save_name,audio)

        # print("npy ", i, ' finished')
        # i = i+1
# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.ComParE_2016,
#     feature_level=opensmile.FeatureLevel.Functionals,)    

# embedding_model = vggish()
# embedding_model.eval()

#w2cmodel = wav2clip.get_model(frame_length=16000, hop_length=16000)

wav2features(path, save_root, f_type = 'mel-64')