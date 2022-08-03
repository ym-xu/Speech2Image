import numpy as np
import librosa
import os
import opensmile
import torch
from torchvggish import vggish, vggish_input
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

path = './../data/birds/CUB_200_2011_audio/audio'
clss_names = os.listdir(path)
save_root = './../data/birds/CUB_200_2011_audio/audio_'

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

def wav2features(path, save_root, f_type = 'mel'):
    save_root = save_root + f_type
    for clss_name in sorted(clss_names):
        clss_path = os.path.join(path,clss_name)
        audio_names= os.listdir(clss_path)
        i = 0
        audio = []
        for audio_name in sorted(audio_names):
            audio_path =  os.path.join(clss_path,audio_name)
            if f_type == 'npy':
                y,sr = librosa.load(audio_path)
                audio.append(y)
            elif f_type == 'mel':
                y,sr = librosa.load(audio_path)
                mel = audio_processing(y)   
                audio.append(mel)
            elif f_type == 'opensmile':
                opens = smile.process_file(audio_path)
                audio.append(opens)
            elif f_type == 'vggish':
                wav_data, sr = librosa.load(audio_path, dtype='int16')
                mel_features = vggish_input.waveform_to_examples(wav_data / 32768.0, sr)
                print(mel_features.shape)
                if mel_features.shape[0] == 0:
                    wav_data2 = np.hstack((wav_data, wav_data))
                    mel_features = vggish_input.waveform_to_examples(wav_data2 / 32768.0, sr)
                embeddings = embedding_model.forward(mel_features)
                audio.append(embeddings)
                
            
            i = i+1
            if i >= 10:
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
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,)    

embedding_model = vggish()
embedding_model.eval()

wav2features(path, save_root, f_type = 'vggish')