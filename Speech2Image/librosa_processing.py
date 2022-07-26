import numpy as np
import librosa
import os

def audio_processing(input_file, method_type):
    
    y = input_file
    sr = 22050
    window_size = 25
    stride = 10
    input_dim = 40
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)

    if method_type == 'mel':
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = input_dim, n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)

        feat = [feat]
        cmvn = True

    elif method_type == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, hop_length=st, n_mfcc=13)
        feat = np.log(feat + 1e-6)

        feat = [feat]
        cmvn = False

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')  

file_ads = './../data/Flickr8k/audio/wavs'
out_ads = './../data/Flickr8k/audio/wavs_mel'
audio_names = os.listdir(file_ads)
i = 0
for audio_name in sorted(audio_names):
    img_name = audio_name.split('.')[0]
    audio_path = os.path.join(file_ads,audio_name) 
    y,sr = librosa.load(audio_path)
    mel = audio_processing(y, 'mel')
    if not os.path.exists(out_ads):
        os.makedirs(out_ads)
    save_path = out_ads + '/' + img_name +'.npy'
    np.save(save_path,mel)
    

