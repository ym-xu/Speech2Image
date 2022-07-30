import os
import numpy as np
import librosa
import opensmile

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

    #clss_names = os.listdir(file_ads)
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
                y,sr = librosa.load(audio_path)
                audio.append(y)
            save_path = out_ads + '/'+ clss_name
        
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path +'/' + img_name + '.npy'
            np.save(save_name,audio)
            break


wavs_ads = './../data/Flickr8k/audio2'
out_ads = './../data/Flickr8k/audio2/audio_npy'

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
                y = smile.process_file(audio_path)
                audio.append(np.array(y).ravel())
            save_path = out_ads + '/'+ clss_name
        
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path +'/' + img_name + '.npy'
            np.save(save_name,audio)


wavs_ads = './../data/Flickr8k/audio2'
out_ads = './../data/Flickr8k/audio2/audio_opensmile'

opensmile_features(wavs_ads, out_ads)