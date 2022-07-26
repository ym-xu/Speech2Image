import audformat
import opensmile
from scipy.io.wavfile import read as read_wav
import os

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)
sampling_rate, data=read_wav('./../data/Flickr8k/audio/wavs/2878272032_fda05ffac7_4.wav')
print(sampling_rate)
#feats = smile.process_files('./../data/Flickr8k/audio/wavs/2878272032_fda05ffac7_4.wav')

file_ads = './../data/Flickr8k/audio/wavs'
out_ads = './../data/Flickr8k/audio/wavs_mel'
audio_names = os.listdir(file_ads)
i = 0
for audio_name in sorted(audio_names):
    audio_path = os.path.join(file_ads,audio_name) 
#    smile.process_files(audio_path)
