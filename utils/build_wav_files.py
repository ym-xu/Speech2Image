import os
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

