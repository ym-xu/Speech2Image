import torch
import numpy as np
import os

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
print(model)

file_ads = './../data/Flickr8k/audio/wavs'
out_ads = './../data/Flickr8k/audio/wavs_vggish'
audio_names = os.listdir(file_ads)
i = 0
for audio_name in sorted(audio_names):
    img_name = audio_name.split('.')[0]
    audio_path = os.path.join(file_ads,audio_name) 
    audio_fe = model.forward(audio_path)
    print(audio_fe)
    if not os.path.exists(out_ads):
        os.makedirs(out_ads)
    save_path = out_ads + '/' + img_name +'.npy'
    np.save(save_path,audio_fe)
