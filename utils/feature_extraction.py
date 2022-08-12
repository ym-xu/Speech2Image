import os
import sys
import glob

import librosa
import librosa.display

import simplejpeg
import numpy as np

import torch
import torchvision as tv

import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import Audio, display

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model.audioclip import AudioCLIP
from utils.transforms import ToTensor1D


torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Partial-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

LABELS = ['cat', 'thunderstorm', 'coughing', 'alarm clock', 'car horn']

aclp = AudioCLIP(pretrained=f'./assets/{MODEL_FILENAME}')

audio_transforms = ToTensor1D()

image_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
    tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])

paths_to_audio = glob.glob('./demo/audio/*.wav')
print(paths_to_audio)

audio = list()
for path_to_audio in paths_to_audio:
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

    # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
    # thus, the actual time-frequency representation will be visualized
    spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

    audio.append((track, pow_spec))
    print(len(track))


audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
print(audio.shape, audio)
((audio_features, _, _), _), _ = aclp(audio=audio)
audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
print(audio_features[0].shape)


# paths_to_images = glob.glob('./demo/images/*.jpg')
# print(paths_to_images)
# images = list()
# for path_to_image in paths_to_images:
#     with open(path_to_image, 'rb') as jpg:
#         image = simplejpeg.decode_jpeg(jpg.read())
#         images.append(image)
        
# images = torch.stack([image_transforms(image) for image in images])
# ((_, image_features, _), _), _ = aclp(image=images)
# image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

# print(image_features.shape)


path = './../data/birds/CUB_200_2011_audio/audio'
clss_names = os.listdir(path)
save_root = './../data/birds/CUB_200_2011_audio/audio_'


def wav2features(path, save_root, f_type = 'mel'):
    save_root = save_root + f_type
    for clss_name in sorted(clss_names):
        print(clss_name)
        clss_path = os.path.join(path,clss_name)
        audio_names= os.listdir(clss_path)
        i = 0
        audio = []
        for audio_name in sorted(audio_names):
            audiof = list()
            path_to_audio =  os.path.join(clss_path,audio_name)
            track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
            spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
            spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
            pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

            audiof.append((track, pow_spec))
            audiof.append((track, pow_spec))
            #audiof.append(audio_name)
            audiof = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audiof])
            
            ((audio_features, _, _), _), _ = aclp(audio=audiof)
            audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
            audio_features = audio_features[0].numpy()
            audio.append(audio_features)

            i = i+1
            if i == 10:
                save_path = save_root + '/'+ clss_name
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = save_path +'/' + audio_name[:-6] + '.npy'
                np.save(save_name,audio)
                i = 0
                audio = []
                continue
         
def img2features(path, save_root, f_type = 'imgclip'):
    save_root = save_root + f_type

    clss_names = os.listdir(path)
    for clss_name in sorted(clss_names):
        print(clss_name)
        if int(clss_name[:3]) < 166:
            continue
        clss_path = os.path.join(path,clss_name)
        img_names= os.listdir(clss_path)

        for img_name in sorted(img_names):
            path_to_image =  os.path.join(clss_path,img_name)
            print(path_to_image)

            with open(path_to_image, 'rb') as jpg:
                image = simplejpeg.decode_jpeg(jpg.read())

            images = torch.stack([image_transforms(image)])
            ((_, image_features, _), _), _ = aclp(image=images)
            image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
            image_features = image_features.numpy()

            # save_path = save_root + '/'+ clss_name
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # save_name = save_path +'/' + img_name[:-4] + '.npy'
            # np.save(save_name,audio)

        

wav2features(path, save_root, f_type = 'audioclip')
#img2features('./../data/birds/CUB_200_2011/images', './../data/birds/CUB_200_2011_img/img_', f_type = 'imgclip')