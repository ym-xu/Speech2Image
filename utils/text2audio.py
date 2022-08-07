import IPython
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchaudio
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torchaudio.__version__)
print(device)

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

path = './../data/FFHQ-Text/Text/00_Female'
img_names = os.listdir(path)

texts = []
for img_name in sorted(img_names):
    print(img_name)
    i = 0
    for line in open(os.path.join(path , img_name)):
        text = line.strip('\n')
        with torch.inference_mode():
            processed, lengths = processor(text)
            processed = processed.to(device)
            lengths = lengths.to(device)
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
            waveforms, lengths = vocoder(spec, spec_lengths)
        save_path = './../data/FFHQ-Text/audio' + '/'+ img_name[:-4] + '_' + str(i) + '.wav'
        torchaudio.save(save_path, waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
        i = i+1
