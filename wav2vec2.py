from transformers import AutoProcessor, Wav2Vec2Model
import torch
import torchaudio
import json

with open('/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/pretrained_models/mms-1b/config.json') as f:
    config =  json.load(f)

processor = AutoProcessor.from_pretrained("pretrained_models/mms-1b")
# model = Wav2Vec2Model.from_pretrained("pretrained_models/mms-1b")
model = Wav2Vec2Model(**config)
import pdb
pdb.set_trace()
x, y = torchaudio.load('old_man.wav')
x = x.reshape(-1)
inputs = processor(x, sampling_rate=16000, return_tensors="pt")
print(inputs)
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)