import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from gradio import processing_utils
import json
from transformers import LlamaTokenizer, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import GenerationConfig
from blsp.src.modeling_blsp import BlspModel
from blsp.src.speech_text_paired_dataset import get_waveform
from prompt_toolkit import prompt



def parse_args():
    parser = argparse.ArgumentParser(description="Speech-Language Demo")
    parser.add_argument(
        "--blsp_model", type=str, default='pretrained_models/damo/blsp_lslm_7b',
        help="Path to the blsp model"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75,
        help="top_p for generation"
    )
    args = parser.parse_args()
    return args

generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=True,
    temperature=0.9,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)

class ChatHistory(object):
    def __init__(self, tokenizer, extractor):
        super().__init__()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.history = []
        self.audio_file = []
        self.audio_to_history = True

        ### add bos token
        self.add_bos()

    def add_bos(self):
        input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        self.history.append(
            (input_ids,)
        )

    def add_text_history(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[:,1:].cuda()
        self.history.append(
            (input_ids,)
        )

    def add_audio(self, audio_file):
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech):
        if self.audio_to_history:
            return
        self.audio_to_history = True
        speech = get_waveform(speech, output_sample_rate=self.extractor.sampling_rate)
        speech_inputs = self.extractor(
            speech,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features.cuda()
        speech_attention_mask = speech_inputs.attention_mask.cuda()
        self.history.append(
            (speech_values, speech_attention_mask)
        )

print('Initializing Chat')
args = parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.blsp_model)
extractor = WhisperFeatureExtractor.from_pretrained(args.blsp_model)
model = BlspModel.from_pretrained(args.blsp_model)

generation_config.update(
    **{
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
)

def add_text(user_message):
    history.add_text_history("###[Human]:")
    history.add_text_history(user_message)
    history.add_text_history("\n\n\n###[Assistant]:")
    
def add_file(audio_file):
    history.add_text_history("###[Human]:")
    history.add_audio(audio_file)
    history.add_speech_history(history.audio_file[-1])
    history.add_text_history("\n\n\n###[Assistant]:")

model = model.cuda()
model.eval()

# add_text('Please transcribe the provided audio file into accurate English text.')
# add_text('Please translate the provided audio file into Chinese text, 比如hello变成你好')
# add_text('please translate the following english audio into German text')
# add_text('Please tell me what is the emotion of the provided audio file')
# with open('/mnt/petrelfs/zhoudinghao/work/thzhang/s3prl/s3prl/downstream/emotion/meta_data/Session1/test_meta_data.json') as f, open('emotion_results', 'w') as f_w:
#     data = json.load(f)['meta_data']
#     for wav_file in data:
#         path = wav_file['path']
#         label = wav_file['label']
#         add_text('what is the emotion of the input english audio? Please choose from the following four words:happy, neutral, anger, sad')
#         add_file('asr/task/speech/emotion/{}'.format(path))
#         output = model.chat(
#             history=history.history,
#             generation_config=generation_config,
#         )
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#         print(path, response, label)
#         f_w.write(path + '\t' + label + '\t' + response + '\n')

# prompt_text = 'Please tell me the gender of the speaking person in the audio file.'
# 'Please tell me the approximate age of the person who said this audio.'
while True:
    history = ChatHistory(tokenizer, extractor)
    instruction = prompt('tell me instruction:')
    audio_file = prompt('tell me audio file:')
    print(instruction)
    print(audio_file)
    add_text(instruction)
    # add_file('asr/task/speech/emotion/Session1/sentences/wav/Ses01F_script02_2/Ses01F_script02_2_F000.wav')
    # add_file('asr/crawle/youtube/LdAD9GNv8FI/LdAD9GNv8FI_00005_8784_9490.wav')
    add_file(audio_file)
    output = model.chat(
        history=history.history,
        generation_config=generation_config,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)


# 续写标注： Kash pulled Zahra aside quickly as they were leaving."}
# asr宝珠： as they are leaving can kash pull zahra aside really quickly
#  asr model: as they are leaving can cash pulls our side really quickly
# conti model: As they're leaving, can you pull the car over quickly?