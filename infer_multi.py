import argparse
import os
import random
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from gradio import processing_utils
import json
from transformers import LlamaTokenizer, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import GenerationConfig
from blsp.src.modeling_blsp import BlspModel
from blsp.src.speech_text_paired_dataset import get_waveform
from blsp.src.owntoken import WLT, SPECIA_TOKENS, TASK_SPECIFIC_TAG
from prompt_toolkit import prompt
from transformers import PreTrainedModel, LlamaForCausalLM, LlamaTokenizer



def parse_args():
    parser = argparse.ArgumentParser(description="Speech-Language Demo")
    parser.add_argument(
        "--blsp_model", type=str, default='checkpoints/20231211-traintag_newtag_4/checkpoint-30600'
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
        "--temperature", type=float, default=1.0,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="top_p for generation"
    )
    args = parser.parse_args()
    return args

generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
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
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[:,2:].cuda()
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

tokenizer = LlamaTokenizer.from_pretrained('pretrained_models/llama2-7b-hf')
new_embedding_nums = 0

new_embedding_nums += tokenizer.add_tokens(SPECIA_TOKENS)
new_embedding_nums += tokenizer.add_tokens([val for val in WLT.values()])
extractor = WhisperFeatureExtractor.from_pretrained('pretrained_models/whisper-small')
# model = LlamaForCausalLM.from_pretrained('pretrained_models/llama2-7b-hf')
model = BlspModel.from_pretrained(args.blsp_model, new_embedding_nums)
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

def set_input(data):
    # history.add_text_history(data)
    history.add_text_history("<speech>")
    history.add_audio(data['audio'])
    history.add_speech_history(history.audio_file[-1])
    history.add_text_history("</speech>")
    tag_map = TASK_SPECIFIC_TAG[data['task']]
    history.add_text_history(tag_map['SOT'])
    history.add_text_history(tag_map['AL'][data['audio_language']])
    history.add_text_history(tag_map['TT'])
    history.add_text_history(tag_map['TL'][data['text_language']])
    history.add_text_history(tag_map['TS'])
    history.add_text_history(WLT[data['task']])

def add_text(text):
    history.add_text_history(text)
    
def add_speech(speech_path):
    history.add_text_history("<speech>")
    history.add_audio(speech_path)
    history.add_speech_history(history.audio_file[-1])
    history.add_text_history("</speech>")
    
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
    try:
        history = ChatHistory(tokenizer, extractor)
        mode = prompt('tell me mode, check from 1, 2, 3:')
        if mode == '1':
            data = prompt('tell me data:')
            data = eval(data)
            set_input(data)
        elif mode == '2':
            data = prompt('tell me data:')
            data = eval(data)
            set_input(data)
            instruction=prompt('tell me instruction:')
            add_text(instruction)
        elif mode == '3':
            speech = prompt('tell me speech:')
            add_speech(speech)
            instruction=prompt('tell me instruction:')
            if instruction != '':
                add_text(instruction)
            
        # input_ids = tokenizer('hello', return_tensors="pt").input_ids.cuda()
        # # logits = model.forward(input_ids[:, 1:])
        # suffix_input_ids = tokenizer('hello, what is your name', return_tensors="pt").input_ids[:, 1:].cuda()
        # output = model.generate(
        #     input_ids,
        #     suffix_input_ids,
        #     speech_values=None,
        #     speech_attention_mask=None,
        #     generation_config=generation_config
        # )
        
        import pdb
        pdb.set_trace()
        output = model.chat(
            history=history.history,
            generation_config=generation_config,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print('re: ', response)
    except:
        continue


# import json
# with open('/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/data/test_set/asr/librispeech-test-clean.jsonl', 'r') as f, open('results/librispeech-test-clean.txt', 'w') as f2:
#     for x in tqdm(f):
#         history = ChatHistory(tokenizer, extractor)
#         data = json.loads(x)
#         set_input(data)
#         output = model.chat(
#             history=history.history,
#             generation_config=generation_config,
#         )
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#         print(response + '\t' + data['ground_truth'])
#         f2.write(response + '\t' + data['ground_truth'] + '\n')
        
        
        
# 续写标注： Kash pulled Zahra aside quickly as they were leaving."}
# asr宝珠： as they are leaving can kash pull zahra aside really quickly
#  asr model: as they are leaving can cash pulls our side really quickly
# conti model: As they're leaving, can you pull the car over quickly?


# {"task": "ASR", "audio": "silence.wav", "ground_truth": "general health the quality of life", "audio_language": "EN", "text_language": "EN"}
# het oorspronkelijke is een lief versje van hölty die er wel meer lieve gemaakt heeft waarvan het alleen jammer is dat zij jeugdige dichters tot zeer onhollandsche vertalingen verleiden ik althans heb er van dit zelfde versje nog een liggen die beter onder een neurenburger legprent knabenspiele zou passen dan onder de voorstelling van een hoop aardige

# {"task": "ASR", "audio": "exp/speech_llm/GigaSpeech/data/audio/xs_files/xs_chunks_0000/AUD0000000468_S0000084.wav", "ground_truth": "for alice had read several nice little stories about children that got burnt and eaten up by wild beasts and other unpleasant things", "audio_language": "EN", "text_language": "EN"}{"task": "ASR", "audio": "exp/speech_llm/GigaSpeech/data/audio/xs_files/xs_chunks_0000/YOU0000000315_S0000660.wav", "ground_truth": "as they are leaving can kash pull zahra aside really quickly", "audio_language": "EN", "text_language": "EN"}
# {"task": "ASR", "audio": "asr/public-dataset/GigaSpeech/data/audio/test_files/test_chunks_0001/YOU1000000160_S0000009.wav", "ground_truth": "the p s 5 and x box 2 were rely on disks and downloads to provide a consistent quality for a lot of the games", "audio_language": "EN", "text_language": "EN"}


{"task": "ASR", "audio": "asr/public-dataset/GigaSpeech/data/audio/test_files/test_chunks_0001/YOU1000000173_S0000145.wav", "ground_truth": "these 4 by 4 vehicles run on cummin is diesel engines capable of over 400 horsepower in the most powerful trim", "audio_language": "EN", "text_language": "EN"}

{"task": "ASR", "audio": "exp/speech_llm/GigaSpeech/data/audio/xs_files/xs_chunks_0000/YOU0000000315_S0000660.wav", "ground_truth": "as they are leaving can kash pull zahra aside really quickly", "audio_language": "EN", "text_language": "EN"}
{"task": "AAC", "audio": "kid/dog.wav", "ground_truth": "general health the quality of life", "audio_language": "UNKNOWN", "text_language": "UNKNOWN"}
