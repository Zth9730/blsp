import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf
from tqdm import tqdm
import io
from petrel_client.client import Client
import json
try:
    from .owntoken import AUDIO_UNIT_TOKENS
except:
    from owntoken import AUDIO_UNIT_TOKENS
import numpy as np
import torch
import random
import datasets
from dataclasses import dataclass
import subprocess
from transformers import LlamaTokenizer, WhisperFeatureExtractor, AutoTokenizer
from datasets import IterableDataset, interleave_datasets, concatenate_datasets

logger = logging.getLogger(__name__)

System_Content = (
    "###[System]:{content}\n\n\n"
)

Text_Format = (
    "###[Human]:{instruction}\n\n\n"
    "###[Assistant]:"
)

class MyClient(object):
    def __init__(self):
        self.client = Client('~/petreloss.conf')
    
    def get (self, key, enable_stream=False):
        index = key.find("/")
        bucket = key[:index]
        new_key = key[index+1:]
        if bucket == "asr" or bucket == "exp":
            return self.client.get("asr:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == "youtubeBucket":
            return self.client.get("youtube:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == "tts":
            return self.client.get("tts:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == "ASR_20T":
            return self.client.get("tts_asr:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == 's3:':
            return self.client.get("asr:{}".format(key), no_cache=True, enable_stream=enable_stream)
        else:
            with open(key, 'rb') as f:
                return f.read()

def process_dataset(batch, tokenizer, max_length):

    system = "<|system|>you are a helpful speech assistant, you should understand input speech and help user<|/system|>"
    system_input = System_Content.format(content=system)
    input_ids = tokenizer(system_input).input_ids
    labels = [-100] * len(input_ids) 

    for i, turn in enumerate(batch['conversations']):
        if turn["input_type"] == "text":
            instruction = "<text>" + turn["instruction"] + "</text>"
        elif turn["input_type"] == "audio":
            instruction = "<audio>" + turn["instruction"] + "</audio>"
        elif turn["input_type"] == "mix":
            assert "text_instruction" in turn
            assert "audio_instruction" in turn
            instruction = "<text>" + turn["text_instruction"] + "</text>" + "<audio>" + turn["audio_instruction"] + "</audio>"
        
        if turn["output_type"] == "text":
            output = "<text>" + turn["output"] + "</text>"
        elif turn["output_type"] == "audio":
            output = "<audio>" + turn["output"] + "</audio>"
        elif turn["output_type"] == "mix":
            assert "text_output" in turn
            assert "audio_output" in turn
            output = "<text>" + turn["text_output"] + "</text>" + "<audio>" + turn["audio_output"] + "</audio>"

        turn_input = Text_Format.format(instruction=instruction)
        turn_output = output + tokenizer.eos_token

        turn_input_ids = tokenizer(turn_input).input_ids[1:]
        input_ids += turn_input_ids
        labels += [-100] * len(turn_input_ids)

        output_ids = tokenizer(turn_output).input_ids[1:] # remove bos token
        input_ids += output_ids
        labels += output_ids

    batch["input_ids"] = input_ids[:max_length]
    batch["attention_mask"] = ([1] * (len(input_ids)))[:max_length]
    batch["labels"] = labels[:max_length]
    return batch

def load_text_instruction_dataset(
    dataroot="",
    manifest_files="",
    max_length=1000000,
    tokenizer=None,
):
    if os.path.exists(os.path.join(dataroot, f"processed_{manifest_files}")):
        logger.warning("load processed dataset")
        dataset = datasets.load_from_disk(os.path.join(dataroot, f"processed_{manifest_files}"))
        return dataset
    
    logger.warning(f"load dataset from scratch from {dataroot}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        dataroot, data_files=manifest_files_list, split="train", streaming=False
    )

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
        },
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False
    )

    dataset.save_to_disk(os.path.join(dataroot, f"processed_{manifest_files}"))

    return dataset

def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][: len(v)])
    return res

@dataclass
class SpeechTextPairedDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def offline_process(
    dataroot="",
    manifest_files="",
    max_length=100000,
    lm_path="",
):
    text_tokenizer = AutoTokenizer.from_pretrained("checkpoints/conversion_speech_hubert_new",  trust_remote_code=True)
    # text_tokenizer.add_special_tokens({"additional_special_tokens": AUDIO_UNIT_TOKENS})

    dataset = load_text_instruction_dataset(
        dataroot,
        manifest_files,
        max_length,
        text_tokenizer,
    )
    for key in dataset[0].keys():
        print(key, len(dataset[0][key]))

if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })
    
