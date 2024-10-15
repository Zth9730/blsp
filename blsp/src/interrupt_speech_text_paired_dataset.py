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

Text_Format = (
    "###[Human]:{instruction}"
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

    input_ids = tokenizer("###[Human]:").input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)    

    if batch["input_type"] == "text":
        instruction = "<text>" + batch["instruction"] + "</text>"
    elif batch["input_type"] == "audio":
        instruction = "<audio>" + batch["instruction"] + "</audio>"
    elif batch["input_type"] == "mix":
        assert "text_instruction" in batch
        assert "audio_instruction" in batch
        instruction = "<text>" + batch["text_instruction"] + "</text>" + "<audio>" + batch["audio_instruction"] + "</audio>"
    
    input_ids += tokenizer(instruction).input_ids[1:]
    attention_mask += [1] * len(instruction)
    labels += tokenizer("<|listen|>").input_ids[1] * len(instruction)    
    
    input_tag_ids = tokenizer("<|listen|>").input_ids[1] * len(instruction)

    if batch["output_type"] == "text":
        output = "<text>" + batch["output"] + "</text>"
    elif batch["output_type"] == "audio":
        output = "<audio>" + batch["output"] + "</audio>"
    elif batch["output_type"] == "mix":
        assert "text_output" in batch
        assert "audio_output" in batch
        output = "<text>" + batch["text_output"] + "</text>" + "<audio>" + batch["audio_output"] + "</audio>"
    
    output = output + tokenizer.eos_token
    output_ids = tokenizer(output).input_ids[1:] # remove bos token
    input_ids += output_ids
    attention_mask += [1] * len(output_ids)
    labels += output_ids

    input_tag_ids += tokenizer("<|silence|>").input_ids[1] * len(output)

    batch["input_ids"] = input_ids[:max_length]
    batch["input_tag_ids"] = input_tag_ids[:max_length]
    batch["attention_mask"] = attention_mask[:max_length]
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
        input_tag_ids = [sample["input_tag_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        input_tag_ids = collate_tokens(input_tag_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)
        
        return {
            "input_ids": input_ids,
            "input_tag_ids": input_tag_ids,
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
    
