import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf
import io
from petrel_client.client import Client

try:
    from .owntoken import WLT, SPECIA_TOKENS, TASK_SPECIFIC_TAG
except:
    from owntoken import WLT, SPECIA_TOKENS, TASK_SPECIFIC_TAG

import numpy as np
import torch
import random
import datasets
from dataclasses import dataclass
import subprocess
from transformers import LlamaTokenizer, WhisperFeatureExtractor
from datasets import IterableDataset

logger = logging.getLogger(__name__)

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
        elif bucket == 's3:':
            return self.client.get("asr:{}".format(key), no_cache=True, enable_stream=enable_stream)
        else:
            with open(key, 'rb') as f:
                return f.read()


def process_dataset_multitask(batch, tokenizer):
    # tag: SOT-AL-TT-TL-TS-WLT 
    client = MyClient()
    audio_path = batch["audio"]
    try:
        s3b = client.get(audio_path)
        with io.BytesIO(s3b) as fobj: 
            info = sf.info(fobj)
            if info.duration >= 30 and batch['task'] in ["ASR", "S2TT"]:
                is_readable = False 
            else:
                is_readable = True
    except:
        is_readable = False

    input_ids = [tokenizer.bos_token_id] + tokenizer("<speech>").input_ids[2:]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)    

    # end of speech tag: </speech>
    suffix_input_ids = tokenizer('</speech>').input_ids[2:]
    suffix_attention_mask = [1] * len(suffix_input_ids)
    suffix_labels = [-100] * len(suffix_input_ids)
    
    # multi-task tag
    multitask_tag_ids = []
    tag_map = TASK_SPECIFIC_TAG[batch['task']]

    sot_ids = tokenizer(tag_map['SOT']).input_ids[2:]
    multitask_tag_ids += sot_ids
    audio_language = tokenizer(tag_map['AL'][batch['audio_language']]).input_ids[2:]
    multitask_tag_ids += audio_language
    task_tag = tokenizer(tag_map['TT']).input_ids[2:]
    multitask_tag_ids += task_tag
    text_language = tokenizer(tag_map['TL'][batch['text_language']]).input_ids[2:]
    multitask_tag_ids += text_language
    timestamps = tokenizer(tag_map['TS']).input_ids[2:]
    multitask_tag_ids += timestamps
    wlt = tokenizer(WLT[batch['task']]).input_ids[2:]
    multitask_tag_ids += wlt
    
    suffix_input_ids += multitask_tag_ids
    suffix_attention_mask += [1] * len(multitask_tag_ids)
    # suffix_labels = [-100] * len(suffix_input_ids)
    suffix_labels += multitask_tag_ids

    # ground_truth
    ground_truth_ids = tokenizer(batch["ground_truth"]).input_ids[1:] + [tokenizer.eos_token_id]
    suffix_input_ids += ground_truth_ids
    suffix_attention_mask += [1] * len(ground_truth_ids)
    suffix_labels += ground_truth_ids
    
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels
    batch["suffix_input_ids"] = suffix_input_ids
    batch["suffix_attention_mask"] = suffix_attention_mask
    batch["suffix_labels"] = suffix_labels
    batch["audio_path"] = audio_path
    batch["is_readable"] = is_readable

    return batch

def process_dataset_multitask_test(batch, tokenizer):
    # tag: SOT-AL-TT-TL-TS-WLT 
    client = MyClient()
    audio_path = batch["audio"]
    try:
        s3b = client.get(audio_path)
        with io.BytesIO(s3b) as fobj: 
            info = sf.info(fobj)
            if info.duration >= 30 and batch['task'] in ["ASR", "S2TT"]:
                is_readable = False 
            else:
                is_readable = True
    except:
        is_readable = False

    input_ids = [tokenizer.bos_token_id] + tokenizer("<speech>").input_ids[2:]
    attention_mask = [1] * len(input_ids)
    
    
    # end of speech tag: </speech>
    suffix_input_ids = tokenizer('</speech>').input_ids[2:]
    suffix_attention_mask = [1] * len(suffix_input_ids)
    
    # multi-task tag
    multitask_tag_ids = []
    tag_map = TASK_SPECIFIC_TAG[batch['task']]

    sot_ids = tokenizer(tag_map['SOT']).input_ids[2:]
    multitask_tag_ids += sot_ids
    audio_language = tokenizer(tag_map['AL'][batch['audio_language']]).input_ids[2:]
    multitask_tag_ids += audio_language
    task_tag = tokenizer(tag_map['TT']).input_ids[2:]
    multitask_tag_ids += task_tag
    text_language = tokenizer(tag_map['TL'][batch['text_language']]).input_ids[2:]
    multitask_tag_ids += text_language
    timestamps = tokenizer(tag_map['TS']).input_ids[2:]
    multitask_tag_ids += timestamps
    wlt = tokenizer(WLT[batch['task']]).input_ids[2:]
    multitask_tag_ids += wlt
    
    suffix_input_ids += multitask_tag_ids
    suffix_attention_mask += [1] * len(multitask_tag_ids)

    # ground_truth
    ground_truth_ids = tokenizer(batch["ground_truth"]).input_ids[1:] + [tokenizer.eos_token_id]

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["suffix_input_ids"] = suffix_input_ids
    batch["suffix_attention_mask"] = suffix_attention_mask
    batch["audio_path"] = audio_path
    batch["is_readable"] = is_readable
    batch["ground_truth_ids"] = ground_truth_ids
    return batch

def process_dataset(batch, tokenizer, instructions):
    client = MyClient()
    if batch['task'] == "asr": 
        instructions = instructions.split('/')[1:]
        instruction = random.choice(instructions)
    else:
        instruction = instructions.split('/')[0]
    input_ids = tokenizer(f"###[Human]:{instruction}").input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)

    audio_path = batch["audio"]
    try:
        s3b = client.get(audio_path)
        with io.BytesIO(s3b) as fobj: 
            info = sf.info(fobj)
            if info.duration >= 30:
                is_readable = False
            else:
                is_readable = True
    except:
        is_readable = False

    suffix_input_ids, suffix_attention_mask, suffix_labels = [], [], []
    ### \n\n\n###[Assistant]:
    new_input_ids = tokenizer("\n\n\n###[Assistant]:").input_ids[1:] # remove bos token
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += [-100] * len(new_input_ids)
    ### response
    new_input_ids = tokenizer(batch["text"]).input_ids[1:] + [tokenizer.eos_token_id]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += new_input_ids

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels
    batch["suffix_input_ids"] = suffix_input_ids
    batch["suffix_attention_mask"] = suffix_attention_mask
    batch["suffix_labels"] = suffix_labels
    batch["audio_path"] = audio_path
    batch["is_readable"] = is_readable

    return batch

def load_speech_text_paired_dataset(
    dataroot="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    num_proc=64,
    sort=False,
    shuffle=True,
    iterable=False,
    multitask=True,
    training=True,
):
    if os.path.exists(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all"))):
        logger.warning("load processed dataset")
        if iterable:
            dataset = IterableDataset.load_from_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))
            return dataset
        dataset = datasets.load_from_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        return dataset
    
    logger.warning(f"load dataset from scratch from {dataroot}/{manifest_files}")

    manifest_files_list = manifest_files.split(",")
    raw_dataset = datasets.load_dataset(
        dataroot, data_files=manifest_files_list, split="train", streaming=False
    )

    if multitask:
        dataset = raw_dataset.map(
            process_dataset_multitask if training else process_dataset_multitask_test,
            fn_kwargs={
                "tokenizer": tokenizer,
            },
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=False,
            num_proc=128,
        )
    else:
        dataset = raw_dataset.map(
            process_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "instructions": instruction
            },
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=False,
            num_proc=num_proc,
        )       

    def is_readable(flag):
        return flag
    
    dataset = dataset.filter(
        is_readable,
        input_columns=["is_readable"]
    )

    dataset.save_to_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))

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

def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3 and (meta[0].endswith(".wav") or meta[0].endswith(".flac")):
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp


    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext in [".wav", ".flac", ".ogg", ".mp3"]:
            pass
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLACC/OGG/MP3 audios")
    client = MyClient()
    s3b = client.get(path_or_fp)
    with io.BytesIO(s3b) as fobj:
        waveform, sample_rate = sf.read(
            fobj, dtype="float32", always_2d=True, frames=frames, start=start
        )
    if len(waveform) / sample_rate > 30:
        size = len(waveform)
        target_size = 30 * sample_rate
        diff = size - target_size
        start, end = 0, target_size
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        waveform = waveform[start:end]
    waveform = waveform.T

    waveform, sample_rate = convert_waveform(waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


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
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)

        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=self.sampling_rate) for sample in samples
        ]
        if isinstance(self.extractor, WhisperFeatureExtractor):
            speech_inputs = self.extractor(
                raw_speech, 
                sampling_rate=self.sampling_rate, 
                return_attention_mask=True,
                return_tensors="pt"
            )
        else:
            speech_inputs = self.extractor(
                raw_speech, 
                padding=True,
                sampling_rate=self.sampling_rate, 
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_inputs['input_features'] = speech_inputs['input_values']

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "suffix_labels": suffix_labels,
            "speech_values": speech_inputs.input_features,
            "speech_attention_mask": speech_inputs.attention_mask
        }


def offline_process(
    dataroot="",
    manifest_files="",
    lm_path="",
    instruction="",
    num_proc=128,
):
    text_tokenizer = LlamaTokenizer.from_pretrained(lm_path)
    text_tokenizer.add_tokens(SPECIA_TOKENS)
    text_tokenizer.add_tokens([val for val in WLT.values()])

    dataset = load_speech_text_paired_dataset(
        dataroot,
        manifest_files,
        text_tokenizer,
        instruction,
        num_proc
    )
    for key in dataset[0].keys():
        if key != "audio_path" and key != "is_readable":
            print(key, len(dataset[0][key]))
        else:
            print(key, dataset[0][key])
    print(len(dataset))


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })
    
