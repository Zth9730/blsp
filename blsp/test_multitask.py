#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import evaluate

import datasets
from datasets import interleave_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate
import torch

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
    Seq2SeqTrainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, WhisperConfig, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers.deepspeed import is_deepspeed_zero3_enabled

from src.speech_text_paired_dataset import load_speech_text_paired_dataset, SpeechTextPairedDataCollator
from src.modeling_blsp import BlspModel
from src.modeling_whisper_encoder import WhisperEncoder
from src.configuration_blsp import BlspConfig
from src.owntoken import SPECIA_TOKENS, WLT

logger = logging.getLogger(__name__)

import os
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    llama_model: str = field(
        default="decapoda-research/llama-7b-hf", metadata={"help": "the path of base model"}
    )
    whisper_model: str = field(
        default="openai/whisper-small", metadata={"help": "the path of whisper model"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data: str = field(
        metadata={
            "help": "the root to load dataset"
        },
    )
    data2: str = field(
        default="",
        metadata={
            "help": "the root to load dataset2"
        },
    )
    manifest_files: str = field(
        default="",
        metadata={
            "help": "The name of the training unit text paired set split to use."
        },
    )
    instruction: str = field(
        default="",
        metadata={
            "help": "The text prefix instruction before speech input, default None"
        },
    )


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # training_args.local_rank = int(os.environ["LOCAL_RANK"])

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # 4. Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained('pretrained_models/llama2-7b-hf')
    new_embedding_nums = 0
    new_embedding_nums += tokenizer.add_special_tokens({'additional_special_tokens': SPECIA_TOKENS})
    new_embedding_nums += tokenizer.add_special_tokens({'additional_special_tokens': [val for val in WLT.values()]})

    extractor = WhisperFeatureExtractor.from_pretrained('pretrained_models/whisper-small')
    # elif blsp_config.speech_encoder == 'mms':
    #     extractor = Wav2Vec2FeatureExtractor.from_pretrained("/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/pretrained_models/mms-1b")

    ### 5. Load dataset
    
    dataset = load_speech_text_paired_dataset(
        dataroot=data_args.data,
        manifest_files=data_args.manifest_files,
        tokenizer=tokenizer,
        instruction=data_args.instruction
    )
    

    # 6. Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently


    model = BlspModel.from_pretrained('/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/checkpoints/multitask_now_4/checkpoint-96400/', new_embedding_nums)

    # model.whisper_model = WhisperEncoder.from_pretrained(model_args.whisper_model)
    # model.llama_model = LlamaForCausalLM.from_pretrained(model_args.llama_model, _fast_init=not is_deepspeed_zero3_enabled())
    
    # model.llama_model.resize_token_embeddings(len(tokenizer))
    
    
    # if blsp_config.stage == 'multi-task':
    #     for name, param in model.llama_model.named_parameters():
    #         param.requires_grad = False
    #     model.llama_model.model.embed_tokens.new_weight.requires_grad = True
    #     model.llama_model.lm_head.new_weight.requires_grad = True
        
    # elif blsp_config.stage == 'chat':
    #     for name, param in model.whisper_model.named_parameters():
    #         param.requires_grad = False

    # 6. Define data collator
    data_collator = SpeechTextPairedDataCollator(
        pad_id=tokenizer.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor
    )


    # 7. Initialize Trainer
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    
    def compute_metrics(pred):
        import pdb
        pdb.set_trace()
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # compute orthographic wer
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
        # compute normalised WER
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

        return {"wer_ortho": wer_ortho, "wer": wer}

    tester = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # 8. Training
    output = tester.evaluate()
    print(output)

if __name__ == "__main__":
    main()