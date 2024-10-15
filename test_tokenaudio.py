from torch.utils.data import DataLoader
import evaluate
from safetensors import safe_open

import datasets
import torch
from blsp.src.token_speech_text_paired_dataset import load_speech_text_paired_dataset, collate_tokens
from blsp.src.modeling_blsp import BlspModel, PartiallyTrainedEmbedding, PartiallyTrainedLMHead
from blsp.src.owntoken import SPECIA_TOKENS, WLT, AUDIO_UNIT_TOKENS
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from glob import glob

from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    LlamaConfig, 
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import GenerationConfig
from tqdm import tqdm
import os
import logging

import argparse

# 创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加一个参数
parser.add_argument('--input', help='input file')

# 解析命令行参数
args = parser.parse_args()


def setup_logger(log_file):
    # 创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 创建一个文件handler，用于写入日志文件
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # 创建一个控制台handler，用于输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将handler添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

model_path = '/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/checkpoints/hubert_backup'
os.environ['HF_EVALUATE_OFFLINE'] = '1'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# 创建文件处理器
data_path = args.input
logger_file_name = os.path.join("results", model_path)
os.makedirs(logger_file_name, exist_ok=True) 
logger_file_name = logger_file_name + '/' + data_path.split("/")[-1] + "2"
print(logger_file_name)
exit()
logger = setup_logger(logger_file_name)

if data_path == "data/internlm2_evaluate/ClothAQA_all":
    batch_size = 1
else:
    batch_size=12
    
generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
    temperature=1.0,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)

def main():
    tokenizer = AutoTokenizer.from_pretrained('pretrained_models/llama2-hf-7b', trust_remote_code=True)
    origin_vocab_size = tokenizer.vocab_size
    new_embedding_nums = tokenizer.add_special_tokens({"additional_special_tokens": SPECIA_TOKENS + [val for val in WLT.values()] + AUDIO_UNIT_TOKENS})
    generation_config.update(
        **{
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )
    config = LlamaConfig.from_pretrained("pretrained_models/llama2-hf-7b")
    model = LlamaForCausalLM(config)
    if new_embedding_nums > 0:
        model.model.embed_tokens = PartiallyTrainedEmbedding(new_embedding_nums, *model.model.embed_tokens.weight.shape, model.model.embed_tokens.padding_idx)
        model.lm_head = PartiallyTrainedLMHead(new_embedding_nums, *model.model.embed_tokens.weight.shape[::-1], bias=False)

    sub_checkpoint_name = [_ for _ in glob(os.path.join(model_path, "*.safetensors"))]
    torch_dicts = {}
    for sub_checkpoint in sub_checkpoint_name:
        with safe_open(sub_checkpoint, framework="pt", device=0) as f_safe:
            for k in f_safe.keys():
                torch_dicts[k] = f_safe.get_tensor(k)
    model.load_state_dict(torch_dicts)
    model.eval()
    model.to(device)

    dataset = load_speech_text_paired_dataset(
        dataroot=data_path,
        manifest_files="*.jsonl",
        tokenizer=tokenizer,
        instruction="",
        training=False,
        interleave=True
    )
    
    def _collate_fn(samples):
        
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]

        input_ids = collate_tokens(input_ids, tokenizer.pad_token_id)
        attention_mask = collate_tokens(attention_mask, 0)

        ground_truth_ids = collate_tokens([sample["ground_truth_ids"] for sample in samples], tokenizer.pad_token_id)
        
        # audio_path = [sample["audio_path"] for sample in samples]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ground_truth_ids": ground_truth_ids,
            # "audio_path": audio_path,
        }
        
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)
    
    # 7. Initialize Trainer
    # metric = evaluate.load(f'metric/cer.py')
    metric = evaluate.load(f'metric/wer.py')
    normalizer = BasicTextNormalizer()
    if data_path.split("/")[-1] in ["NS", "sed_audioset"]:
        skip_special_tokens = False
    else:
        skip_special_tokens = True
    for batch in tqdm(eval_dataloader):
        # audio_path = batch.pop('audio_path')
        batch = {k: v.to(device) for k, v in batch.items()}
        ground_truth_ids = batch.pop('ground_truth_ids')
        with torch.no_grad():
            outputs = model.generate(generation_config=generation_config, **batch)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
        references = tokenizer.batch_decode(ground_truth_ids, skip_special_tokens=skip_special_tokens)
        for i, _tmps in enumerate(zip(predictions, references)):
            prediction, reference = _tmps
            if reference == "":
                predictions[i] == ""
            else:
                # logger.info("audio_path: " + audio_path[i])
                logger.info("hyp: " + prediction)
                logger.info("ref: " + reference + "\n")
        metric.add_batch(predictions=predictions, references=references)
    results = metric.compute()

    logger.info("result wer: {}".format(results))
if __name__ == "__main__":
    main()
