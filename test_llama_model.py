from torch.utils.data import DataLoader
import evaluate
import datasets
import torch
from blsp.src.speech_text_paired_dataset import load_speech_text_paired_dataset, collate_tokens, get_waveform
from blsp.src.modeling_blsp import BlspModel
from blsp.src.modeling_whisper_encoder import WhisperEncoder
from blsp.src.configuration_blsp import BlspConfig
from blsp.src.owntoken import SPECIA_TOKENS, WLT
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from sklearn.metrics import accuracy_score

from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    LlamaConfig, 
    WhisperConfig, 
    WhisperFeatureExtractor, 
    Wav2Vec2FeatureExtractor,
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

model_path = 'checkpoints/librispeech_train_train_tag_test'
# model_path = 'checkpoints/20231211-traintag_newtag_4/checkpoint-30600'
model_path = '/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/checkpoints/librispeech_train_train_tag_test_2_notag_3'
model_path = 'checkpoints/multitask_20240108_asr_aac_mc/checkpoint-66000'
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_resume/checkpoint-49000"
# model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn2/checkpoint-6000"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3/checkpoint-21000"
model_path= "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3/checkpoint-253000"
# model_path= "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume/checkpoint-310000"
# model_path = "debug/checkpoint-21000"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_3/checkpoint-44000"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_4/checkpoint-41000"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_4/checkpoint-36000"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_5/new_average_checkpoint"
# model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_4_caption_train/checkpoint-2000"
# model_path = "/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_4_ns_train/checkpoint-20000"
# model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_5_emo_train/checkpoint-2000"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_4_caption_train/average_checkpoint"
model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_5_emo_train_debug_sp/checkpoint-1000"
model_path = "checkpoints/20240118_shuf_5_shuf_train_debug_sp/checkpoint-86000"
model_path = "checkpoints/only_for_caption_2_no_sp/checkpoint-111000"
model_path = "checkpoints/only_for_caption_2_no_sp_llama2_2/checkpoint-2000"
# model_path = "checkpoints/fbank_asr/checkpoint-14000"
# model_path = "checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_5_emo_train/average_checkpoint"
os.environ['HF_EVALUATE_OFFLINE'] = '1'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# 创建文件处理器
logger_file_name = "results/librispeech_test_traintask.log"

logger_file_name = "results/gigaspeech_test_traintask.log"
logger_file_name = "results/librispeech_train_librispeech_test.log"
logger_file_name = 'results/librispeech_train_large_v3_test.log'
logger_file_name = 'results/librispeech_train_large_v3_test-other.log'
logger_file_name = 'results/librispeech_interleave_test_clean'
logger_file_name = 'results/librispeech_test_clean_asr_acc'
logger_file_name = 'results/librispeech_test_other_asr_acc'
logger_file_name = 'results/gigaspeech_test_other_asr_acc'
logger_file_name = 'results/gigaspeech_dev_other_asr_acc'
logger_file_name = "results/aishell1_dev_new"
# logger_file_name = "results/internlm2_librispeech_dev_clean"
# logger_file_name = "results/SRWT_gigaspeech_test"
logger_file_name = "results/covost2_zh-en"
logger_file_name = "results/aac_clotho"
logger_file_name = "results/ser_meld"
logger_file_name = "results/aqa_coltho_train"
# logger_file_name = "results/vocalsound"
# logger_file_name = "results/covost2_en-zh"
# # logger_file_name = "results/covost2_en-de"
# # logger_file_name = "results/covost2_de-en"
# logger_file_name = "results/cochlscene_new"
# # logger_file_name = "results/tut2017_test"
# # logger_file_name = "results/ns"
# # logger_file_name = "results/tut2017_train"
# # logger_file_name = "results/aishell1_test_new"
logger_file_name = "results/aishell_test_new_new_debug"
logger_file_name = "results/aishell2_android_test"
logger_file_name = "results/emo_only_train_train"
logger_file_name = "results/stage1_librispeech_test_clean"
logger_file_name = "results/stage2_librispeech_test_clean"
logger_file_name = "results/stage1_aishell_test"
logger_file_name = "results/stage2_aishell_test"
logger_file_name = "results/stage1_meld"
# logger_file_name = "results/stage1_VocalSound" 
logger_file_name = "results/stage1_tut2017" 
logger_file_name = "results/stage1_cochlscene"
logger_file_name = "results/stage1_ns"
# logger_file_name = "results/stage1_librispeech_test_other"
# logger_file_name = "results/stage2_librispeech_test_other"
# logger_file_name = "results/stage2_VocalSound" 
# logger_file_name = "results/stage2_tut2017" 
# logger_file_name = "results/stage2_cochlscene"
logger_file_name = "results/stage2_meld"
# logger_file_name = "results/stage2_aac"

logger_file_name = "results/stage2_aishell_test_21000"
logger_file_name = "results/stage2_VocalSound_21000" 
logger_file_name = "results/stage2_tut2017_21000" 
# logger_file_name = "results/stage2_cochlscene_21000"
logger_file_name = "results/stage2_meld_21000"
logger_file_name = "results/aqa_coltho_stage2_21000"

logger_file_name = "results/stage1_aac_cloth"

logger_file_name = "results/stage2_VocalSound_addition_asr" 
logger_file_name = "results/stage2_cochlscene_addition_asr" 
logger_file_name = "results/stage2_tut2017_addition_asr" 
logger_file_name = "results/stage2_NS_addition_asr" 
logger_file_name = "results/stage2_aishell2_android_test_addition_asr" 
logger_file_name = "results/stage2_aac_clotho_addition_asr" 
logger_file_name = "results/stage2_ClothoAQA_addition_asr" 
logger_file_name = "results/stage2_meld_addition_asr" 
logger_file_name = "results/stage2_librispeech_clean_addition_asr" 
logger_file_name = "results/stage2_librispeech_other_addition_asr" 
logger_file_name = os.path.join("results", model_path)
os.makedirs(logger_file_name, exist_ok=True) 

data_path = 'data//asr/gigaspeech/dev'
# data_path = 'data/test_set/asr/librispeech/test-other'
# data_path = "data/internlm2_evaluate/SRWT_wenetspeech"
# data_path = "data/internlm2_evaluate/librispeech_test_other"
data_path = "data/internlm2_evaluate/aishell_asr"
# data_path = 'data/internlm2_evaluate/covost2_zh-en'
# data_path = "data/internlm2_evaluate/aqa_coltho_train"
# data_path = "data/internlm2_evaluate/VocalSound"
# data_path = "data/internlm2_evaluate/covost2_en-zh"
# # data_path = "data/internlm2_evaluate/covost2_en-de"
# # data_path = "data/internlm2_evaluate/covost2_de-en"
# data_path = "data/internlm2_evaluate/cochlscene"
# data_path = "data/internlm2_evaluate/tut2017_test"
data_path = "data/internlm2_evaluate/NS"
# data_path = "data/internlm2_evaluate/tut_train"
# data_path = 'data/internlm2_evaluate/aishell2_android_test'
# data_path = 'data/internlm2_evaluate/meld_train/'
# data_path = "data/internlm2_evaluate/aac_clotho"
# data_path = "data/internlm2_evaluate/ClothoAQA"
# data_path = "data/internlm2_evaluate/meld"
# data_path = "data/internlm2_evaluate/librispeech_test_other"
# data_path = "data/internlm2_evaluate/librispeech_test_clean"
data_path = "data/internlm2_evaluate/ClothAQA_all"
data_path = args.input
logger_file_name = logger_file_name + '/' + data_path.split("/")[-1]

logger = setup_logger(logger_file_name)

if data_path == "data/internlm2_evaluate/ClothAQA_all":
    batch_size = 1
else:
    batch_size=32
    
generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
    temperature=0.9,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)

def main():
    tokenizer = AutoTokenizer.from_pretrained('pretrained_models/llama2-hf-7b', trust_remote_code=True)
    new_embedding_nums = tokenizer.add_special_tokens({"additional_special_tokens": SPECIA_TOKENS + [val for val in WLT.values()]})
    generation_config.update(
        **{
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )
    extractor = WhisperFeatureExtractor.from_pretrained('pretrained_models/whisper-large-v3')
    dataset = load_speech_text_paired_dataset(
        dataroot=data_path,
        manifest_files="*.jsonl",
        tokenizer=tokenizer,
        instruction="",
        training=False,
    )
    model = BlspModel.from_pretrained(model_path, new_embedding_nums)
    model.eval()
    model.to(device)
    
    def _collate_fn(samples):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]

        input_ids = collate_tokens(input_ids, tokenizer.pad_token_id)
        attention_mask = collate_tokens(attention_mask, 0)
        suffix_input_ids = collate_tokens(suffix_input_ids, tokenizer.pad_token_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)

        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=16000) for sample in samples
        ]
        
        speech_inputs = extractor(
            raw_speech, 
            sampling_rate=16000, 
            return_attention_mask=True,
            return_tensors="pt"
        )

        ground_truth_ids = collate_tokens([sample["ground_truth_ids"] for sample in samples], tokenizer.pad_token_id)
        audio_path = [sample["audio_path"] for sample in samples]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "speech_values": speech_inputs.input_features,
            "speech_attention_mask": speech_inputs.attention_mask,
            "ground_truth_ids": ground_truth_ids,
            "audio_path": audio_path
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
        audio_path = batch.pop('audio_path')
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
                logger.info("audio_path: " + audio_path[i])
                logger.info("hyp: " + prediction)
                logger.info("ref: " + reference + "\n")
        metric.add_batch(predictions=predictions, references=references)
    # results = metric.compute(tokenizer=tokenizer)
    results = metric.compute()

    logger.info("result wer: {}".format(results))
if __name__ == "__main__":
    main()
