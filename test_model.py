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
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, WhisperConfig, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import GenerationConfig
from tqdm import tqdm
import os
os.environ['HF_EVALUATE_OFFLINE'] = '1'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = '/mnt/petrelfs/zhoudinghao/work/thzhang/blsp/checkpoints/multitask_now_4/checkpoint-96400/'
model_path = 'checkpoints/20231211-traintag_newtag_4/checkpoint-30600'
data_path = 'data/test_set/asr/gigaspeech/test'
# data_path = 'data/test_set/asr/librispeech/test-clean'
batch_size=16
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
    tokenizer = LlamaTokenizer.from_pretrained('pretrained_models/llama2-7b-hf')
    new_embedding_nums = 0
    added_tokens = SPECIA_TOKENS + [val for val in WLT.values()]
    new_embedding_nums += tokenizer.add_special_tokens({'additional_special_tokens': added_tokens})
    generation_config.update(
        **{
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )
    extractor = WhisperFeatureExtractor.from_pretrained('pretrained_models/whisper-small')
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
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "speech_values": speech_inputs.input_features,
            "speech_attention_mask": speech_inputs.attention_mask,
            "ground_truth_ids": ground_truth_ids
        }
        
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)
    
    # 7. Initialize Trainer
    metric = evaluate.load(f'metrics/wer.py')
    normalizer = BasicTextNormalizer()
    
    # def compute_metrics(pred):
    #     import pdb
    #     pdb.set_trace()
    #     pred_ids = pred.predictions
    #     label_ids = pred.label_ids

    #     # replace -100 with the pad_token_id
    #     label_ids[label_ids == -100] = tokenizer.pad_token_id

    #     # we do not want to group tokens when computing the metrics
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    #     # compute orthographic wer
    #     wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
    #     # compute normalised WER
    #     pred_str_norm = [normalizer(pred) for pred in pred_str]
    #     label_str_norm = [normalizer(label) for label in label_str]
    #     # filtering step to only evaluate the samples that correspond to non-zero references:
    #     pred_str_norm = [
    #         pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    #     ]
    #     label_str_norm = [
    #         label_str_norm[i]
    #         for i in range(len(label_str_norm))
    #         if len(label_str_norm[i]) > 0
    #     ]

    #     wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    #     return {"wer_ortho": wer_ortho, "wer": wer}

    for batch in tqdm(eval_dataloader):

        batch = {k: v.to(device) for k, v in batch.items()}
        ground_truth_ids = batch.pop('ground_truth_ids')

        with torch.no_grad():
            outputs = model.generate(generation_config=generation_config, **batch)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(ground_truth_ids, skip_special_tokens=True)
        for i, _tmps in enumerate(zip(predictions, references)):
            prediction, reference = _tmps
            if reference == "":
                predictions[i] == ""
            else:
                print("ids: " + batch[i])
                print("hyp: " + prediction)
                print("ref: " + reference)
        metric.add_batch(predictions=predictions, references=references)
    resutls = metric.compute()
    print(resutls)
if __name__ == "__main__":
    main()