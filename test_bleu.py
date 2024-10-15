from tqdm import tqdm
import evaluate
from blsp.src.owntoken import SPECIA_TOKENS, WLT

from transformers import (

    AutoTokenizer,
)
import sacrebleu
tokenizer = AutoTokenizer.from_pretrained('pretrained_models/internlm2-7b', trust_remote_code=True)
new_embedding_nums = tokenizer.add_special_tokens({"additional_special_tokens": SPECIA_TOKENS + [val for val in WLT.values()]})
metric = evaluate.load(f'metric/bleu.py')

with open('results/checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_3/checkpoint-19000/covost2_de-en', 'r') as f1:
    tmps = []
    hyps = []
    refs = []
    for i, x in enumerate(f1):
        if "hyp:" in x and len(tmps) == 0:
            tmps.append(x.split('hyp:')[-1].strip())
        elif "ref:" in x and len(tmps) == 1:
            tmps.append(x.split("ref:")[-1].strip())
        elif "audio_path:" in x:
            continue
        else:
            if len(tmps) != 2:
                tmps = []
                continue
            if tmps[0] != "" and tmps[1] != "":
                hyps.append(tmps[0])
                refs.append(tmps[1])
            tmps = []
    results = sacrebleu.corpus_bleu(hyps,[refs], tokenize='13a').score
    print("result wer: {}".format(results))
        