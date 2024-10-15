from tqdm import tqdm
import evaluate
from blsp.src.owntoken import SPECIA_TOKENS, WLT
from sklearn.metrics import accuracy_score


# with open('results/ser_meld', 'r') as f1:
with open('results/checkpoints/multitask_20240118_whisper_v3_internlm2_large_bn3_resume_debug_shuf_4/average_checkpoint/ClothAQA_all', 'r') as f1:
    tmps = []
    hyps, refs = [], []
    bi_refs, bi_hyps = [], []
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
                hyps.append(tmps[0].lower())
                refs.append(tmps[1].lower())
                if tmps[1].lower() in ["yes", "no"]:
                    bi_hyps.append(tmps[0].lower())
                    bi_refs.append(tmps[1].lower())
            tmps = []
    results = accuracy_score(refs, hyps)
    print("result acc: {}".format(results))
    bi_results = accuracy_score(bi_refs, bi_hyps)
    print("result bi acc: {}".format(bi_results))
        
