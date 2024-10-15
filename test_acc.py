from tqdm import tqdm
import evaluate
from blsp.src.owntoken import SPECIA_TOKENS, WLT
from sklearn.metrics import accuracy_score


# with open('results/ser_meld', 'r') as f1:
dicts = ["tut2017_test", "meld", "VocalSound", "cochlscene"]
for xx in dicts:
    try:
        with open('results/checkpoints/20240118_shuf_5_shuf_train_debug_sp/checkpoint-86000/{}'.format(xx), 'r') as f1:
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
                        hyps.append(tmps[0].lower())
                        refs.append(tmps[1].lower())
                    tmps = []
            results = accuracy_score(refs, hyps)
            print("{} result acc: {}".format(xx, results))

    except Exception as e:
        print(e)
        
