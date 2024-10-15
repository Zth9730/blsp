from tqdm import tqdm
import evaluate
from blsp.src.owntoken import SPECIA_TOKENS, WLT
import json

cer_list = ["aishell_asr", "aishell_dev", "aishell2_android_test", "aishell2_mic_test", "aishell2_ios_test"]
wer_list = ["librispeech_test_clean","librispeech_test_other", "librispeech_dev_clean","librispeech_dev_other"]
all_list = cer_list + wer_list
for xx in all_list:
    if xx in cer_list:
        metric = evaluate.load(f'metric/cer.py')
    else:
        metric = evaluate.load(f'metric/wer.py')
    try:
        with open('results/checkpoints/hubert/checkpoint-1300/{}'.format(xx), 'r') as f1:
            tmps = []
            for i, x in enumerate(f1):
                if "hyp:" in x and len(tmps) == 0:
                    tmps.append(x.split('hyp:')[-1].strip().replace('<s>', '').replace('</s>', ''))
                elif "ref:" in x and len(tmps) == 1:
                    tmps.append(x.split("ref:")[-1].strip().replace('<s>', '').replace('</s>', ''))
                elif "audio_path:" in x:
                    continue
                else:
                    if len(tmps) != 2:
                        tmps = []
                        continue
                    if tmps[0] != "" and tmps[1] != "":
                        metric.add(predictions=tmps[0], references=tmps[1])
                    tmps = []
            results = metric.compute()
            print("{} result: {}".format(xx, results))
    except:
        print("{} error".format(xx))
                