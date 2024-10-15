from tqdm import tqdm
import evaluate
from blsp.src.owntoken import SPECIA_TOKENS, WLT
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import re
from heareval_score import Top1Accuracy, ChromaAccuracy
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np
def get_note_info(text):
    result = {}
    pattern = r'<\|pitch\|><\|midi_pitch_(.*?)\|>'
    try:
        result['pitch'] = int(re.findall(pattern, text)[0] if len(re.findall(pattern, text)) > 0 else 128)
    except:
        result['pitch'] = 128
    pattern = r'<\|velocity\|><\|midi_velocity_(.*?)\|>'
    try:
        result['velocity'] = int(re.findall(pattern, text)[0] if len(re.findall(pattern, text)) > 0 else 128)
    except:
        result['velocity'] = 128
    pattern = r'<\|instrument\|>(.*)'
    try:
        result['instrument'] = re.findall(pattern, text)[0].split("<")[0] if len(re.findall(pattern, text)) > 0 else ''
    except:
        result['instrument'] = ''
    pattern = r'<\|sonic\|>(.*)'
    try:
        result['sonic'] = re.findall(pattern, text)[0].split("<")[0] if len(re.findall(pattern, text)) > 0 else ''
    except:
        result['sonic'] = ''
    return result


with open('results/checkpoints/20240118_shuf_5_shuf_train_debug_sp/checkpoint-24000//NS', 'r') as f1:
    tmps = []
    hyps = []
    refs = []
    for i, x in enumerate(f1):
        if "hyp:" in x and len(tmps) == 0:
            tmps.append(get_note_info(x.split('hyp:')[-1].strip().replace("</s>", "").replace("<s>", "")))
        elif "ref:" in x and len(tmps) == 1:
            ref_ = x.split('ref:')[-1].strip().replace("</s>", "").replace("<s>", "")
            if ref_.endswith("<|sonic|>"):
                ref_ = ref_.replace("<|sonic|>", "")
            tmps.append(get_note_info(ref_))
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
    pitch_gt = [[res['pitch']] for res in refs]
    pitch_res = [[res['pitch']] for res in hyps]
    velocity_gt = [res['velocity'] for res in refs]
    velocity_res = [res['velocity'] for res in hyps]
    instrument_gt = [res['instrument'] for res in refs]
    instrument_res = [res['instrument'] for res in hyps]
    sonic_label = {'dark', 'distortion', 'nonlinear_env', 'bright', 'fast_decay', 'long_release',
                    'tempo-synced', 'reverb', 'multiphonic', 'percussive'}

    sonic_gt, sonic_res = [], []
    for res_id in range(len(refs)):

        if refs[res_id]['sonic'] not in [None, '']:
            sonic_gt.append([r for r in refs[res_id]['sonic'].split(',') if r in sonic_label])
        else:
            sonic_gt.append([])
        if hyps[res_id]['sonic'] not in [None, '']:
            sonic_res.append([r for r in hyps[res_id]['sonic'].split(',') if r in sonic_label])
        else:
            sonic_res.append([])
    import pdb
    pdb.set_trace()
    instrument_score = accuracy_score(instrument_gt, instrument_res)
    print("instrument_score:", instrument_score, len(instrument_gt))
    velocity_score = accuracy_score(velocity_gt, velocity_res)
    print("velocity_score:", velocity_score, len(velocity_gt))


    sonic_mul = MultiLabelBinarizer().fit_transform(sonic_gt + sonic_res)
    sonic_gt_mul = sonic_mul[:len(sonic_gt)]
    sonic_res_mul = sonic_mul[len(sonic_gt):]
    pitch_mul = MultiLabelBinarizer().fit_transform(np.array(pitch_gt + pitch_res))
    sonic_score = average_precision_score(sonic_gt_mul, sonic_res_mul, average="macro")
    print("sonic_score:", sonic_score)

    print("pitch_all", pitch_mul.shape)


    pitch_gt_mul = pitch_mul[:len(pitch_gt)]
    pitch_res_mul = pitch_mul[len(pitch_gt):]
    top1acc = Top1Accuracy({})
    score = top1acc._compute(pitch_res_mul, pitch_gt_mul)
    chromaacc = ChromaAccuracy({})
    chroma_score = chromaacc._compute(pitch_res_mul, pitch_gt_mul)

    print("pitch_score top1acc:", score, "chromaacc:", chroma_score, len(pitch_gt))
        