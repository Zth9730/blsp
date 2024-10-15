from tqdm import tqdm
from metrics import CocoTokenizer
from caption_evaluation_tools.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from caption_evaluation_tools.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from caption_evaluation_tools.coco_caption.pycocoevalcap.rouge.rouge import Rouge
from caption_evaluation_tools.coco_caption.pycocoevalcap.cider.cider import Cider
from caption_evaluation_tools.coco_caption.pycocoevalcap.spice.spice import Spice

def compute_caption(gts, res):
    preds_str = res
    references = gts
    tokenizer = CocoTokenizer(preds_str, references)
    res, gts = tokenizer.tokenize()

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]
    f_res = {}
    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))

        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.3f" % (m, sc))
                f_res[m] = sc
        else:
            print("%s: %0.3f" % (method, score))
            f_res[method] = score
    f_res["SPIDEr"] = (f_res['CIDEr']+f_res['SPICE']) / 2.
    return f_res

with open('results/checkpoints/debug_caption_2/checkpoint-16000/aac_clotho', 'r') as f1:
# with open("/mnt/petrelfs//zhoudinghao/work/thzhang/Qwen-Audio/eval_audio/test_qwenaudio_caption.txt",) as f1:
    dict = {}
    for i, x in tqdm(enumerate(f1)):
        if "hyp:" in x:
            if len(dict[ids]) == 0:
                dict[ids].append(x.split("hyp:")[-1].strip())
        elif "ref:" in x:
            dict[ids].append(x.split("ref:")[-1].strip())

        elif "audio_path:" in x:
            ids = x.split("/")[-1].strip()
            if ids not in dict:
                dict[ids] = []
        else:
            continue

hyps = []
refs = []
import random

for key in dict.keys():
    hyps.append(dict[key][0])
    x = dict[key][1:]
    random.shuffle(x)
    refs.append(x)

results = compute_caption(refs, hyps)
print("result wer: {}".format(results))
        