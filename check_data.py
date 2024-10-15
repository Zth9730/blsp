import json
import re
from tqdm import tqdm

def has_english(text):
    # 使用正则表达式匹配是否包含英文字符
    return bool(re.search('[a-zA-Z]', text))

with open("data/tts_asr_sample/sample_tts_asr.jsonl") as f, open("data/tts_asr_sample/sample_tts_asr_no_en.jsonl", 'w') as f_w:
    for x in tqdm(f):
        data = json.loads(x)
        text = data['ground_truth']
        if has_english(text):
            continue
        else:
            data = json.dumps(data, ensure_ascii=False)
            f_w.write(data + '\n')
            
            
        