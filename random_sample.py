from petrel_client.client import Client
import re
import torchaudio
from tqdm import tqdm
import random
import io
import json

def remove_punctuation(text):
    # 使用正则表达式匹配除了空格和单引号之外的所有标点符号，并替换为空字符串
    return re.sub(r'[^\w\s\']', '', text)

class MyClient(object):
    def __init__(self):
        self.client = Client('~/petreloss.conf')
    
    def get (self, key, enable_stream=False):
        index = key.find("/")
        bucket = key[:index]
        new_key = key[index+1:]
        if bucket == "asr" or bucket == "exp":
            return self.client.get("asr:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == "youtubeBucket":
            return self.client.get("youtube:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == "tts":
            return self.client.get("tts:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)
        elif bucket == "ASR_20T":
            return self.client.get("tts_asr:s3://{}/".format(bucket) + new_key, no_cache=True, enable_stream=enable_stream)

        elif bucket == 's3:':
            return self.client.get("asr:{}".format(key), no_cache=True, enable_stream=enable_stream)
        else:
            with open(key, 'rb') as f:
                return f.read()
            
from joblib import Parallel, delayed

# client = MyClient()
# url = "ASR_20T/ASR_full_data/speech_annotations/20210825/bilibili/8859/1987571137/-8635868574911660367/383381662_audio-2074.wav"
# s3b = client.get(url)
# with io.BytesIO(s3b) as fobj:
#     y, sr = torchaudio.load(fobj)
# print(y, sr)
# exit()
lists = []
with open('/mnt/petrelfs/share_data/zhengken/semantic_tokens/asr_weread_150k_hrs_shuffle_20240220_train_sy.lst.filter') as f:
    for l in tqdm(f):
        lists.append([l.split('|')[0].replace('s3://', '').replace("asr:", ""), remove_punctuation(l.split('|')[1])])
# client = MyClient()
# url = "tts/weread-16k-res/1094_SVP_0bf2gaanwaaayuanrwtkbvrxymgd3myabwya_segment17.wav"
# s3b = client.get(url)
# with io.BytesIO(s3b) as fobj:
#     y, sr = torchaudio.load(fobj)
# print(y.shape, sr)
# exit()
n_job = 128

def process(l):
    text = l[1]
    # s3b = client.get(url)
    # with io.BytesIO(s3b) as fobj:
    #     y, sr = torchaudio.load(fobj)
    # print(y.shape, sr)
    # exit()
    
    if ' ' not in text and 50 < len(text) <= 60:
        return l

results = Parallel(n_jobs=n_job)(delayed(process)(file_path) for file_path in tqdm(lists))
results = [result for result in results if result is not None]
random.shuffle(results)

max_length = 432000000
cnt = 0
with open('sample_tts_asr.jsonl', 'a') as w:
    for result in results:
        cnt += len(result[1])
        if cnt <= max_length:
            data = {}
            data['task'] = "ASR"
            data['audio'] = result[0]
            data['ground_truth'] = result[1]
            data['audio_language'] = "ZH"
            data['text_language'] = "ZH"
            data = json.dumps(data, ensure_ascii=False)
            w.write(data + '\n')    
    print(cnt)