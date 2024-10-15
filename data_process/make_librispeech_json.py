import json

with open("librispeech_wo_itn_train.jsonl") as f, open("librispeech_wo_itn_train2.jsonl", 'w') as f2:
    for line in f:
        data = json.loads(line)
        audio = data['audio']
        audio = "asr/public-dataset/librispeech/" + audio.split("/")[0] + '/' + audio.split("/")[-1]
        audio = audio.replace(".flac", ".wav")
        data['audio'] = audio
        data = json.dumps(data)
        f2.write(data + '\n')
        