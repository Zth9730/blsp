import json
with open('./meld_eval.jsonl') as f1, open('meld_test.jsonl', 'w') as f2:
    for l in f1:
        data = json.loads(l)
        data['task'] = 'ER'
        data['ground_truth'] = data['gt']
        data['text_language'] = "EN"
        data['audio_language'] = "EN"
        data['audio'] = 'exp/speech_llm/' + data['audio']
        del data['gt']
        data = json.dumps(data)
        f2.write(data+ '\n')
