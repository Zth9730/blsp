SPECIA_TOKENS = [
    '<speech>',
    '</speech>',
    '<|startoftranscripts|>',
    '<|startofanalysis|>',
    '<|zh|>',
    '<|en|>',
    '<|de|>',
    '<|es|>',
    '<|fr|>',
    '<|it|>',
    '<|ja|>',
    '<|ko|>',
    '<|unknown|>',
    '<|transcribe|>',
    '<|translate|>',
    '<|analysis|>',
    "<|zh_tr|>", #traditional chinese 繁体中文
    "<|zh_jh|>",  # 江淮
    "<|zh_gan|>", # 赣语
    "<|zh_wu|>",  # 吴语
    "<|zh_jin|>", # 晋语
    "<|zh_min|>",  # 闽南
    "<|zh_cp|>",  # 中原
    "<|zh_yue|>",  # 粤语
    "<|zh_lu|>",  # 山东
    "<|zh_jl|>",  # 辽宁
    "<|zh_ly|>",  # 兰州
    "<|zh_xiang|>",  # 湖南
    "<|zh_bj|>",  # 北京
    '<|caption|>',
    '<|question-answer|>',
    '<|timestamps|>',
    '<|notimestamps|>',
    "<|sil|>",
    "<|caption_clotho|>",  # Clotho caption style
    "<|caption_wavcaps|>" # Wavcaps cation style
    "<|caption_songdescriber|>", # songdescriber caption style
    # "<|audioset_ontology|>",  # Audioset ontology style
    # "<|caption_plain|>",  # plain caption
    "<|slurp_ic|>",
    "<|fluent_speech_commands_ic|>"
    "<|itn|>",  # inversed text normalized
    "<|wo_itn|>", 
    "<|startofentityvalue|>",
    "<|endofentityvalue|>",
    "<|startofentitytype|>",
    "<|endofentitytype|>",
    "<|audio_grounding|>",
    "<|startofword|>",
    "<|endofword|>",
    "<|keyword|>",
    "<|delim|>",    # delimiter of timestamps pair in audio grounding 音频理解的时间戳对儿分隔符
    "<|music_description|>",
    "<|pitch|>",  # note analysis: pitch
    *[f"<|midi_pitch_{i}|>" for i in range(128)],  # midi pitch 0-127
    "<|velocity|>",  # note analysis: velocity
    *[f"<|midi_velocity_{i}|>" for i in range(128)],  # midi velocity 0-127
    "<|sonic|>",  # note analysis:  sonic
    "<|instrument|>",  # note analysis:  instrument
    "<|speaker_meta|>",  # meta information of speaker
    "<|song_meta|>",  # meta information of song
    "<|question|>",  # AQA: question
    "<|answer|>",  # AQA: answer
    "<|choice|>",  # AQA: answer choice
    "<|scene|>",  # scene recognition
    "<|event|>",  # sound event
    "<|speech_understanding|>",  # speech language understanding
    "<|scenario|>",  # speech language understanding: scenario
    "<|action|>",  # speech language understanding: action
    "<|entities|>",  # speech language understanding: entities
    "<|speech_edit|>",  # speech edit
    "<|im_end|>",
    "<|im_start|>",
    *[f"<|{i *0.01:.2f}|>" for i in range(3001)]
]

# SPECIA_TOKENS = [
#     '<speech>',
#     '</speech>',
#     '<|startoftranscripts|>',
#     '<|startofanalysis|>',
#     '<|zh|>',
#     '<|en|>',
#     '<|de|>',
#     '<|es|>',
#     '<|fr|>',
#     '<|it|>',
#     '<|ja|>',
#     '<|ko|>',
#     '<|nl|>',
#     '<|pt|>',
#     '<|unknown|>',
#     '<|transcribe|>',
#     '<|translate|>',
#     '<|analysis|>',
#     '<|caption|>',
#     '<|question-answer|>',
#     '<|timestamps|>',
#     '<|notimestamps|>',
#     *[f"<|{i *0.01:.2f}|>" for i in range(3001)]
# ]

WLT = {
    "ASR": "<|wo_itn|>",
    "S2TT": "<|wo_itn|>",
    "OSR": "<|multi_speaker_transcription|>",
    "DASR": "<|dialect_transcription|>",
    "SRWT": "<|world_level_transcription|>",
    "DID": "<|dialect_classification|>",
    "LID": "<|language_classification|>",
    "SGC": "<|gender_classification|>",
    "ER": "<|emotion_classification|>",
    "SV": "<|speaker_verification|>",
    "SD": "<|speaker_diarization|>",
    "SER": "<|speech_entity_recognition|>",
    "KS": "<|speech_classification|>",
    "IC": "<|intent_classification|>",
    "SF": "<|slot_filling|>",
    "SAP": "<|age_prediction|>",
    "AAC": "<|audio_caption|>",
    "SEC": "<|event_classification|>",
    "ASC": "<|scene_classification|>",
    "SED": "<|time_level_event_detection|>",
    "AQA": "<|audio_question_answering|>",
    "SID": "<|singer_classification|>",
    "SMER": "<|song_emotion_classification|>",
    "MC": "<|music_caption|>",
    "SE": "<|speech_edit|>",
    "MIC": "<|music_instruments_classification|>",
    "MNA": "<|music_note_analysis|>",
    "MGR": "<|music_genre_classification|>",
    "MR": "<|music_classification|>",
    "MQA": "<|music_question_answering|>",
    "VSC": "<|vocal_sound_classification|>"
}

LANGUAGE_MAP = {
    "ZH": "<|zh|>",
    "EN": "<|en|>",
    "DE": "<|de|>",
    "ES": "<|es|>",
    "FR": "<|fr|>",
    "IT": "<|it|>",
    "JA": "<|ja|>",
    "KO": "<|ko|>",
    "NL": '<|nl|>',
    "PT": '<|pt|>',
    "UNKNOWN": "<|unknown|>"
}

TASK_SPECIFIC_TAG = {
    "ASR": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|transcribe|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "transcription"
    },
    "S2TT": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|translate|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "translation"
    },
    "OSR": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|transcribe|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "multi speaker transcription"
    },
    "DASR": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|transcribe|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "dialect transcription"
    },
    "SRWT": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|transcribe|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|timestamps|>",
        "WLT": "world level transcription"
    },
    "DID": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "dialect info"
    },
    "LID": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "language info"
    },
    "SGC": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speaker info"
    },
    "ER": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "emotion"
    },
    "SV": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speaker info"
    },
    "SD": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|timestamps|>",
        "WLT": "speaker info"
    },
    "SER": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|transcribe|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|timestamps|>",
        "WLT": "transcription with entity"
    },
    "KS": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speech classification"
    },
    "IC": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speech semantic info",
        "DID":{
            "slurp": "<|slurp_ic|>",
            "fluent_speech_commands": "<|fluent_speech_commands_ic|>"
        }
    },
    "SF": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speech semantic info"
    },
    "SAP": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speaker info"
    },
    "AAC": {
        "SOT": "<|startofanalysis|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|caption|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "audio info",
        "DID": {
            "Clotho": "<|caption_clotho|>",
            "WavCaps": "<|caption_wavcaps|>",
            "songdescriber": "<|caption_songdescriber|>"
        }
    },
    "SEC": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "sound envent"
    },
    "ASC": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "scene"
    },
    "SED": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|timestamps|>",
        "WLT": "sound envent"
    },
    "AQA": {
        "SOT": "<|startofanalysis|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|question-answer|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "audio info"
    },
    "SID": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "speaker info"
    },
    "SMER": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "emotion"
    },
    "MC": {
        "SOT": "<|startofanalysis|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|caption|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "song info"
    },
    "MIC": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "song info"
    },
    "MNA": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "song info"
    },
    "MGR": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "song info"
    },
    "MR": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "song classification"
    },
    "MQA": {
        "SOT": "<|startofanalysis|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|question-answer|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "song info"
    },
    "VSC": {
        "SOT": "<|startoftranscripts|>",
        "AL": LANGUAGE_MAP,
        "TT": "<|analysis|>",
        "TL": LANGUAGE_MAP,
        "TS": "<|notimestamps|>",
        "WLT": "vocal"
    }
}

import json
# with open("/mnt/petrelfs/zhoudinghao/work/thzhang/thzhang/wav2seq/first_save-bpe-tokenizer-vocab10000_convert.json") as f:
# with open("/mnt/petrelfs/zhoudinghao/work/thzhang/thzhang/wav2seq/cluster_ids-bpe-tokenizer-vocab10000_convert.json") as f:
# with open("/mnt/petrelfs/zhoudinghao/work/thzhang/thzhang/wav2seq/hubert_train-bpe-tokenizer-vocab10000_convert.json") as f:
#     data = json.load(f)
#     vocab = data['model']['vocab']

# AUDIO_UNIT_TOKENS = list(vocab.keys())
AUDIO_UNIT_TOKENS =  [
    "<text>",
    "</text>",
    "<audio>",
    "</audio>",
    *[f"<|audiounit-{i}|>" for i in range(10000)]
    ]
