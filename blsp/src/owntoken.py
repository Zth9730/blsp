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
    '<|nl|>',
    '<|pt|>',
    '<|unknown|>',
    '<|transcribe|>',
    '<|translate|>',
    '<|analysis|>',
    '<|caption|>',
    '<|question-answer|>',
    '<|timestamps|>',
    '<|notimestamps|>',
    *[f"<|{i *0.01:.2f}|>" for i in range(3001)]
]

WLT = {
    "ASR": "<|transcription|>",
    "S2TT": "<|translation|>",
    "OSR": "<|multi_speaker_transcription|>",
    "DASR": "<|dialect_transcription|>",
    "SRWT": "<|world_level_transcription|>",
    "DID": "<|dialect_classification|>",
    "LID": "<|language_classification|>",
    "SGC": "<|<|gender_classification|>",
    "ER": "<|emotion_classification|>",
    "SV": "<|speaker_verification|>",
    "SD": "<|speaker_diarization|>",
    "SER": "<|transcription_with_entity|>",
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
        "WLT": "speech semantic info"
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
        "WLT": "audio info"
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
