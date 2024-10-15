import io
import numpy as np
import torch    
from petrel_client.client import Client
import soundfile as sf
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
        elif bucket == 'emotion-data':
            return self.client.get("1988:{}".format(key), no_cache=True, enable_stream=enable_stream)
        else:
            with open(key, 'rb') as f:
                return f.read()


client = MyClient()
path_or_fp = "exp/speech_llm/audioset_strong/audiosetdata/Accelerating,revving,vroom/snippedYM4fHpRTVN4k.wav"
s3b = client.get(path_or_fp)
with io.BytesIO(s3b) as fobj:
    waveform, sample_rate = sf.read(
        fobj, dtype="float32", always_2d=True
    )

# def convert_waveform(
#     waveform,
#     sample_rate: int,
#     normalize_volume: bool = False,
#     to_mono: bool = False,
#     to_sample_rate = None,
# ):
#     """convert a waveform:
#     - to a target sample rate
#     - from multi-channel to mono channel
#     - volume normalization
#     Args:
#         waveform (numpy.ndarray or torch.Tensor): 2D original waveform
#             (channels x length)
#         sample_rate (int): original sample rate
#         normalize_volume (bool): perform volume normalization
#         to_mono (bool): convert to mono channel if having multiple channels
#         to_sample_rate (Optional[int]): target sample rate
#     Returns:
#         waveform (numpy.ndarray): converted 2D waveform (channels x length)
#         sample_rate (float): target sample rate
#     """
#     try:
#         import torchaudio.sox_effects as ta_sox
#     except ImportError:
#         raise ImportError("Please install torchaudio: pip install torchaudio")

#     effects = []
#     if normalize_volume:
#         effects.append(["gain", "-n"])
#     if to_sample_rate is not None and to_sample_rate != sample_rate:
#         effects.append(["rate", f"{to_sample_rate}"])
#     if to_mono and waveform.shape[0] > 1:
#         effects.append(["channels", "1"])
#     if len(effects) > 0:
#         is_np_input = isinstance(waveform, np.ndarray)
#         _waveform = torch.from_numpy(waveform) if is_np_input else waveform
#         converted, converted_sample_rate = ta_sox.apply_effects_tensor(
#             _waveform, sample_rate, effects
#         )
#         if is_np_input:
#             converted = converted.numpy()
#         return converted, converted_sample_rate
#     return waveform, sample_rate


# audio_path1 = "ASR_20T/ASR_full_data/ASR/dialog/speech_dialog_p2_006880_0116.wav"
# audio_path2 = "asr/public-dataset/aishell/wav/train/S0336/BAC009S0336W0351.wav"
# audio_path3 = "tts/weread-16k-res/1094_SVP_0bc3oaalyaaanaapailgc5shy4gdxryabpaa_segment36.wav"
# s3b = client.get(audio_path1)
# with io.BytesIO(s3b) as fobj: 
#     waveform, sample_rate = sf.read(
#             fobj, dtype="float32", always_2d=True, frames=-1, start=0
#         )
# print(sample_rate)
#     # convert_waveform(waveform, sample_rate,to_sample_rate=8000)
#     # print(waveform)