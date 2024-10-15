from transformers import PreTrainedModel, LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
import torch
import torch.nn as nn
from pathlib import Path
from prompt_toolkit import prompt
from blsp.src.owntoken import AUDIO_UNIT_TOKENS
from einops import rearrange, pack, unpack
import fairseq
import torchaudio
import joblib
from torchaudio.functional import resample
import ChatTTS
import datetime
device = torch.device('cuda')

def resample(waveform, sample_rate, resample_rate=16000):
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return waveform

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path,
        kmeans_path,
        target_sample_hz = 16000,
        seq_len_multiple_of = None,
        output_layer = 9
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)

        assert model_path.exists(), f'path {checkpoint_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        self.model.eval()

        kmeans = joblib.load(kmeans_path)
        self.kmeans = kmeans

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @torch.no_grad()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None,
        return_feature = False,
    ):
        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model(
            wav_input,
            features_only = True,
            mask = False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
            output_layer = self.output_layer
        )

        embed, packed_shape = pack([embed['x']], '* d')
        if return_feature:
            return embed
        codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

        codebook_indices = torch.from_numpy(codebook_indices).to(device).long()

        if flatten:
            return codebook_indices

        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        return codebook_indices
checkpoint_path = "/mnt/hwfile/speech/zhengken/kmeans_models/v2.0/checkpoint_last.pt"
kmeans_path = "/mnt/hwfile/speech/zhengken/kmeans_models/v2.0/xmly_mls_aishell2_librispeech_novel_dubbing_23kh_9L_km10000_v2.0.mdl"
kmeans_model = HubertWithKmeans(checkpoint_path, kmeans_path, output_layer=9).cuda()
chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

tokenizer = LlamaTokenizer.from_pretrained('checkpoints/conversion_speech_hubert_new')
# tokenizer.add_special_tokens({"additional_special_tokens": AUDIO_UNIT_TOKENS})

model = LlamaForCausalLM.from_pretrained('checkpoints/speeech_hubert_new_emo_cap/checkpoint-7300')

System_Content = (
    "###[System]:{content}\n\n\n"
)

Text_Format = (
    "###[Human]:{instruction}\n\n\n"
    "###[Assistant]:"
)

generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
    temperature=1.0,
    top_p=0.95,
    num_beams=1,
    num_return_sequences=1,

)

def process_dataset(tokenizer, instruction, instruction2=None, type="text"):
    if type == "text":
        instruction = "<text>" + instruction + "</text>"
    elif type == "audio":
        instruction = "<audio>" + instruction + "</audio>"
    elif type == "mix":
        instruction = "<text>" + instruction + "</text>" + "<audio>" + instruction2 + "</audio>"
    
    input = Text_Format.format(instruction=instruction)

    input_ids = tokenizer(input, return_tensors="pt").input_ids[:, 1:]
    return input_ids

def process_audio(audio_instruction):
    if not Path(audio_instruction).is_file():
        wav = chat.infer([audio_instruction])[0]
        wav = torch.from_numpy(wav)
        wav_input = resample(wav, 24000).cuda()
        torchaudio.save("chattts_sys/{}.wav".format(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")), wav_input.cpu(), 16000)
    else:
        wav_input, sr = torchaudio.load(audio_instruction)
        wav_input = resample(wav_input, sr).cuda()
    
    ids = kmeans_model(wav_input).cpu().tolist()

    return "".join([f"<|audiounit-{x}|>" for x in ids])

multi_stage = False
system = "<|system|>you are a helpful speech assistant, you should understand input speech and help user<|/system|>"
system_input = System_Content.format(content=system)
start_ids = tokenizer(system_input, return_tensors="pt").input_ids
while True:
    if not multi_stage:
        history = start_ids
    input_type = prompt('tell me input type (text, audio, mix):')
    try:
        assert input_type in ["text", "audio", "mix"]
    except:
        continue
    if input_type == "mix":
        instruction = prompt('tell me text instruction:')
        instruction2 = prompt('tell me audio instruction:')
        instruction2 = process_audio(instruction2)
    else:
        instruction = prompt(f'please tell me {input_type} instruction:')
        if input_type == "audio":
            instruction = process_audio(instruction)
        instruction2 = None

    input_ids = torch.concat([history, process_dataset(tokenizer, instruction, instruction2, input_type)], dim=-1)
    generate_ids = model.generate(input_ids, generation_config=generation_config, eos_token_id=tokenizer.eos_token_id)
    import pdb
    pdb.set_trace()
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].split("###[Assistant]:")[-1].strip()
    print(result)
    prompt_multi_stage = prompt('if begin multi stage conversation? (true, false):')
    if prompt_multi_stage == "true":
        history = torch.concat([input_ids, tokenizer(result + tokenizer.eos_token, return_tensors="pt").input_ids[:, 1:]], dim=-1)
        multi_stage = True
    else:
        multi_stage = False


### interleave
# while True:
#     human_input_type = prompt('tell me input type of human (text, audio, mix):')
#     try:
#         assert human_input_type in ["text", "audio", "mix"]
#     except:
#         continue
#     if human_input_type == "mix":
#         instruction = prompt('tell me text instruction:')
#         instruction2 = prompt('tell me audio instruction:')
#         instruction2 = process_audio(instruction2)
#     else:
#         instruction = prompt(f'please tell me {human_input_type} instruction:')
#         if human_input_type == "audio":
#             instruction = process_audio(instruction)
#         instruction2 = None
#     human_input_ids = torch.concat([start_ids, process_dataset(tokenizer, instruction, instruction2, human_input_type)], dim=-1)


#     instruction = prompt(f'please tell me assistant instruction:')
#     assistant_input_ids = torch.concat([human_input_ids, tokenizer(instruction, return_tensors="pt").input_ids[:,1:]], dim=-1)

#     real_input_type = prompt('tell me input type of real (text, audio, mix):')
#     try:
#         assert real_input_type in ["text", "audio", "mix"]
#     except:
#         continue
#     if real_input_type == "mix":
#         instruction = prompt('tell me text instruction:')
#         instruction2 = prompt('tell me audio instruction:')
#         instruction2 = process_audio(instruction2)
#     else:
#         instruction = prompt(f'please tell me {real_input_type} instruction:')
#         if real_input_type == "audio":
#             instruction = process_audio(instruction)
#         instruction2 = None

#     input_ids = torch.concat([assistant_input_ids, process_dataset(tokenizer, instruction, instruction2, real_input_type)], dim=-1)
#     generate_ids = model.generate(input_ids, generation_config=generation_config, eos_token_id=tokenizer.eos_token_id)
#     import pdb
#     pdb.set_trace()
#     result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
#     print(result)