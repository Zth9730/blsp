from safetensors import safe_open
from glob import glob
import os

checkpoints_dir = "checkpoints/conversion_speech_hubert/checkpoint-5200"
safe_checkpoints = glob(os.path.join(checkpoints_dir, "*.safetensors"))

dcits = {}
for checkpoint in safe_checkpoints:
    with safe_open(checkpoint, framework="pt", device=0) as f:
        for k in f.keys():
            dcits[k] = f.get_tensor(k).clone()

import pdb
pdb.set_trace()