from safetensors import safe_open
from safetensors.torch import save_file
import argparse
import torch
import shutil

from glob import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Average Transformers Checkpoint Demo")
    parser.add_argument(
        "-m", "--model_dir", type=str, help="Path to the checkpoint dir")
    parser.add_argument(
        "-n", "--avg_nums", type=int, default=None,
        help="nums of the avg checkpoints"
    )
    args = parser.parse_args()
    return args

args = parse_args()
avg_dir = os.path.join(args.model_dir, "average_checkpoint")
os.makedirs(avg_dir, exist_ok=True)

checkpoints_dir = sorted(glob(os.path.join(args.model_dir, "checkpoint-*")), key=lambda x: int(x.split('-')[-1]))
if args.avg_nums is None:
    checkpoints_dir = checkpoints_dir
else:
    checkpoints_dir = checkpoints_dir[args.avg_nums:]
num = len(checkpoints_dir)
print(num)
print(checkpoints_dir)
shutil.copy(os.path.join(checkpoints_dir[0], "config.json"), avg_dir)
shutil.copy(os.path.join(checkpoints_dir[0], "generation_config.json"), avg_dir)
shutil.copy(os.path.join(checkpoints_dir[0], "model.safetensors.index.json"), avg_dir)

sub_checkpoint_name = [_.split("/")[-1] for _ in glob(os.path.join(checkpoints_dir[0], "*.safetensors"))]
for sub_checkpoint in sub_checkpoint_name:
    avg = {}
    for checkpoint in checkpoints_dir:  
        with safe_open(os.path.join(checkpoint, sub_checkpoint), framework="pt", device=0) as f:
            for k in f.keys():
                if k not in avg.keys():
                    avg[k] = f.get_tensor(k).clone()
                else:
                    avg[k] += f.get_tensor(k)
    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(os.path.join(avg_dir, sub_checkpoint)))
    save_file(avg, os.path.join(avg_dir, sub_checkpoint), metadata={"format": "pt"})
