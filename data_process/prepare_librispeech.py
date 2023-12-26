import json
import argparse
from tqdm import tqdm
import os
from  pathlib import Path

from torchaudio.datasets import LIBRISPEECH

from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def process(args):
    input_dir = Path(args.input_dir).absolute()
    with open(args.output_file, "w") as fout:
        for split in args.splits.split(","):
            print(f"Fetching split {split}...")
            dataset = LIBRISPEECH(input_dir.as_posix(), url=split, download=False)
            for _, _, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
                sample_id = f"{spk_id}-{chapter_no}-{utt_no:04d}"
                audio = os.path.join(input_dir, "LibriSpeech", split, f"{spk_id}", f"{chapter_no}", sample_id+".flac")
                text = normalizer(utt)
                json_string = json.dumps({
                    "task": "ASR",
                    "audio": audio,
                    "ground_truth": text,
                    "audio_language": "EN",
                    "text_language": "EN"
                })
                fout.write(json_string + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to the input directory", required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output manifest file", required=True)
    parser.add_argument("--splits", type=str, default="test-other",
                        help="the splits")
    
    args = parser.parse_args()

    process(args)