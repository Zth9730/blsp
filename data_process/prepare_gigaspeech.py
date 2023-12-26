import pathlib
import json
import argparse
import tqdm
import os
import csv
import glob
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def process(args):
    with open(args.output_file, "w") as fout:
        for split in ['test']:
            if split in ['xs', 'test', 'dev']:
                tags = ''
            else:
                tags = '_additional'
            csv_files = glob.glob("{}/{}_metadata{}/*.csv".format(args.input_dir,split, tags))
            for csv_file in tqdm.tqdm(csv_files):
                csv_ids = csv_file.split('/')[-1].split('.')[0].replace('_metadata', '')
                with open(csv_file, "r") as fin:
                    data = csv.DictReader(fin)
                    for segment in data:
                        sid = segment["sid"]
                        text = segment["text_tn"]
                        path = os.path.join("asr/public-dataset/GigaSpeech/data/audio","{}_files{}".format(split, tags), csv_ids, sid+".wav")
                        text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".")
                        text = text.replace("<QUESTIONMARK>", "?").replace("<EXCLAMATIONPOINT>", "!")
                        text = normalizer(text)
                        json_string = json.dumps({
                            "task": "ASR",
                            "audio": path,
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
    
    args = parser.parse_args()

    process(args)