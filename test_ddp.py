import torch.distributed as dist

import torch
import json
import gc 
import os
import logging
import time

from torch.utils.data.datapipes.iter import Mapper
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data import datapipes


import os
from torch.utils.data import DataLoader

from evaluate_function import *

function_map = {
    "aishell_asr": test_cer,
    "aishell_dev": test_cer,
    "aishell2_ios_test": test_cer,
    "aishell2_mic_test": test_cer,
    "aishell2_android_test": test_cer,
    "aac_clotho": test_caption,
    "ClothAQA_all": test_aqa,
    "cochlscene": test_acc,
    "covost2_de-en": test_bleu,
    "covost2_en-de": test_bleu,
    "covost2_en-zh": test_bleu,
    "covost2_zh-en": test_bleu,
    "covost2_fr-en": test_bleu,
    "covost2_it-en": test_bleu,
    "covost2_es-en": test_bleu,
    "librispeech_dev_clean": test_wer,
    "librispeech_dev_other": test_wer,
    "librispeech_test_clean": test_wer,
    "librispeech_test_other": test_wer,
    "meld": test_acc,
    "NS": test_ns,
    "tut2017_test": test_acc,
    "VocalSound": test_acc
}

evaluate_list = [os.path.join("data/internlm2_evaluate", _) for _ in function_map.keys()]

def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    return world_size, local_rank, rank

class TextLineDataPipe(IterDataPipe):
    """ Streamming Text line
    """

    def __init__(self, filenames, mode='r'):
        super().__init__()
        _dp = datapipes.iter.FileLister(filenames)
        _dp = datapipes.iter.FileOpener(_dp, mode=mode)
        self.dp = _dp

    def __iter__(self):
        for fname, stream in self.dp:
            for line in stream:
                line = line.strip('\n')
                yield {"file_name": fname, "line": line}
            stream.close()

@functional_datapipe("map_ignore_error")
class MapperIgnoreErrorDataPipe(Mapper):

    def __init__(self,
                 dataset: IterDataPipe,
                 fn,
                 input_col=None,
                 output_col=None,
                 log_error: bool = True) -> None:
        super().__init__(dataset, fn, input_col, output_col)
        self._iter = None
        self.log_error = log_error

    def __iter__(self):
        if self._iter is None:
            self._iter = iter(self.datapipe)

        while True:
            try:
                elem = next(self._iter)
                yield self._apply_fn(elem)
            except StopIteration:
                self._iter = None
                return
            except Exception as ex:
                if self.log_error:
                    logging.warning(str(ex))

def parse_line(elem):
    line = elem['line']
    segs = line.split(',')
    name = segs[0]
    Id = segs[1]
    elem['name'] = name
    elem['id'] = Id
    return elem

def decode_audio(elem):
    import yt_dlp
    ydl = yt_dlp.YoutubeDL()
    videos = ydl.extract_info(elem['id'], download=False)
    if videos['subtitles']: 
        has_subtitles = True
    else:
        has_subtitles = False
    elem['has_subtitles'] = has_subtitles
    return elem


if __name__ == '__main__':
    world_size, _, rank =init_distributed()
    
    file_name = '/mnt/afs/ocr/workspace/wavlist1.key.todo.map.20240126'
    file_name='/mnt/afs/ocr/workspace/wavlist1.key.todo.map.20240126.v2'
    file_name='/mnt/afs/zhoudinghao/data/subtitles/粤语/喜马拉雅/do.list'
    file_name='/mnt/afs/zhoudinghao/data/subtitles/粤语/喜马拉雅/do.list'
    file_name = '/mnt/afs/zhoudinghao/work/wenet3/wenet/examples/youtube/youtube.list.txt'
    dataset = TextLineDataPipe(file_name).sharding_filter()
    dataset = dataset.map_ignore_error(parse_line).map_ignore_error(decode_audio)
    dataloader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=False,
                            num_workers=4,
                            persistent_workers=True,
                            collate_fn=lambda batch: batch,
                            prefetch_factor=20)
    
    data = time.strftime("%Y-%m-%d", time.localtime()) 
    result_dir = '/mnt/afs/zhoudinghao/data/subtitles/' + str(data) 
    result_dir = '/mnt/afs/zhoudinghao/data/subtitles/' '粤语/喜马拉雅/'+ str(data) 
    result_dir = '/mnt/afs/zhoudinghao/data/subtitles/'+'youtube/check/'+str(data) 
    os.makedirs(result_dir, exist_ok=True)
    file_name = os.path.join(result_dir,str(rank)+'.jsonl' + '.part')
    f = open(file_name, 'a+', encoding='utf-8')
    
    print('-----'*10, 'i am processing', rank , ' saving {}'.format(file_name))
    dist.barrier()
    for data in dataloader:
        f.write('{}\t{}\t{}\n'.format(data['name'], data['id'], data['has_subtitles']))
        f.flush()
    f.close()

