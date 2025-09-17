# ---------------------------------------------------------------------------------------------------------------------------------
 
import os
import gc
 
from typing import Any, Union
 
from tqdm import tqdm
 
from easydict import EasyDict
 
import numpy as np
import pandas as pd
 
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
 
from transformers import AutoModelForCausalLM, AutoTokenizer
 
import pickle
 
from giga.configs import GigaConfigs
 
# ---------------------------------------------------------------------------------------------------------------------------------
 
class GPTDatasetFromPandas(Dataset):
    def __init__(self, data: pd.DataFrame, prompt: str):
        self.data = data
        self.prompt = prompt
        self.columns = data.columns
 
    def __len__(self) -> int:
        return len(self.data)
 
    def __getitem__(self, idx: int) -> dict:
        assert idx < len(self.data)
        sample = self.data.iloc[idx]
        output = {f'{column}': sample[column] for column in self.data.columns}
        output['prompt'] = self.prompt
        return output
 
# ---------------------------------------------------------------------------------------------------------------------------------
 
class Inference:
    def __init__(self,
                 configs: Union[GPTConfigs, EasyDict],
                 model: str,
                 dataset: GPTDatasetFromPandas,
                 truncation_column: str=None,
                 fixation_columns: str=None,
                 overlapping_window_step: int=None,
                ):
        self.configs = configs
        self.model_name = self.configs['PATH_MODEL']
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            legacy=True
        )
        self.dataset = dataset
        # Squeeze comments.
        if truncation_column and fixation_columns and overlapping_window_step:
            prompt = self.dataset.prompt
            self.dataset = GigaDatasetFromPandas(self._squeeze_dataset(truncation_column,
                                                 fixation_columns,
                                                 overlapping_window_step), prompt)
 
        self.world_size = min(self.configs['WORLD_SIZE'], torch.cuda.device_count())
        self.distributed = self.world_size > 1
 
    def _squeeze_dataset(self,
                         truncation_column: str,
                         fixation_columns: str,
                         overlapping_window_step: int,
                         overlapping_window_lookback: int=1
                        ):
        squeezed_dataset = pd.DataFrame()
        fixation_values = self.dataset.data.drop_duplicates(subset=fixation_columns)[fixation_columns].values
        for values in fixation_values:
            query_dict = {fixation_columns[col]: values[col] for col in range(len(fixation_columns))}
            query = ' and '.join([f'{column} == {repr(value)}' for column, value in query_dict.items()])
            filtred_data = self.dataset.data.query(query).reset_index()
            l_bound, r_bound = 0, overlapping_window_step
            while r_bound <= len(filtred_data):
                truncated_data = '\n'.join(filtred_data.iloc[l_bound:r_bound, :][truncation_column].astype(str).values)
                insert_values = query_dict
                query_dict[truncation_column] = truncated_data
                squeezed_dataset = pd.concat([squeezed_dataset,
                                              pd.DataFrame(query_dict, index=[0])
                                             ], axis=0)
                l_bound += int(overlapping_window_step - overlapping_window_lookback)
                r_bound = int(l_bound + overlapping_window_step)
        return squeezed_dataset.reset_index().drop(['index'], axis=1)
 
    def _prepare_batch(self, batch: dict) -> dict:
        res = {f'{column}': [] for column in self.dataset.columns}
        res[f'{self.configs['CONTENT_COLUMN']}_truncated'] = []
        res['tensors'] = []
        for row in range(self.configs['BATCH_SIZE']):
 
            prompt = batch[row]['prompt']
            content = batch[row][self.configs['CONTENT_COLUMN']]
 
            content_encoded = self.tokenizer(content,
                                             truncation=True,
                                             max_length=self.configs['MAX_TOKENS_COUNT'],
                                             add_special_tokens=False
                                            )['input_ids']
            content_truncated = self.tokenizer.decode(content_encoded)
            content_prepared = f'{prompt}\n{content_truncated}\nA'
 
            for key in list(res.keys()):
                if key not in (f'{self.configs['CONTENT_COLUMN']}_truncated', 'tensors'):
                    res[key].append(batch[row][key])
 
            res[f'{self.configs['CONTENT_COLUMN']}_truncated'].append(content_truncated)
            res['tensors'].append(content_prepared)
 
        res['tensors'] = self.tokenizer(res['tensors'],
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt'
                                        )['input_ids']
 
        return res
 
    def _prepare_loader(self) -> DataLoader:
        dataloader_cfg = {
            'batch_size': self.configs['BATCH_SIZE'],
            'drop_last': True,
            'pin_memory': True,
            'shuffle': False,
            'collate_fn': self._prepare_batch
        }
 
        sampler = DistributedSampler(self.dataset) if self.distributed else None
 
        return DataLoader(
            self.dataset,
            sampler=sampler,
            **dataloader_cfg
        )
 
    def _ddp_setup(self, rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12345'
        init_process_group(backend='nccl', rank=rank, world_size=world_size)
 
    def _any2device(self, value: Any, device: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.Tensor(value).to(device)
        elif isinstance(value, list):
            return [torch.Tensor(val).to(device) for val in value]
        elif torch.is_tensor(value) or isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        return value
 
    def _get_local_responses(self, generate_cfg: dict, loader: DataLoader, device: torch.device, save_path: str):
        self.model.eval()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(loader)):
 
                print(f'inference on index: {index}\n')
                try:
                    output = self.model.generate(
                        self._any2device(batch['tensors'], device),
                        **generate_cfg
                    )
 
                    lengths = [len(flud) for flud in batch[f'{self.configs['CONTENT_COLUMN']} truncated']]
                    responses = [self.tokenizer.decode(output[row][:], skip_special_tokens=True) for row in range(self.configs['BATCH_SIZE'])]
 
                    with open(os.path.join(save_path, f'batch_{index + 1}.pkl'), 'wb') as file:
                        pickle.dump({
                            f'batch_{index + 1}': batch,
                            'responses': responses
                        }, file)
                except torch.OutOfMemoryError:
 
                    # Release memory
                    del batch
                    gc.collect()
                    torch.cuda.empty_cache()
 
                    continue
 
                # Release memory
                del output, lengths, responses, batch
                gc.collect()
                torch.cuda.empty_cache()
 
                os.system('clear')
 
 
    def _single_gpu(self) -> None:
        loader = self._prepare_loader()
        self.model.to(torch.device('cuda:0'))
        generate_cfg = {
            'max_length': 32768,
            'repetition_penalty': 1.07,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        self._get_local_responses(generate_cfg, loader, torch.device('cuda:0'), self.configs['SAVE_PATH'])
        
    def _distributed(self, rank: int, word_size: int, save_path: str) -> None:
        self._ddp_setup(rank, word_size)
        loader = self._prepare_loader()
        self.model.to(torch.device(f'cuda:{rank}'))
        generate_cfg = {
            'max_length': 32768,
            'repetition_penalty': 1.07,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        # create save folder for each GRU with number 'rank'
        gpu_save_path = os.path.join(save_path, f'gpu_id_{rank}_local_responses')
        if not os.path.exists(gpu_save_path):
            os.mkdir(gpu_save_path)
        self._get_local_responses(generate_cfg, loader, torch.device(f'cuda:{rank}'), gpu_save_path)
        destroy_process_group()
   
    def inference(self):
        if self.distributed:
            world_size = min(self.configs['WORLD_SIZE'], torch.cuda.device_count()) if self.configs['WORLD_SIZE'] is not None else torch.cuda.device_count()
            print(f'World size: ', world_size)
            # If last inference - failed, create new trial folder(can be added into single gpu mode to).
            trials_count = len([name for name in os.listdir(self.configs['SAVE_PATH']) if name.find('trial') > -1])
            next_trial_path = os.path.join(self.configs['SAVE_PATH'], f'trial_{trials_count + 1}')
            if not os.path.exists(next_trial_path):
                os.mkdir(next_trial_path)
            mp.spawn(self._distributed, args=(world_size, next_trial_path), nprocs=world_size)
        else:
            self._single_gpu()
   
# ---------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
# --------------------TESTS(08.09.2025)--------------------------------------------------------------------------------------------
 
    import time
    from pathlib import Path
   
    start = time.time()
    abs_path = Path(__file__).resolve().parent.parent
    configs = ConfigsGPT(
        DATA_PATH = os.path.join(abs_path, 'data/testing/hranidengi_headers.xlsx'),
        SAVE_PATH = os.path.join(abs_path, 'data/testing'),
        PATH_MODEL = os.path.join(abs_path, 'qwen3:0.6b'),
        CONTENT_COLUMN = 'flud name',
        PROMPT = 'base',
        WORLD_SIZE = 4,
        BATCH_SIZE = 1,
        MAX_TOKENS_COUNT = 1500
    ).get_configs()
   
    # headers loading
    data = pd.read_excel(configs.DATA_PATH).drop(
        ['start date', 'latest update'],
        axis=1
    )
   
    with open(configs.PROMPTS['base'], 'r') as f:
        prompt = ''.join(f.readlines())
   
    dataset = GPTDatasetFromPandas(data, prompt)
   
    # inference for comments
    test = Inference(
                     configs,
                     configs.PATH_MODEL,
                     dataset
                     # truncation_column='message',
                     # fixation_columns=['topic_name', 'topic_link', 'flud_name', 'flud_link'],
                     # overlapping_window_step=3
                    )
 
    test.inference()
 
    print('Time: ', time.time() - start)
 
# ---------------------------------------------------------------------------------------------------------------------------------