# ---------------------------------------------------------------------------------------------------------------------------------

import os
import sys
import gc

from pathlib import Path
from typing import Callable, Union

import json
import pickle

import numpy as np
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from easydict import EasyDict

from configs import ConfigsBERT

# ---------------------------------------------------------------------------------------------------------------------------------

class DatasetBERT(Dataset):
	"""
	Usage example:
		
		import pandas as pd

		dataset = DatasetBERT(
				'/absolute/data/path/data.xlsx',
				pd.read_excel
		)

	"""
	def __init__(self,
				data_path: str,
				load_function: Callable,
				):
		self.data = load_function(data_path)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		assert idx < len(self.data), f'Index error! Index {idx} out of range!'
		row = self.data.iloc[idx]
		row_returned = {column: row[column] for column in self.data.columns}
		return row_returned

# ---------------------------------------------------------------------------------------------------------------------------------

class InferenceBERT:
	"""
	Usage example:

		from configs import ConfigsBERT

		abs_path = Path(__file__).resolve().parent.parent
		configs = ConfigsBERT(
						MODEL_NAME = 'distilbert-base-uncased',
						DATA_PATH = 'absolute/data/path/tonality.xlsx',
						SAVE_PATH = 'absolute/save/path/',
						DATA_FORMAT = 'excel',
						BATCH_SIZE = 3,
						ADD_SPECIAL_TOKENS = True,
						TRUNCATION = True,
						MAX_LENGTH = 512
			).get_configs()

		dataset = DatasetBERT(
				'/absolute/data/path/data.xlsx',
				pd.read_excel
		)

		bert = InferenceBERT(
				bert_configs=configs,
				data=dataset,
				emb_column='sentence'
			)
		bert.inference(0, 1)
	"""
	def __init__(self,
				bert_configs: Union[EasyDict, ConfigsBERT],
				data: DatasetBERT,
				emb_column: str
				):
		self.configs = bert_configs.get_configs() if isinstance(bert_configs, ConfigsBERT) else bert_configs
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.tokenizer_params = {
			'add_special_tokens': self.configs.ADD_SPECIAL_TOKENS,
			'truncation': self.configs.TRUNCATION,
			'max_length': self.configs.MAX_LENGTH
		}
		self.model = AutoModel.from_pretrained(self.configs.MODEL_NAME).to(self.device)
		self.tokenizer = AutoTokenizer.from_pretrained(self.configs.MODEL_NAME)
		self.last_processed_batch = 0
		self.loader_params = {
			'batch_size': self.configs.BATCH_SIZE
		}
		self.loader = DataLoader(data, **self.loader_params)
		self.emb_column = emb_column

	def inference(self,
				 start: int,
				 end: int):
		loop = tqdm(self.loader, leave=False)
		encoder = lambda x: self.tokenizer.encode(x, **self.tokenizer_params)

		for batch in loop:
			if start <= self.last_processed_batch < end:

				try:
					tokenized = list(map(encoder, batch[self.emb_column]))

					max_len = 0
					for i in tokenized: max_len = max(max_len, len(i))

					padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
					attention_mask = np.where(padded != 0, 1, 0)

					input_ids = torch.tensor(padded).to(self.device)
					attention_mask = torch.tensor(attention_mask).to(self.device)

					with torch.no_grad():
						last_hidden_states = self.model(
							input_ids, attention_mask=attention_mask
						)

					features = last_hidden_states[0][:, 0, :].cpu().numpy()

					SAVE_BATCH = os.path.join(self.configs.SAVE_PATH, f'{self.configs.MODEL_NAME}_embedded_batch_{self.last_processed_batch}.json')
					with open(SAVE_BATCH, 'w') as file:
						batch_json = {}
						batch_json[self.emb_column] = batch[self.emb_column]
						batch_json['content_embedded'] = features.tolist()
						json.dump(batch_json, file)

					# Release memory
					del input_ids, attention_mask, last_hidden_states, features, batch, padded, batch_json
					gc.collect()
					if self.device == 'cuda': torch.cuda.empty_cache()

				except Exception as e:
					print('Exception while embedding: ', e)

				self.last_processed_batch += 1

		print(f'Inference has stopped on {self.last_processed_batch} batch...')

# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	# tests: 29.08.2025
	abs_path = Path(__file__).resolve().parent.parent
	configs = ConfigsBERT(
					MODEL_NAME = 'distilbert-base-uncased',
					DATA_PATH = os.path.join(abs_path, 'data/tonality/tonality.xlsx'),
					SAVE_PATH = os.path.join(abs_path, 'data/tests'),
					DATA_FORMAT = 'excel',
					BATCH_SIZE = 3,
					ADD_SPECIAL_TOKENS = True,
					TRUNCATION = True,
					MAX_LENGTH = 512
		).get_configs()
	dataset = DatasetBERT(
					configs.DATA_PATH,
					configs.FORMATS_MAP[configs.DATA_FORMAT]
					)
	# tests: 31.08.2025
	bert = InferenceBERT(
			configs,
			dataset,
			'sentence'
		)
	bert.inference(0, 1)

# ---------------------------------------------------------------------------------------------------------------------------------
