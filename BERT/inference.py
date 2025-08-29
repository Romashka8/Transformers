# ---------------------------------------------------------------------------------------------------------------------------------

import os
import sys
import gc

from pathlib import Path
from typing import Callable

import json
import pickle

import pandas as pd

import torch
import transformers as ppb
from torch.utils.data import Dataset, DataLoader

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
				load_function: Callable):
		self.data = load_function(data_path)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		assert idx < len(self.data), f'Index error! Index {idx} out of range!'
		row = self.data.iloc[idx]
		row_returned = {column: row[column] for column in self.data.columns}
		return row_returned

# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	# tests: 29.08.2025
	abs_path = Path(__file__).resolve().parent.parent
	configs = ConfigsBERT(
					MODEL_NAME = 'distilbert-base-uncased',
					DATA_PATH = os.path.join(abs_path, 'data/tonality/tonality.xlsx'),
					SAVE_PATH = os.path.join(abs_path, 'data/tests'),
					DATA_FORMAT = 'excel',
					BATCH_SIZE = 1000,
					ADD_SPECIAL_TOKENS = True,
					TRUNCATION = True,
					MAX_LENGTH = 512
		).get_configs()
	dataset = DatasetBERT(
					configs.DATA_PATH,
					configs.FORMATS_MAP[configs.DATA_FORMAT]
					)
	print(dataset[0])

# ---------------------------------------------------------------------------------------------------------------------------------
