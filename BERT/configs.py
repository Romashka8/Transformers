# ---------------------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import pandas as pd

from dataclasses import dataclass, field

from easydict import EasyDict

# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class ConfigsBERT:

	"""
	Usage example:

	configs_bert = ConfigsBERT(
					MODEL_NAME = 'distilbert-base-uncased',
					DATA_PATH = '/absolute/data/path/data.excel',
					SAVE_PATH = '/absolute/save/path/',
					DATA_FORMAT = 'excel',
					BATCH_SIZE = 1000,
					ADD_SPECIAL_TOKENS = True,
					TRUNCATION = True,
					MAX_LENGTH = 512
				   ).get_configs()

	"""

	# Setup by user
	MODEL_NAME: str
	DATA_PATH: str
	SAVE_PATH: str
	DATA_FORMAT: str
	BATCH_SIZE: int=1000
	ADD_SPECIAL_TOKENS: bool=True
	TRUNCATION: bool=True
	MAX_LENGTH: int=512
	RETURN_TOKEN_TYPE_IDS: bool=False,
	PADDING:str='max_length',
	RETURN_ATTENTION_MASK: bool=True,
	RETURN_TENSORS: str='pt'
	
	cfg: EasyDict = field(init=False, default_factory=EasyDict)

	def __post_init__(self):
		# Setup automaticly
		self.cfg.ABSOLUTE_PATH = Path(__file__).resolve().parent.parent
		assert os.path.exists(self.DATA_PATH), f'Incorrect DATA_PATH - "{self.DATA_PATH}"!'
		self.cfg.DATA_PATH = self.DATA_PATH
		assert os.path.exists(self.SAVE_PATH), f'Incorrect SAVE_PATH - "{self.SAVE_PATH}"!'
		self.cfg.SAVE_PATH = self.SAVE_PATH

		self.cfg.FORMATS_MAP = {
			'excel': pd.read_excel,
			'csv': pd.read_csv,
			'pickle': pd.read_pickle,
		}
		assert self.DATA_FORMAT in self.cfg.FORMATS_MAP.keys(), f'Incorrenct DATA_FORMAT type - "{self.DATA_FORMAT}"!'
		self.cfg.DATA_FORMAT = self.DATA_FORMAT

		self.cfg.MODELS_MAP = {
			'bert-base-uncased': 'bert-base-uncased',
			'distilbert-base-uncased': 'distilbert-base-uncased',
			'rubert-base-cased': 'rubert-base-cased'
		}
		assert self.MODEL_NAME in self.cfg.MODELS_MAP.keys(), f'Incorrect MODEL_NAME - "{self.MODEL_NAME}"!'
		self.cfg.MODEL_NAME = self.cfg.MODELS_MAP[self.MODEL_NAME]

		self.cfg.BATCH_SIZE = self.BATCH_SIZE
		self.cfg.ADD_SPECIAL_TOKENS = self.ADD_SPECIAL_TOKENS
		self.cfg.TRUNCATION = self.TRUNCATION
		self.cfg.MAX_LENGTH = self.MAX_LENGTH
		self.cfg.RETURN_TOKEN_TYPE_IDS = self.RETURN_TOKEN_TYPE_IDS
		self.cfg.PADDING = self.PADDING
		self.cfg.RETURN_ATTENTION_MASK = self.RETURN_ATTENTION_MASK
		self.cfg.RETURN_TENSORS = self.RETURN_TENSORS

	def get_configs(self):
		return self.cfg

# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	
	test_path = Path(__file__).resolve().parent.parent
	test = ConfigsBERT(
		'distilbert-base-uncased',
		os.path.join(test_path, 'data/tests'),
		os.path.join(test_path, 'data/tests'),
		'excel')
	print(test.get_configs())

# ---------------------------------------------------------------------------------------------------------------------------------