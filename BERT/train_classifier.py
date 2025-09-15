# ---------------------------------------------------------------------------------------------------------------------------------

import os
import sys
import gc

from pathlib import Path
from typing import Callable, Union, Optional, Any

import json
import pickle

import numpy as np
import pandas as pd

import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

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
				data_path: Union[str, pd.DataFrame],
				load_function: Optional[Callable]=None,
				tokenizer: Optional[Any]=None,
				tokenizer_params: Optional[dict]=None,
				emb_column: Optional[str]=None
				):
		self.data = load_function(data_path) if isinstance(data_path, str) else data_path 
		self.tokenizer = tokenizer
		self.tokenizer_params = tokenizer_params
		self.emb_column = emb_column

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		assert idx < len(self.data), f'Index error! Index {idx} out of range!'
		row = self.data.iloc[idx]
		row_returned = {column: row[column] for column in self.data.columns}
		if self.tokenizer and self.tokenizer_params:
			encoder = lambda x: self.tokenizer.encode_plus(x, **self.tokenizer_params)
			row_encoded = encoder(row[self.emb_column])
			row_returned['input_ids'] = row_encoded['input_ids'].flatten()
			row_returned['attention_mask'] = row_encoded['attention_mask'].flatten()
		return row_returned

# ---------------------------------------------------------------------------------------------------------------------------------

# RAW AND UNFINISHED CODE!
class TrainClassifierBERT:
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

		bert = TrainClassifierBERT(
					configs,
					emb_column='sentence',
					tg_column='sentiment_values', # NOT BINARIZED, JUST FOR TESTS!
					n_classes=2,
					epochs=1,
					model_save_path='model/save/path/model.pt'
				)

		tonality_df = pd.read_excel(configs.DATA_PATH)
		train_set = tonality_df[tonality_df['splitset_label'] == 1]
		test_set = tonality_df[tonality_df['splitset_label'] == 2]
		bert.setup_training_toolkit(train_set, test_set)
		bert._fit()	
	"""
	def __init__(self,
				bert_configs: Union[EasyDict, ConfigsBERT],
				emb_column: str,
				tg_column: str,
				n_classes: int,
				epochs: int,
				model_save_path: str,
				load_fitted: Optional[str]=False
			):
		self.configs = bert_configs.get_configs() if isinstance(bert_configs, ConfigsBERT) else bert_configs
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.tokenizer_params = {
			'add_special_tokens': self.configs.ADD_SPECIAL_TOKENS,
			'truncation': self.configs.TRUNCATION,
			'max_length': self.configs.MAX_LENGTH,
        	'return_token_type_ids': self.configs.RETURN_TOKEN_TYPE_IDS,
        	'padding': self.configs.PADDING,
        	'return_attention_mask': self.configs.RETURN_ATTENTION_MASK,
        	'return_tensors': self.configs.RETURN_TENSORS
		}

		if not load_fitted:
			self.model = BertForSequenceClassification.from_pretrained(self.configs.MODEL_NAME)
			self.out_features = self.model.config.hidden_size
			self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
		else:
			self.model = torch.load(load_fitted).to(self.device)
			self.out_features = self.model.config.hidden_size

		self.tokenizer = AutoTokenizer.from_pretrained(self.configs.MODEL_NAME)
		self.loader_params = {
			'batch_size': self.configs.BATCH_SIZE
		}
		self.emb_column = emb_column
		self.tg_column = tg_column
		self.epochs = epochs
		self.model.to(self.device)
		self.model_save_path = model_save_path

	def setup_training_toolkit(self,
							   train_data: pd.DataFrame,
							   test_data: pd.DataFrame,
		):
		# create datasets
		self.train_set = DatasetBERT(train_data,
									 tokenizer=self.tokenizer,
									 tokenizer_params=self.tokenizer_params,
									 emb_column=self.emb_column
									)
		self.valid_set = DatasetBERT(test_data,
									 tokenizer=self.tokenizer,
									 tokenizer_params=self.tokenizer_params,
									 emb_column=self.emb_column)

		# create data loaders
		self.train_loader = DataLoader(self.train_set,
									   batch_size=self.configs.BATCH_SIZE,
									   shuffle=True)
		self.valid_loader = DataLoader(self.valid_set,
									   batch_size=self.configs.BATCH_SIZE,
									   shuffle=True)
		
		# helpers initialization
		self.optimizer = AdamW(self.model.parameters(),
							   lr=2e-5,
							)
		self.scheduler = get_linear_schedule_with_warmup(
				self.optimizer,
				num_warmup_steps=0,
				num_training_steps=len(self.train_loader) * self.epochs
			)
		self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

	def _fit(self):
		self.model = self.model.train()
		losses = []
		correct_predictions = 0

		for data in self.train_loader:
			input_ids = data['input_ids'].to(self.device)
			attention_mask = data['attention_mask'].to(self.device)
			targets = torch.tensor(data[self.tg_column], dtype=torch.long).to(self.device)

			outputs = self.model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

			preds = torch.argmax(outputs.logits, dim=1)
			loss = self.loss_fn(outputs.logits, targets)

			correct_predictions += torch.sum(preds == targets)

			losses.append(loss.item())
            
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optimizer.step()
			self.scheduler.step()
			self.optimizer.zero_grad()

			# Release memory:
			del input_ids, attention_mask, targets, outputs, data, preds, loss
			gc.collect()
			if self.device == 'cuda': torch.cuda.empty_cache()

		train_acc = correct_predictions.double() / len(self.train_set)
		train_loss = np.mean(losses)
		return train_acc, train_loss

	def _eval(self):
		self.model = self.model.eval()
		losses = []
		correct_predictions = 0

		with torch.no_grad():
			for data in self.valid_loader:
				input_ids = data['input_ids'].to(self.device)
				attention_mask = data['attention_mask'].to(self.device)
				targets = torch.tensor(data[self.tg_column], dtype=torch.long).to(self.device)

				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask
					)

				preds = torch.argmax(outputs.logits, dim=1)
				loss = self.loss_fn(outputs.logits, targets)
				correct_predictions += torch.sum(preds == targets)
				losses.append(loss.item())

				# Release memory:
				del input_ids, attention_mask, targets, outputs, data, preds, loss
				gc.collect()
				if self.device == 'cuda': torch.cuda.empty_cache()
        
		val_acc = correct_predictions.double() / len(self.valid_set)
		val_loss = np.mean(losses)
		return val_acc, val_loss
    
	def train(self):
		best_accuracy = 0
		for epoch in range(self.epochs):
			print(f'Epoch {epoch + 1}/{self.epochs}')
			train_acc, train_loss = self._fit()
			print(f'Train loss {train_loss} accuracy {train_acc}')

			val_acc, val_loss = self._eval()
			print(f'Val loss {val_loss} accuracy {val_acc}')
			print('-' * 10)

			if val_acc > best_accuracy:
				torch.save(self.model, self.model_save_path)
				best_accuracy = val_acc

		# Get best model
		self.model = torch.load(self.model_save_path)
    
	def predict(self, text):
		encoding = self.tokenizer.encode_plus(
			text,
			**self.tokenizer_params
		)
        
		out = {
			  'text': text,
			  'input_ids': encoding['input_ids'].flatten(),
			  'attention_mask': encoding['attention_mask'].flatten()
		}

		input_ids = out['input_ids'].to(self.device)
		attention_mask = out['attention_mask'].to(self.device)
        
		outputs = self.model(
			input_ids=input_ids.unsqueeze(0),
			attention_mask=attention_mask.unsqueeze(0)
		)
        
		prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

		return prediction


# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	# tests: 10.09.2025
	abs_path = Path(__file__).resolve().parent.parent
	configs = ConfigsBERT(
					MODEL_NAME = 'distilbert-base-uncased',
					DATA_PATH = os.path.join(abs_path, 'data/tonality/tonality_binarized.xlsx'),
					SAVE_PATH = os.path.join(abs_path, 'data/tests'),
					DATA_FORMAT = 'excel',
					BATCH_SIZE = 1,
					ADD_SPECIAL_TOKENS = True,
					TRUNCATION = True,
					MAX_LENGTH = 512,
					RETURN_TOKEN_TYPE_IDS = False,
					PADDING = 'max_length',
					RETURN_ATTENTION_MASK = True,
					RETURN_TENSORS = 'pt'
			).get_configs()
	# dataset = DatasetBERT(
	# 				configs.DATA_PATH,
	# 				configs.FORMATS_MAP[configs.DATA_FORMAT]
	# 				)
	# tests: 10.09.2025
	bert = TrainClassifierBERT(
			configs,
			emb_column='sentence',
			tg_column='sentiment_values', # NOT BINARIZED, JUST FOR TESTS!
			n_classes=2,
			epochs=1,
			model_save_path='model/save/path/model.pt'
		)
	# tests: 11.09.2025
	tonality_df = pd.read_excel(configs.DATA_PATH)
	train_set = tonality_df[tonality_df['splitset_label'] == 1]
	test_set = tonality_df[tonality_df['splitset_label'] == 2]
	bert.setup_training_toolkit(train_set, test_set)
	bert._fit()

# ---------------------------------------------------------------------------------------------------------------------------------
