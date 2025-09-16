# ---------------------------------------------------------------------------------------------------------------------------------
 
import os
from pathlib import Path
 
from typing import Dict
from dataclasses import dataclass, field
 
from easydict import EasyDict
 
# ---------------------------------------------------------------------------------------------------------------------------------
 
@dataclass
class ConfigsGPT:
 
    """
   
    Usage example:
   
        abs_path = Path(__file__).resolve().parent.parent
 
        giga_configs = ConfigsGPT(
            DATA_PATH = os.path.join(abs_path, 'data/testing/hranidengi_headers.xlsx'),
            SAVE_PATH = os.path.join(abs_path, 'data/testing'),
            PATH_MODEL = os.path.join(abs_path, 'models/GigaChat-7b.3.1-25.3-32k-sft-base'),
            CONTENT_COLUMN = 'message',
            PROMPT = 'base',
            WORLD_SIZE = 4,
            BATCH_SIZE = 1,
            MAX_TOKENS_COUNT = 1500
        ).get_configs()
        print(giga_configs)
       
    """
   
    DATA_PATH: str
    SAVE_PATH: str
    PATH_MODEL: str
    CONTENT_COLUMN: str
    PROMPT: str
    PROMPTS: Dict[str, str] = field(default_factory=dict)
 
    # Число видеокарт для инференса.
    # В скрипте потом выберется min(cfg.WORLD_SIZE, torch.cuda.device_count())
    WORLD_SIZE: int = 4
    BATCH_SIZE: int = 1
    MAX_TOKENS_COUNT: int = 1500
 
    cfg: EasyDict = field(init=False, default_factory=EasyDict)
 
    def __post_init__(self):
        self.cfg.ABSOLUTE_PATH = Path(__file__).resolve().parent.parent
        self.cfg.DATA_PATH = self.DATA_PATH
        self.cfg.SAVE_PATH = self.SAVE_PATH
        self.cfg.PATH_MODEL = self.PATH_MODEL
        self.cfg.CONTENT_COLUMN = self.CONTENT_COLUMN
        self.cfg.PROMPTS = {
            'test': os.path.join(self.cfg.ABSOLUTE_PATH, 'prompts/test.txt'),
        }
        self.cfg.PROMPT = self.cfg.PROMPTS[self.PROMPT]
        self.cfg.WORLD_SIZE = self.WORLD_SIZE
        self.cfg.BATCH_SIZE = self.BATCH_SIZE
        self.cfg.MAX_TOKENS_COUNT = self.MAX_TOKENS_COUNT
       
    def get_configs(self):
        return self.cfg
 
# ---------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
# --------------------TESTS(08.09.2025)--------------------------------------------------------------------------------------------
 
    abs_path = Path(__file__).resolve().parent.parent
 
    giga_configs = ConfigsGPT(
        DATA_PATH = os.path.join(abs_path, 'data/testing/hranidengi_headers.xlsx'),
        SAVE_PATH = os.path.join(abs_path, 'data/testing'),
        PATH_MODEL = os.path.join(abs_path, 'qwen3:0.6b'),
        CONTENT_COLUMN = 'message',
        PROMPT = 'test',
        WORLD_SIZE = 4,
        BATCH_SIZE = 1,
        MAX_TOKENS_COUNT = 1500
    ).get_configs()
    print(giga_configs)
   
# ---------------------------------------------------------------------------------------------------------------------------------