import transformers 
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

"""

"""

class BaseTrainer():
    def __init__(self, model, **trainer_kwargs):
        self.trainer_kwargs = trainer_kwargs
        self.trainer = Trainer(**trainer_kwargs)

    def train(self, train_dataset, eval_dataset, **kwargs):
        raise NotImplementedError
    

    def get_trainer(self):
        return self.trainer

    def get_trainer_kwargs(self):
        return self.trainer_kwargs

    def save_model(self, output_dir):
        self.trainer.save_model(output_dir)


