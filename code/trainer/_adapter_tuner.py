import datasets
import transformers
from transformers import TrainingArguments, Trainer
from evaluator import Evaluator
import pandas as pd
from dataloader import InitiativeExcelLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorWithPadding, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    TrainingArguments,
    HfArgumentParser,
    AdapterTrainer,
    Seq2SeqAdapterTrainer,
    T5Tokenizer,
    T5ForConditionalGeneration
)

from utils.HuggingFacePathHandler import get_huggingface_path
import torch
import numpy as np

import os  
from prompts import PromptTemplate
from formatter import DatasetFormatter
from utils.constants import Models
from prompts import InstructionLoader
from dataclasses import dataclass, field
from typing import Optional

from transformers import  get_linear_schedule_with_warmup, set_seed

from torch.utils.data import DataLoader
from datetime import datetime


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_source_length: Optional[int] = field(
        default=5000, metadata={
            "help": "Maximum source length for tokenizer. Default value is 5000."
        }
    )
    


@dataclass
class DataFormatArguments:
    """
    Arguments pertaining to how we need to format the input/output data for training our model.
    """
    template_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the model to select a template file for formatting data e.g T5, TkInstruct."}
    )
    template_id: Optional[int] = field(
        default=None, 
        metadata={
            "help": "The id of the prompt to use to format the data. e.g. 1, 2 etc."
        }
    )
    instruction_id: Optional[int] = field(
        default=None, 
        metadata={
            "help": "The id of the instruction to use to format the data. e.g. 1, 2 etc."
        }
    )
    reasoning_enabled: bool = field(
        default=False, 
        metadata={
            "help": "Use the reasoning in the data format"
        }
    )
    filter_dataset: bool = field(
        default=False, 
        metadata={
            "help": "Filter the eval dataset based if reasoning format is compared"
        }
    )
    trial: int = field(
        default=0, 
        metadata={
            "help": "Experiment number used to locate the result files."
        }
    )
    shuffle_seed: int = field(
        default=1, 
        metadata={
            "help": "Seed to shuffle the dataset before training."
        }
    )
    

class AdapterTuner():
    
    def __init__(self, **kwargs):
        parser = HfArgumentParser((ModelArguments, DataFormatArguments, TrainingArguments))
        self.model_args, self.data_format_args, self.training_args = parser.parse_args_into_dataclasses()
        print(self.model_args)
        print(self.data_format_args)
        print(self.training_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.model_args.model_name_or_path.split("/")[-1]
        self.result_dir = f'modular_approach/results/adapter-tuning/{self.model_name}/'
        self.logs_dir = f'modular_approach/logs/adapter-tuning/{self.model_name}/'
        self._init_classes()
        self._load_dataset()
        self.output_df = pd.DataFrame(columns=['precision','recall','f1','preds','labels'])
        self.get_model()
        

    def _init_classes(self):
        self.dataloader = InitiativeExcelLoader()
        # if self.data_format_args.template_name:
        #     self.template_obj = PromptTemplate(self.data_format_args.template_name, self.data_format_args.template_id)
        #     #if self.model_name_or_path == Models.T5_FLAN_SMALL:
        #     self.formatter = DatasetFormatter(self.template_obj, self.dataloader)
        #     self.instructions = InstructionLoader().get_instructions()
        self.evaluator = Evaluator()
                
        
    def _load_dataset(self):
        self.formatted_example = None
        # if self.data_format_args.instruction_id:
        #     if self.data_format_args.reasoning_enabled:
        #         data = self.formatter.fine_tune_reasoning_format(instruction=self.instructions[self.data_format_args.instruction_id])
        #     else:
        #         data = self.formatter.fine_tune_format(instruction=self.instructions[self.data_format_args.instruction_id])
        #     print(data['test'][0]['formatted_output'])
        #     print(data['test'][0]['label'])
        #     self.formatted_example = data['test'][0]['formatted_output']
        #     #self.formatted
        #     data_df = pd.DataFrame(data=data['test'])
        # else:
        data_df = pd.DataFrame(self.dataloader.dataset['test'])
        
        #if self.data_format_args.reasoning_enabled or self.data_format_args.filter_dataset: data_df = data_df.loc[data_df['reasoning'] != 'None']
        
        self.dataset = datasets.Dataset.from_pandas(data_df).shuffle(seed=self.data_format_args.shuffle_seed)
        #self.dataset.to_csv(self.result_dir + f'shuffled_data_{self.data_format_args.trial}.csv')
        print("Saved shuffled data")
        

    def preprocess_function(self, examples):
        if self.data_format_args.instruction_id:
            model_inputs = self.tokenizer(examples['formatted_output'], padding="longest", max_length=self.model_args.max_source_length, truncation=True)   
        else:
            model_inputs = self.tokenizer(examples['review'], padding="longest", max_length=self.model_args.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["true_nc"], padding="longest", max_length=self.model_args.max_source_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    def get_model(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(get_huggingface_path(self.model_args.model_name_or_path), local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_args.model_name_or_path), local_files_only=True)#.to(self.device)
        # self.tokenizer = T5Tokenizer.from_pretrained(self.model_args.model_name_or_path)
        # self.model = T5ForConditionalGeneration.from_pretrained(self.model_args.model_name_or_path)
        # add new adapter
        self.model.add_adapter("noncore")
        # activate adapter for training
        self.model.train_adapter("noncore")
        #self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        #print(self.dataset)
        self.tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset.column_names)
        print(self.tokenized_datasets)


    def model_init(self):   
        return AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_args.model_name_or_path))#.to(self.device)

    def collate_fn(self,examples):
                return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")


    def split_dataset(self, train_split=90, kfold=True):
        train_data = {}
        test_data = {}
        num_rows = self.tokenized_datasets.num_rows
        if not kfold:
            for column in self.tokenized_datasets.column_names:
                split_index = num_rows * train_split // 100
                train_data[column] = self.tokenized_datasets[column][:split_index]
                test_data[column] = self.tokenized_datasets[column][split_index:]
            train_ds = datasets.Dataset.from_dict(train_data)
            test_ds = datasets.Dataset.from_dict(test_data)
            #print(train_ds)
            yield train_ds, test_ds

        else :
            step = num_rows * (100-train_split) // 100
            for k in range(0, num_rows, step):
                for column in self.tokenized_datasets.column_names:
                    test_data[column] = self.tokenized_datasets[column][k: step+k]
                    train_data[column] = self.tokenized_datasets[column][:k] + self.tokenized_datasets[column][step+k:]
                train_ds = datasets.Dataset.from_dict(train_data)
                test_ds = datasets.Dataset.from_dict(test_data)
            
           
                # Instantiate dataloaders.
                # train_dataloader = DataLoader(
                #     train_ds, collate_fn=self.collate_fn
                # )
                # eval_dataloader = DataLoader(
                #     test_ds, collate_fn=self.collate_fn 
                # )

                # yield train_dataloader, eval_dataloader
                yield train_ds, test_ds


    def compute_metrics(self, eval_pred):
        #print(eval_pred.shape)
        print(eval_pred)
        logits, labels = eval_pred
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        print('decoded labels', decoded_labels)
        
        decoded_preds = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        print('decoded preds', decoded_preds)
        if self.data_format_args.reasoning_enabled: result = self.evaluator.calculate_metrics_by_format(preds=decoded_preds, labels=decoded_labels)
        else: result = self.evaluator.calculate_metrics(preds=decoded_preds, labels=decoded_labels)
        self.output_df = self.output_df.append({'precision': result['precision'],'recall': result['recall'],
                     'f1': result['f1'], 'preds': decoded_preds,'labels': decoded_labels}, ignore_index = True)
        #print('result : ', result)
        return result
    
    def output_results(self):
        self.output_df.index += 1
        self.output_df = self.output_df[self.output_df.index % self.training_args.num_train_epochs == 0]
        
        df2 = self.output_df[['precision','recall','f1']].mean()
        new_row = pd.DataFrame({'precision': df2['precision'],'recall': df2['recall'], 'f1': df2['f1'], 
                    'preds':f'model: {self.model_name}\n'\
                        f'learning_rate: {self.training_args.learning_rate}\n'\
                        f'num_train_epochs: {self.training_args.num_train_epochs}\n'\
                        f'weight_decay: {self.training_args.weight_decay}\n' \
                        f'per_device_train_batch_size: {self.training_args.per_device_train_batch_size}\n'\
                        f'gradient_accumulation_steps: {self.training_args.gradient_accumulation_steps}',
                    'labels': self.formatted_example}, index =[0])
        
        self.output_df = pd.concat([new_row, self.output_df]).reset_index(drop = True)
        result_file = self.result_dir + f'trial_{self.data_format_args.trial}.csv'
        self.output_df.to_csv(result_file)
        
        print("Check logs: ", self.logs_dir+f'trial_{self.data_format_args.trial}')
        print("Check results: ", result_file)


    def train(self):
        print("Training Model : ", self.model_name)
        start_time = datetime.now()
        self.training_args.log_level = 'debug'
        self.training_args.log_level_replica='passive',
        training_args = Seq2SeqTrainingArguments(
            #f"{self.model_name}-finetuned-squad",
            output_dir="test_trainer",
            overwrite_output_dir = self.training_args.overwrite_output_dir,
            evaluation_strategy = self.training_args.evaluation_strategy,
            learning_rate = self.training_args.learning_rate,
            num_train_epochs = self.training_args.num_train_epochs,
            weight_decay = self.training_args.weight_decay,
            per_device_train_batch_size = self.training_args.per_device_train_batch_size,
            predict_with_generate = True,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            #logging_steps=10,
            logging_dir = self.logs_dir+f'trial_{self.data_format_args.trial}',
            log_level = self.training_args.log_level,
            logging_strategy = self.training_args.logging_strategy
            #push_to_hub=True,
        )
        #print(training_args)
        metrics = []
        
        for train_ds, val_ds in self.split_dataset():
            
            print(f"Training set samples: {len(train_ds)} \t Testset samples: {len(val_ds)}")
            trainer = Seq2SeqAdapterTrainer (
                #model_init = self.model_init,
                model=self.model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer = self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
        
            trainer.train()

        self.output_results()
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        print('Training Finished!!!')
