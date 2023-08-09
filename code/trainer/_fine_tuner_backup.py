import datasets
import transformers
from transformers import TrainingArguments, Trainer
from evaluator import Evaluator
import pandas as pd
from dataloader import InitiativeExcelLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, Seq2SeqTrainer, Seq2SeqTrainingArguments 
from utils.HuggingFacePathHandler import get_huggingface_path
import torch
import numpy as np
import torch.nn as nn
from ray import tune 
import os  
from prompts import PromptTemplate
from formatter import DatasetFormatter
from utils.constants import Models
from prompts import InstructionLoader
from dataclasses import dataclass





class FineTuner():
    
    def __init__(self, model_path, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.args = kwargs
        self.max_source_length = 5000
        self.template = 't5'
        self.template_id = 3 # 3: for reasoning 1: for prompt
        self.instruction_id = None # None: standard -1: last instruction
        self.reasoning_enabled = False 
        self.filter_dataset = False
        self.result_dir = f'modular_approach/results/fine-tuning/{self.model_name}/'
        self.logs_dir = f'modular_approach/logs/fine-tuning/{self.model_name}/'
        self.trial = '0'
        print(self.trial)
        self.learning_rate = 3e-5
        self.num_train_epochs = 5
        self.weight_decay = 0.001
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self._init_classes()
        self._load_dataset()
        self.output_df = pd.DataFrame(columns=['precision','recall','f1','preds','labels'])
        self.get_model()
        
    

    def _init_classes(self):
        self.dataloader = InitiativeExcelLoader()
        self.template_obj = PromptTemplate(self.template, self.template_id)
        #if self.model_path == Models.T5_FLAN_SMALL:
        self.formatter = DatasetFormatter(self.template_obj, self.dataloader)
        self.evaluator = Evaluator()
        self.instructions = InstructionLoader().get_instructions()
        
        
    def _load_dataset(self):
        self.formatted_example = None
        if self.instruction_id:
            if self.reasoning_enabled:
                data = self.formatter.fine_tune_reasoning_format(instruction=self.instructions[self.instruction_id])
            else:
                data = self.formatter.fine_tune_format(instruction=self.instructions[self.instruction_id])
            print(data['test'][0]['formatted_output'])
            print(data['test'][0]['label'])
            self.formatted_example = data['test'][0]['formatted_output']
            data_df = pd.DataFrame(data=data['test'])
        else:
            data_df = pd.DataFrame(self.dataloader.dataset['test'])
        
        if self.reasoning_enabled or self.filter_dataset: data_df = data_df.loc[data_df['reasoning'] != 'None']
        data_df.to_csv(self.result_dir + f'shuffled_data_{self.trial}.csv')
        print("Saved shuffled data")
        self.dataset = datasets.Dataset.from_pandas(data_df).shuffle(seed=42)
        

    def preprocess_function(self, examples):
        if self.instruction_id:
            model_inputs = self.tokenizer(examples['formatted_output'], padding="longest", max_length=self.max_source_length, truncation=True)   
        else:
            model_inputs = self.tokenizer(examples['review'], padding="longest", max_length=self.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["true_nc"], padding="longest", max_length=self.max_source_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    def get_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(get_huggingface_path(self.model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_path)).to(self.device)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        print(self.dataset)
        self.tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset.column_names)
        print(self.tokenized_datasets)


    def model_init(self):   
        return AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_path)).to(self.device)


    def load_dataset(self, train_split=90, kfold=True):
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
                yield train_ds, test_ds


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        print('decoded labels', decoded_labels)
        
        decoded_preds = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        print('decoded preds', decoded_preds)
        if self.reasoning_enabled: result = self.evaluator.calculate_metrics_by_format(preds=decoded_preds, labels=decoded_labels)
        else: result = self.evaluator.calculate_metrics(preds=decoded_preds, labels=decoded_labels)
        self.output_df = self.output_df.append({'precision': result['precision'],'recall': result['recall'],
                     'f1': result['f1'], 'preds': decoded_preds,'labels': decoded_labels}, ignore_index = True)
        #print('result : ', result)
        return result


    def my_hp_space_ray(self, trial):
        return {
            "learning_rate": tune.loguniform(5e-5, 4e-2),
            "num_train_epochs": tune.choice(range(10, 30)),
            "seed": tune.choice(range(1, 41)),
            "per_device_train_batch_size": tune.choice([16, 32, 64]),
            "weight_decay": tune.loguniform(0.001, 0.1)
        }


    def train(self):
        print("Training Model : ", self.model_name)
        
        training_args = Seq2SeqTrainingArguments(
            #f"{self.model_name}-finetuned-squad",
            output_dir="test_trainer",
            overwrite_output_dir = True,
            evaluation_strategy = "epoch",
            learning_rate = self.learning_rate,
            num_train_epochs = self.num_train_epochs,
            weight_decay = self.weight_decay,
            per_device_train_batch_size = self.per_device_train_batch_size,
            predict_with_generate = True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            #logging_steps=10,
            logging_dir = self.logs_dir+f'trial_{self.trial}',
            log_level = 'debug',
            logging_strategy = 'epoch'
            #push_to_hub=True,
        )
        print(training_args)
        metrics = []
        for train_ds, val_ds in self.load_dataset():
            
            print(train_ds.num_rows, val_ds.num_rows)
            trainer = Seq2SeqTrainer(
                #model_init = self.model_init,
                model=self.model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer = self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )
        
            trainer.train()

            # best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize",
            #     resources_per_trial={
            #         "cpu": 2,
            #         "gpu": 4
            #     },
            #     hp_space=self.my_hp_space_ray)
            # print('Best run : ', best_run)
            #metrics.append(trainer.evaluate())
        #print(pd.DataFrame.from_dict(metrics).mean())
        #print(metrics)
        #training_args.

        self.output_df.index += 1
        self.output_df = self.output_df[self.output_df.index % self.num_train_epochs == 0]
        
        df2 = self.output_df[['precision','recall','f1']].mean()
        new_row = pd.DataFrame({'precision': df2['precision'],'recall': df2['recall'],
                     'f1': df2['f1'], 'preds':f'model: {self.model_name}, learning_rate: {self.learning_rate}, num_train_epochs: {self.num_train_epochs}, weight_decay: {self.weight_decay}, ' \
                     f'per_device_train_batch_size: {self.per_device_train_batch_size}, gradient_accumulation_steps: {self.gradient_accumulation_steps}','labels': self.formatted_example}, index =[0])
        self.output_df = pd.concat([new_row, self.output_df]).reset_index(drop = True)
        result_file = self.result_dir + f'trial_{self.trial}.csv'
        self.output_df.to_csv(result_file)
        print("Check logs: ", self.logs_dir+f'trial_{self.trial}')
        print("Check results: ", result_file)
        print('Training Finished!!!')
        