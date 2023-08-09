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
    HfArgumentParser,
    EarlyStoppingCallback, 
    IntervalStrategy
)
from utils.HuggingFacePathHandler import get_huggingface_path
import torch
import numpy as np
# from ray import tune 
import os, sys  
from prompts import PromptTemplate
from formatter import DatasetFormatter
from utils.constants import Models
from prompts import InstructionLoader
from dataclasses import dataclass, field
from typing import Optional
# from accelerate import Accelerator
from transformers import  get_linear_schedule_with_warmup, set_seed

from torch.utils.data import DataLoader
from utils.trainerHelper import get_nc_clusters
import evaluate
import pathlib

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: Optional[str] = field(
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
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The input data dir. Should contain the.xlsx files (or other data files) for the task"}
    )
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
    result_dir: str = field(
        default=None, metadata={"help": "The folder path to store results e.g. fine-tuning/v2/ "}
    )
    abstractive: bool = field(
        default=False, 
        metadata={
            "help": "Choose the data label type."
        }
    )
    eos: bool = field(
        default=False, 
        metadata={
            "help": "Enable EOS token as label for only core reviews"
        }
    )
    target_column_name: Optional[str] = field(
        default='gold', metadata={"help": "The name of the model to select a template file for formatting data e.g T5, TkInstruct."}
    )

class GPTFineTuner():
    
    def __init__(self, **kwargs):
        parser = HfArgumentParser((ModelArguments, DataFormatArguments, Seq2SeqTrainingArguments))
        self.model_args, self.data_format_args, self.training_args = parser.parse_args_into_dataclasses()
        print(self.model_args)
        print(self.data_format_args)
        print(self.training_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.model_args.model_path.split("/")[-1]
        
        self.type_of_nc = 'strong'
        if 'weak' in self.data_format_args.target_column_name:
            self.type_of_nc = 'strong_weak'
        
        self.result_dir = f'modular_approach/results/{self.data_format_args.result_dir}{self.data_format_args.data_dir}/{self.model_name}/{self.type_of_nc}'
        self.logs_dir = f'modular_approach/logs/{self.data_format_args.result_dir}{self.data_format_args.data_dir}/{self.model_name}/{self.type_of_nc}'
        self.model_dir = f'modular_approach/models/{self.data_format_args.result_dir}{self.data_format_args.data_dir}/{self.model_name}/{self.type_of_nc}'
        self.data_folder = f'modular_approach/dataset/{self.data_format_args.data_dir}/'
        self.output_dir = f'{self.training_args.output_dir}{self.data_format_args.data_dir}/{self.model_name}/{self.type_of_nc}'
        pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self._init_classes()
        self._load_dataset()
        self.output_df = pd.DataFrame(columns=['preds','labels'])
        self.get_model()
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.fold_data = pd.read_csv('/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/v5/gpt/shuffled_data.csv')
        self.fold_data = self.fold_data.replace('None',str(''))
        self.fold_data = self.fold_data.fillna('')
        
        

    def _init_classes(self):
        self.dataloader = InitiativeExcelLoader()
        if self.data_format_args.template_name:
            self.template_obj = PromptTemplate(self.data_format_args.template_name, self.data_format_args.template_id)
            #if self.model_path == Models.T5_FLAN_SMALL:
            self.formatter = DatasetFormatter(self.template_obj, self.dataloader)
            self.instructions = InstructionLoader().get_instructions()
        self.evaluator = Evaluator()
                
        
    def _load_dataset(self):
        self.formatted_example = None
        data_df = pd.DataFrame(self.dataloader.dataset['test'])
        
        column = self.data_format_args.target_column_name

        #if self.data_format_args.eos:
        data_df = data_df.replace('None',str(''))

        data_df = data_df.loc[data_df['added'] != 'yes']
        
        self.dataset = datasets.Dataset.from_pandas(data_df).shuffle(seed=self.data_format_args.shuffle_seed)
        self.dataset.to_csv(self.result_dir + f'shuffled_data_{self.data_format_args.trial}.csv')
        print("Saved shuffled data")
                

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples['chatgpt_output'], padding="longest", truncation=True)

        print('Column to be consider as gold:', self.data_format_args.target_column_name)
        with self.tokenizer.as_target_tokenizer():
            #print(examples[column])
            labels = self.tokenizer(examples[self.data_format_args.target_column_name], padding="longest", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    

    def get_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(get_huggingface_path(self.model_args.model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_args.model_path))#.to(self.device)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        print(self.dataset)
        
        #self.tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset.column_names)
        #print(self.tokenized_datasets)


    def model_init(self):   
        return AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_args.model_path))#.to(self.device)

    def collate_fn(self,examples):
                return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")


    def getFoldDataset(self, fold):

        train_data = self.fold_data.loc[self.fold_data['fold'] != fold]
        test_data = self.fold_data.loc[self.fold_data['fold'] == fold]
        test_data = test_data.loc[self.fold_data['added'] != 'yes']

        train_dataset = datasets.Dataset.from_pandas(train_data)
        train_ds = train_dataset.map(self.preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        
        test_dataset = datasets.Dataset.from_pandas(test_data)
        test_ds = test_dataset.map(self.preprocess_function, batched=True, remove_columns=test_dataset.column_names)

        return train_ds, test_ds


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
            
            yield train_ds, test_ds

        else :
            step = num_rows * (100-train_split) // 100
            rem = num_rows % (100-train_split)
            k1 = 0
            while k1 < num_rows:
                if rem > 0:
                    k2 = step+k1+1
                    rem -= 1
                else:
                    k2 = step+k1
                for column in self.tokenized_datasets.column_names:
                    test_data[column] = self.tokenized_datasets[column][k1: k2]
                    train_data[column] = self.tokenized_datasets[column][:k1] + self.tokenized_datasets[column][k2:]

                k1 = k2
                train_ds = datasets.Dataset.from_dict(train_data)
                test_ds = datasets.Dataset.from_dict(test_data)
            
                yield train_ds, test_ds
    

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        print('decoded labels', decoded_labels)
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        print('decoded preds', decoded_preds)

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        try:
            result = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)
        except ZeroDivisionError:
            result = {'bleu': 0.0}
        result = {"bleu": result["bleu"]}
        print("bleu score:", result)
        self.output_df = self.output_df.append({'bleu': result['bleu'], 'preds': decoded_preds, 'labels': decoded_labels}, ignore_index = True)
        
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    
    
    # def my_hp_space_ray(self, trial):
    #     return {
    #         "learning_rate": tune.loguniform(5e-5, 4e-2),
    #         "num_train_epochs": tune.choice(range(10, 30)),
    #         "seed": tune.choice(range(1, 41)),
    #         "per_device_train_batch_size": tune.choice([16, 32, 64]),
    #         "weight_decay": tune.loguniform(0.001, 0.1)
    #     }

    def output_results(self):
        self.output_df.index += 1
        model_info = f'model: {self.model_name}\n'\
                            f'learning_rate: {self.training_args.learning_rate}\n'\
                            f'num_train_epochs: {self.training_args.num_train_epochs}\n'\
                            f'weight_decay: {self.training_args.weight_decay}\n' \
                            f'per_device_train_batch_size: {self.training_args.per_device_train_batch_size}\n'\
                            f'gradient_accumulation_steps: {self.training_args.gradient_accumulation_steps}'

                                   
        df2 = self.output_df[['bleu']].mean()

        new_row = pd.DataFrame({'bleu': df2['bleu'], 'preds': model_info,
                    'labels': self.formatted_example,
                    }, index =[0])
        

        all_df = pd.concat([new_row, self.output_df]).reset_index(drop = True)         
        result_file = self.result_dir + f'trial_{self.data_format_args.trial}_all.csv'
        all_df.to_csv(result_file)

        # Consider only last(best) result from every epoch
        self.output_df = self.output_df[self.output_df.index % self.training_args.num_train_epochs == 0]
        
        df2 = self.output_df[['bleu']].mean()

        new_row = pd.DataFrame({'bleu': df2['bleu'], 'preds': model_info,
                    'labels': self.formatted_example,
                    }, index =[0])
        
        
        self.output_df = pd.concat([new_row, self.output_df]).reset_index(drop = True)
        result_file = self.result_dir + f'trial_{self.data_format_args.trial}.csv'
        self.output_df.to_csv(result_file)
        
        print("Check logs: ", self.logs_dir+f'trial_{self.data_format_args.trial}')
        print("Check results: ", result_file)

    def train(self):
        print("Training Model : ", self.model_name)

        training_args = Seq2SeqTrainingArguments(
            #f"{self.model_name}-finetuned-squad",
            output_dir=self.output_dir+f'trial_{self.data_format_args.trial}',
            overwrite_output_dir = self.training_args.overwrite_output_dir,
            evaluation_strategy = self.training_args.evaluation_strategy,
            learning_rate = self.training_args.learning_rate,
            num_train_epochs = self.training_args.num_train_epochs,
            weight_decay = self.training_args.weight_decay,
            per_device_train_batch_size = self.training_args.per_device_train_batch_size,
            predict_with_generate = True,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            generation_max_length=self.training_args.generation_max_length,
            logging_dir = self.logs_dir+f'trial_{self.data_format_args.trial}',
            log_level = self.training_args.log_level,
            logging_strategy = self.training_args.logging_strategy,
            # save_strategy='no',
            # save_total_limit=2,
            # load_best_model_at_end=False
            save_strategy=self.training_args.evaluation_strategy,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model='loss'
        )
       
        i = 1
        for train_ds, val_ds in self.split_dataset():
        #while i <= 10:
            #train_ds, val_ds = self.getFoldDataset(i)
            print(f"Training Fold : {i}, Training set samples: {len(train_ds)} \t Testset samples: {len(val_ds)}")
            # if i == 1:
            trainer = Seq2SeqTrainer(
                model_init = self.model_init,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer = self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
            )
        
            trainer.train()

            trainer.save_model(f'{self.model_dir}trial_{self.data_format_args.trial}/out_fold{i}')
            i += 1
           

            # best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize",
            #     resources_per_trial={
            #         "cpu": 2,
            #         "gpu": 4
            #     },
            #     hp_space=self.my_hp_space_ray)
            # print('Best run : ', best_run)
            #metrics.append(trainer.evaluate())
        
        self.output_results()
        print('Training Finished!!!')

        
if __name__ == "__main__":
    args = sys.argv
    GPTFineTuner().train()