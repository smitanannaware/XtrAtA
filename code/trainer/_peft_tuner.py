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
    HfArgumentParser
)
from utils.HuggingFacePathHandler import get_huggingface_path
import torch
import numpy as np
# from ray import tune 
import os  
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
from peft import LoraConfig, get_peft_model, TaskType

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

class FineTuner():
    
    def __init__(self, **kwargs):
        parser = HfArgumentParser((ModelArguments, DataFormatArguments, Seq2SeqTrainingArguments))
        self.model_args, self.data_format_args, self.training_args = parser.parse_args_into_dataclasses()
        print(self.model_args)
        print(self.data_format_args)
        print(self.training_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.model_args.model_path.split("/")[-1]
        self.result_dir = f'modular_approach/results/{self.data_format_args.result_dir}{self.model_name}/'
        self.logs_dir = f'modular_approach/logs/{self.data_format_args.result_dir}{self.model_name}/'
        self._init_classes()
        self._load_dataset()
        self.output_df = pd.DataFrame(columns=['preds','labels'])
        self.get_model()
        self.bleu = evaluate.load("bleu")
        

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
        if self.data_format_args.instruction_id:
            if self.data_format_args.reasoning_enabled:
                data = self.formatter.fine_tune_reasoning_format(instruction=self.instructions[self.data_format_args.instruction_id])
            else:
                data = self.formatter.fine_tune_format(instruction=self.instructions[self.data_format_args.instruction_id])
            print(data['test'][0]['formatted_output'])
            print(data['test'][0]['label'])
            self.formatted_example = data['test'][0]['formatted_output']
            #self.formatted
            data_df = pd.DataFrame(data=data['test'])
        else:
            data_df = pd.DataFrame(self.dataloader.dataset['test'])
        
        if self.data_format_args.eos:
            data_df['label'] = data_df['label'].replace('None','', inplace=True)

        if self.data_format_args.reasoning_enabled or self.data_format_args.filter_dataset: data_df = data_df.loc[data_df['reasoning'] != 'None']
        data_df.to_csv(self.result_dir + f'shuffled_data_{self.data_format_args.trial}.csv')
        print("Saved shuffled data")
        self.dataset = datasets.Dataset.from_pandas(data_df).shuffle(seed=self.data_format_args.shuffle_seed)

        self.nc_cluster_map = get_nc_clusters()
        print('nc_cluster_map: ', self.nc_cluster_map)

        

    def preprocess_function(self, examples):
        if self.data_format_args.instruction_id:
            model_inputs = self.tokenizer(examples['formatted_output'], padding="longest", max_length=self.model_args.max_source_length, truncation=True)   
        else:
            model_inputs = self.tokenizer(examples['review'], padding="longest", max_length=self.model_args.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        if self.data_format_args.abstractive:
            column = "abs_true_nc"
        else:
            column = "true_nc"
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples[column], padding="longest", max_length=self.model_args.max_source_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    def get_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(get_huggingface_path(self.model_args.model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_args.model_path))#.to(self.device)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        print(self.dataset)
        self.tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset.column_names)
        print(self.tokenized_datasets)


    def model_init(self):   
        return AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_args.model_path))#.to(self.device)

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
                
                # For cluster evaluation
                train_nc_list = ', '.join(self.dataset['true_nc'][:k]+self.dataset['true_nc'][step+k:])
                self.train_nc_list = train_nc_list.lower().split(', ')

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

        if self.data_format_args.abstractive:
            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

            result = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["bleu"]}
            print("bleu score:", result)
            self.output_df = self.output_df.append({'bleu': result['bleu'], 'preds': decoded_preds, 'labels': decoded_labels}, ignore_index = True)
            
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}

            return result

        if self.data_format_args.reasoning_enabled: result = self.evaluator.calculate_metrics_by_format(preds=decoded_preds, labels=decoded_labels)
        else: result = self.evaluator.calculate_metrics(preds=decoded_preds, labels=decoded_labels)
    
        # Evaluate by Exact match, no penalization
        result_0 = self.evaluator.calculate_metrics_exact_match(preds=decoded_preds, labels=decoded_labels)
        # Evaluate by partial match with tokenization
        result_1 = self.evaluator.calculate_metrics_exact_match_with_partial(preds=decoded_preds, labels=decoded_labels)
        # Evaluate by partial match with space as separator
        result_11 = self.evaluator.calculate_metrics_exact_match_with_partial(preds=decoded_preds, labels=decoded_labels, tokenize=False)
        # Evaluate clusters
        print('train_nc_list', self.train_nc_list)
        train_nc_clusters = set([self.nc_cluster_map[aspect.strip()] if aspect.strip() in self.nc_cluster_map else float("inf") for aspect in self.train_nc_list])
        print('train_clusters: ', train_nc_clusters)
        result_2 = self.evaluator.calculate_metrics_by_clusters(preds=decoded_preds, labels=decoded_labels, train_clusters=train_nc_clusters, aspect_cluster_map=self.nc_cluster_map)
        
        self.output_df = self.output_df.append({'precision_bnp': result['precision'],'recall_bnp': result['recall'],'f1_bnp': result['f1'], 
                                            'precision_exactM': result_0['precision'],'recall_exactM': result_0['recall'],'f1_exactM': result_0['f1'],
                                            'precision_partialTokenized': result_1['precision'],'recall_partialTokenized': result_1['recall'],'f1_partialTokenized': result_1['f1'],
                                            'precision_partialWords': result_11['precision'],'recall_partialWords': result_11['recall'],'f1_partialWords': result_11['f1'],
                                            'precision_clusters': result_2['precision'],'recall_clusters': result_2['recall'],'f1_clusters': result_2['f1'],
                                            'preds': decoded_preds, 'labels': decoded_labels,
                                            'train_nc_list': self.train_nc_list}, ignore_index = True)
        print(f'result_bnp: {result}, result_exactM: {result_0}, result_partialTokenized: {result_1}, result_partialWords: {result_11}, result_clusters: {result_2}')
        #print('result : ', result)
        return result_0


    def output_results(self):
        self.output_df.index += 1
        model_info = f'model: {self.model_name}\n'\
                            f'learning_rate: {self.training_args.learning_rate}\n'\
                            f'num_train_epochs: {self.training_args.num_train_epochs}\n'\
                            f'weight_decay: {self.training_args.weight_decay}\n' \
                            f'per_device_train_batch_size: {self.training_args.per_device_train_batch_size}\n'\
                            f'gradient_accumulation_steps: {self.training_args.gradient_accumulation_steps}'

        metrics_columns = ['precision_bnp','recall_bnp','f1_bnp',
                                    'precision_exactM', 'recall_exactM', 'f1_exactM', 
                                    'precision_partialTokenized', 'recall_partialTokenized', 'f1_partialTokenized', 
                                    'precision_partialWords', 'recall_partialWords', 'f1_partialWords',
                                    'precision_clusters','recall_clusters','f1_clusters']
                                    
        if self.data_format_args.abstractive:
            df2 = self.output_df[['bleu']].mean()

            new_row = pd.DataFrame({'bleu': df2['bleu'], 'preds': model_info,
                        'labels': self.formatted_example,
                        }, index =[0])
        else:
            df2 = self.output_df[metrics_columns].mean()
            new_row = pd.DataFrame({'precision_bnp': df2['precision_bnp'],'recall_bnp': df2['recall_bnp'], 'f1_bnp': df2['f1_bnp'],
                        'precision_exactM': df2['precision_exactM'],'recall_exactM': df2['recall_exactM'], 'f1_exactM': df2['f1_exactM'],
                        'precision_partialTokenized': df2['precision_partialTokenized'],'recall_partialTokenized': df2['recall_partialTokenized'],'f1_partialTokenized': df2['f1_partialTokenized'],
                        'precision_partialWords': df2['precision_partialWords'],'recall_partialWords': df2['recall_partialWords'],'f1_partialWords': df2['f1_partialWords'],
                        'precision_clusters': df2['precision_clusters'],'recall_clusters': df2['recall_clusters'],'f1_clusters': df2['f1_clusters'],
                        'preds': model_info,
                        'labels': self.formatted_example,
                        'train_nc_list': ''}, index =[0])

        all_df = pd.concat([new_row, self.output_df]).reset_index(drop = True)         
        result_file = self.result_dir + f'trial_{self.data_format_args.trial}_all.csv'
        all_df.to_csv(result_file)

        # Consider only last(best) result from every epoch
        self.output_df = self.output_df[self.output_df.index % self.training_args.num_train_epochs == 0]
        
        if self.data_format_args.abstractive:
            df2 = self.output_df[['bleu']].mean()

            new_row = pd.DataFrame({'bleu': df2['bleu'], 'preds': model_info,
                        'labels': self.formatted_example,
                        }, index =[0])
        else:
            df2 = self.output_df[metrics_columns].mean()
            new_row = pd.DataFrame({'precision_bnp': df2['precision_bnp'],'recall_bnp': df2['recall_bnp'], 'f1_bnp': df2['f1_bnp'],
                        'precision_exactM': df2['precision_exactM'],'recall_exactM': df2['recall_exactM'], 'f1_exactM': df2['f1_exactM'],
                        'precision_partialTokenized': df2['precision_partialTokenized'],'recall_partialTokenized': df2['recall_partialTokenized'],'f1_partialTokenized': df2['f1_partialTokenized'],
                        'precision_partialWords': df2['precision_partialWords'],'recall_partialWords': df2['recall_partialWords'],'f1_partialWords': df2['f1_partialWords'],
                        'precision_clusters': df2['precision_clusters'],'recall_clusters': df2['recall_clusters'],'f1_clusters': df2['f1_clusters'],
                        'preds': model_info,
                        'labels': self.formatted_example,
                        'train_nc_list': ''}, index =[0])
        
        self.output_df = pd.concat([new_row, self.output_df]).reset_index(drop = True)
        result_file = self.result_dir + f'trial_{self.data_format_args.trial}.csv'
        self.output_df.to_csv(result_file)
        
        print("Check logs: ", self.logs_dir+f'trial_{self.data_format_args.trial}')
        print("Check results: ", result_file)

    
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    def train(self):
        print("Training Model : ", self.model_name)
        
        peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1
            )
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        model = get_peft_model(model, peft_config)
        self.print_trainable_parameters(model)
        
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
            generation_max_length=self.training_args.generation_max_length,
            #logging_steps=10,
            logging_dir = self.logs_dir+f'trial_{self.data_format_args.trial}',
            log_level = self.training_args.log_level,
            logging_strategy = self.training_args.logging_strategy
            #push_to_hub=True,
        )
        
        for train_ds, val_ds in self.split_dataset():
            
            print(f"Training set samples: {len(train_ds)} \t Testset samples: {len(val_ds)}")
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
 
        self.output_results()
        print('Training Finished!!!')
        
        


        
        