import torch
from dataloader import InitiativeExcelLoader
from evaluator import Evaluator
from dataprocessor import CustomDataProcessor
from openprompt.plms import load_plm
from utils.HuggingFacePathHandler import get_huggingface_path
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    TrainingArguments,
    HfArgumentParser
)
from tqdm import tqdm
import time

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
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
        default=500, metadata={
            "help": "Maximum source length for tokenizer. Default value is 5000."
        }
    )
    plm_eval_mode: Optional[str] = field(
        default='store_true', metadata={
            "help": "Eval mode for Pre-trained language model. Default is 'store_true'."
        }
    )

@dataclass
class DataFormatArguments:
    """
    Arguments pertaining to how we need to format the input/output data for training our model.
    """
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

class PrefixTuner():
    
    def __init__(self, **kwargs):
        parser = HfArgumentParser((ModelArguments, DataFormatArguments, TrainingArguments))
        self.model_args, self.data_format_args, self.training_args = parser.parse_args_into_dataclasses()
        print(self.model_args)
        print(self.data_format_args)
        print(self.training_args)
        
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.model_args.model_name_or_path.split("/")[-1]
        
        self.result_dir = f'modular_approach/results/prefix-tuning/{self.model_name}/'
        self.logs_dir = f'modular_approach/logs/prefix-tuning/{self.model_name}/'
        
        self.generation_arguments = {
                "max_length": 512,
                "max_new_tokens": None,
                "min_length": 1,
                "temperature": 1.0,
                "do_sample": False,
                "top_k": 0,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "num_beams": 5,
                "bad_words_ids": [[628], [198]]
            }
        self._init_classes()
        #self._load_dataset()
        self.output_df = pd.DataFrame(columns=['precision','recall','f1','preds','labels'])
        #self.get_model()
        
    def _init_classes(self):
        self.dataloader = InitiativeExcelLoader()
        self.dataprocessor = CustomDataProcessor()
        self.evaluator = Evaluator()
    
    def split_dataset(self, train_split=90, kfold=True):
        train_data = {}
        test_data = {}
        num_rows = len(self.dataset)
        if not kfold:
            split_index = num_rows * train_split // 100
            train_data = self.dataset[:split_index]
            test_data = self.dataset[split_index:]
            train_dataloader = PromptDataLoader(dataset=train_data, template=self.mytemplate, tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.model_args.max_source_length, decoder_max_length=256,
                batch_size=self.training_args.per_device_train_batch_size, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
                truncate_method="head")

            test_dataloader = PromptDataLoader(dataset=test_data, template=self.mytemplate, tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.model_args.max_source_length, decoder_max_length=256,
                batch_size=self.training_args.per_device_eval_batch_size, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
                truncate_method="head")
            #print(train_ds)
            yield train_dataloader, test_dataloader

        else :
            step = num_rows * (100-train_split) // 100
            for k in range(0, num_rows, step):
                test_data = self.dataset[k: step+k]
                train_data = self.dataset[:k] + self.dataset[step+k:]
                train_dataloader = PromptDataLoader(dataset=train_data, template=self.mytemplate, tokenizer=self.tokenizer,
                    tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.model_args.max_source_length, decoder_max_length=256,
                    batch_size=self.training_args.per_device_train_batch_size, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
                    truncate_method="head")

                test_dataloader = PromptDataLoader(dataset=test_data, template=self.mytemplate, tokenizer=self.tokenizer,
                    tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.model_args.max_source_length, decoder_max_length=256,
                    batch_size=self.training_args.per_device_eval_batch_size, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
                    truncate_method="head")
                
                yield train_dataloader, test_dataloader

    def evaluate(self, prompt_model, dataloader):
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()

        for step, inputs in enumerate(dataloader):
            if self.use_cuda:
                inputs = inputs.cuda()
                #print('**inputs:**', inputs)
            _, output_sentence = prompt_model.generate(inputs, **self.generation_arguments)
            generated_sentence.extend(output_sentence)
            groundtruth_sentence.extend(inputs['tgt_text'])
        #score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
        score = self.evaluator.calculate_metrics(preds=generated_sentence, labels=groundtruth_sentence)
        print("test_score", score, flush=True)
        return generated_sentence

    def train(self):
        self.dataset = self.dataprocessor.get_test_examples()
        
        self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(self.model_args.model, get_huggingface_path(self.model_args.model_name_or_path))
        self.mytemplate = PrefixTuningTemplate(model=self.plm,  tokenizer=self.tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

        # wrapped_example = self.mytemplate.wrap_one_example(dataset["train"][0])
        # print(wrapped_example)

        self.use_cuda = True
        prompt_model = PromptForGeneration(plm=self.plm,template=self.mytemplate, freeze_plm=True,tokenizer=self.tokenizer, plm_eval_mode=self.model_args.plm_eval_mode)
        if self.use_cuda:
            prompt_model=  prompt_model.cuda()
        print("Prompt for generation done")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in self.mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
        ]

        for fold_index, (train_dataloader, test_dataloader) in enumerate(self.split_dataset()):
            print("Evaluating ------------------------- fold: {}".format(fold_index))

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate, eps=1e-8)

            tot_step  = len(train_dataloader)*5
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)
            print("Starting training...")
            # training and generation.
            global_step = 0
            tot_loss = 0
            log_loss = 0
            tot_train_time = 0
            pbar = tqdm(total=tot_step, desc="Train")
            pbar_update_freq = 10
            gradient_accumulation_steps = 4
            log_loss = 0
            glb_step = 0
            actual_step = 0
            for epoch in range(int(self.training_args.num_train_epochs)):
                prompt_model.train()
                print("Begin epoch: ", epoch)
                for step, inputs in enumerate(train_dataloader):
                    global_step +=1
                    if self.use_cuda:
                        inputs = inputs.cuda()

                    tot_train_time -= time.time()       
                    loss = prompt_model(inputs)
                    loss.backward()
                    tot_loss += loss.item()
                    actual_step += 1

                    if actual_step % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                        glb_step += 1
                        if glb_step % pbar_update_freq == 0:
                            aveloss = (tot_loss - log_loss)/pbar_update_freq
                            pbar.update(10)
                            pbar.set_postfix({'loss': aveloss})
                            log_loss = tot_loss
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    tot_train_time += time.time()
                    
            print("Done training...")
            generated_sentence = self.evaluate(prompt_model, test_dataloader)
            print("Evaluation done!!!")
            with open(f"{self.result_dir}{self.model_args.plm_eval_mode}_{self.data_format_args.trial}.txt",'a') as f:
                f.write(f'Fold : {fold_index}\n')
                for i in generated_sentence:
                    f.write(i+"\n") 