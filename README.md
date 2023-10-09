# Extraction of Atypical Aspects from Customer Reviews: Datasets and Experiments with Language Models
This repository contains the datasets described in the paper above.

## Dataset
The datasets are available in the dataset directory for restaurant, hotel, and hair salon domains. Each dataset has a train_test and dev directory with data in Excel format, and the train_test folder contains the data used in 10-fold evaluation.

The table below shows details of the columns in the dataset.
| Column | Description |
| ------ | ----------- |
| review | The user review text for the domain. |
| true_strong | Extractive annotation for primary atypical aspects. |
| true_weak | Extractive annotation for secondary atypical aspects. |
| true_other | Extractive annotation for other atypical aspects that do not fall in the strong or weak category. |
| true_strong_weak | Extractive primary and secondary atypical aspects are combined. |
| abs_true_strong | Abstractive annotation for primary atypical aspects. |
| abs_true_weak | Abstractive annotation for secondary atypical aspects. |
| abs_true_other | Abstractive annotation for other atypical aspects that do not fall in the primary or secondary category. |
| abs_true_strong_weak | Abstractive primary and secondary atypical aspects are combined. |

## Code

### Finetuning of FLan T5 models

Run the following command to finetune the FLan T5 models on any of the datasets.

* model_path: Path to the FLan T5 model from HuggingFace.
* data_dir: Path to the dataset.
* target_column_name: Name of the column containing the target text.
* template_name: Name of the template.
* template_id: ID of the template.
* instruction_id: ID of the instruction.
* trial: Experiment number.
* shuffle_seed: Seed for shuffling the dataset.
* eos: Whether to add the EOS token at the end of the target text.
* result_dir: Path to the directory where the results will be stored.
* output_dir: Path to the directory where the output will be stored.
* per_device_train_batch_size: Training batch size.
* per_device_eval_batch_size: Evaluation batch size.
* learning_rate: Learning rate.
* num_train_epochs: Number of training epochs.

```
python -u code/fine_tune.py 
--model_path=google/flan-t5-base 
--data_dir=salons/v2    
--target_column_name=true_strong  
--template_name=t5  
--template_id=3  
--instruction_id=27  
--trial=1001  
--shuffle_seed=42  
--eos  
--result_dir=extractive/fine-tuning/eos/  
--output_dir=code/test_trainer/extractive/fine-tuning/eos/  
--per_device_train_batch_size=1  
--gradient_accumulation_steps=16  
--num_train_epochs=30  
--learning_rate=5e-5  
--weight_decay=0.001  
--overwrite_output_dir  
--evaluation_strategy=epoch  
--generation_max_length=512  
--log_level=debug  
--logging_strategy=epoch  
--gradient_checkpointing
```

## ChatGPT Zero Experiments

Run the following command to ChatGPT zero-shot inferences on any of the datasets.

```
python -u code/chatGPT_few_shot_exp.py 
--data_dir=restaurant/v5    
--ref_column_name=true_strong  
--domain=restaurant
--file_name=trial_1
--result_dir=extractive/zero-shot/eos
--prompt=Given the following restaurant review, can you list atypical aspects for a restaurant? Atypical aspects are not related to service, food, drinks, location, price, menu, discounts, policies, staff, customer satisfaction, or other items commonly associated with a restaurant. Please be precise in your response; it should contain only atypical aspects associated with the restaurant that is reviewed. Extract base noun phrases in the output format as below: 'Atypical aspects: aspect 1, aspect 2, aspect 3.' Output <None> if there are no atypical aspects. Please follow the output format strictly.  Passage:
```


## ChatGPT Few-shot Experiments

Run the following command to ChatGPT few-shot inferences on any of the datasets.

```
python -u code/chatGPT_few_shot_exp.py 
--data_dir=restaurants/v5    
--ref_column_name=true_strong  
--domain=restaurant
--file_name=trial_1
--result_dir=extractive/few-shot/eos
--indices=1,2,3,4,5
--prompt=Given the following restaurant review, can you list atypical aspects for a restaurant? Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Please be precise in your response and it should contain only atypical aspects that are associated with the restaurant that is reviewed. Output <None> if there are no atypical aspects. 
```


