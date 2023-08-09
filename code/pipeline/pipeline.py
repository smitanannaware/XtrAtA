from dataloader import InitiativeExcelLoader
from prompts import InstructionLoader
from model_experiments import TestModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.HuggingFacePathHandler import get_huggingface_path
import torch
from evaluator import Evaluator
import pandas as pd
import numpy as np
from utils.constants import Models
from prompts import PromptTemplate
from formatter import DatasetFormatter
from extractor import BNPExtractor
import pathlib
import random, itertools

class Pipeline():

    def __init__(self, args) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = args.model_name_or_path
        #self.model_name = model_path.split("/")[-1]
        self.model_name = args.model_name_or_path.split("/")[-1]
        self.zero_shot = args.zero_shot
        data_folder = f'modular_approach/dataset/{args.data_dir}/'
        self.data_loader = InitiativeExcelLoader()
        self.instructions = InstructionLoader().get_instructions()
        self.instruction_id = args.instruction_id
        self.template = args.template_name
        self.template_id = args.template_id
        self.ref_column = args.ref_column
        self.few_shot_examples = 2
        self.batch_size = 10
        self.trial = args.trial
        self.abstractive = args.abstractive
        self.type_of_nc = 'strong'
        if 'weak' in self.ref_column:
            self.type_of_nc = 'strong_weak'
        self.result_dir = f'modular_approach/results/{args.result_dir}{args.data_dir}/{self.model_name}/{self.type_of_nc}/'
        pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        self._init_classes()


    def _init_classes(self):
        self.template_obj = PromptTemplate(self.template, self.template_id)
        #if self.model_path == Models.T5_FLAN:
        self.formatter = DatasetFormatter(self.template_obj, self.data_loader)
        self.evaluator = Evaluator()


    def get_model(self):
        tokenizer = AutoTokenizer.from_pretrained(get_huggingface_path(self.model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(self.model_path)).to(self.device)
        return model, tokenizer


    def get_input_data(self, instruction):
        if self.zero_shot:
            data = self.formatter.format(instruction=instruction)
            test_data = pd.DataFrame.from_dict(data['test'])
            print('Formatted data: ',test_data['formatted_output'])
            yield test_data, None
        else:
            max_combination_len = self.few_shot_examples
            number_of_items = [1, 2, 6, 0, 8, 10, 11, 12, 13, 14, 18, 19]
            list_of_combinations = []
            for i in range(1, max_combination_len+1):
                perm = list(itertools.permutations(number_of_items, i))
                #random.shuffle(perm)
                list_of_combinations.extend(set(frozenset(item)for item in perm))

            for indexes in list_of_combinations:
                print(indexes)
                train_examples = [self.data_loader.dataset['dev'][i] for i in indexes]
                
                data = self.formatter.formatFewShotExample(instruction=instruction, fewShotEx = train_examples)

                test_data = pd.DataFrame.from_dict(data['test'])
                #print('Formatted data: ',test_data['formatted_output'])
                yield test_data, indexes


    def output_to_file(self, metrics, model_output, examples, indexes):
        if indexes:
            header = pd.MultiIndex.from_product([self.instructions if not self.instruction_id else [self.instructions[self.instruction_id]],
                                    indexes]
                                    # names=['Prompts','Review/Results']
                                    )
        else:
            header = pd.MultiIndex.from_product([self.instructions if not self.instruction_id else [self.instructions[self.instruction_id]],
                                    ],
                                    names=['Prompts'])

        data = np.array(model_output)
        df = pd.DataFrame(data=data.T, columns=header)
        df.insert(0, 'true_strong', examples['true_strong'])
        df.insert(0, 'true_strong_weak', examples['true_strong_weak'])
        df.insert(0, 'abs_true_strong_alt', examples['abs_true_strong_alt'])
        df.insert(0, 'abs_true_strong_weak_alt', examples['abs_true_strong_weak_alt'])
        df.insert(0, 'review', examples['review'])
        a = [examples['formatted_output'][0], '', '', '', '']
        a.extend(metrics)
        print(a)
        df.loc[len(df)] = a
        df.to_excel(self.result_dir+f'trial_{self.trial}.xlsx')

        # self.instructions.insert(0, 'Format')
        # metrics.insert(0, examples['formatted_output'][0])
        # pd.DataFrame({'prompt': self.instructions[:2], 'metrics': metrics}).to_csv(self.result_dir+f'metrics_results_{self.model_name}.csv')

    
    def compute_extractive_metrics(self, preds, golds):
        exact = self.evaluator.calculate_metrics_exact_match(preds=preds, labels=golds)
        partial_tokenized = self.evaluator.calculate_metrics_exact_match_with_partial(preds=preds, labels=golds)
        partial_bywords = self.evaluator.calculate_metrics_exact_match_with_partial(preds=preds, labels=golds, tokenize=False)

        return exact, partial_tokenized, partial_bywords
    

    def compute_abstractive_metrics(self, preds, golds):
        rouge = self.evaluator.compute_rouge_aggregated(predictions=preds, references=golds)
        bert = self.evaluator.compute_bertscore(predictions=preds, references=golds)
        bert = {'precision': round(np.mean(bert['precision']), 2), 'recall': 
                                       round(np.mean(bert['recall']), 2),'f1': round(np.mean(bert['f1']), 2)}

        return rouge, bert


    def run_experiment(self, model, tokenizer):
        results = []
        model_output = []
        indexes = []
        print(len(self.instructions))
        self.instructions = self.instructions[-5:]

        for instruction in self.instructions if not self.instruction_id else [self.instructions[self.instruction_id]]:
            for test_data, index in self.get_input_data(instruction):
                #test_data = test_data[:10]
                print(test_data['formatted_output'][0])
                testModel = TestModel(test_data['formatted_output'].values.tolist(), self.batch_size, model, tokenizer, self.device)
                testModel.run()
                preds = testModel.predictions
                #preds = list(map(lambda x: x.replace('There are no atypical aspects.', ''), preds))
                preds = list(map(lambda x: x.replace('no response', ''), preds))
                #print(testModel.predictions)
                #print(test_data['label'].values.tolist())
                # result, output_bnps = self.evaluator.calculate_metrics_1(preds=preds, labels=test_data['label'].values.tolist())
                # results.append(result)
                
                if not self.abstractive:
                    if not self.zero_shot:
                        if self.type_of_nc == 'strong':
                            golds = test_data['true_strong'].values.tolist()
                            exact_match, partial_tokenized, partial_bywords = self.compute_extractive_metrics(preds=preds, golds=golds)
                            results.append({'strong_nc_exact': exact_match, 'strong_nc_partial_tokenized': partial_tokenized, 'strong_nc_partial_bywords': partial_bywords})
                        else:
                            golds = test_data['true_strong_weak'].values.tolist()
                            exact_match, partial_tokenized, partial_bywords = self.compute_extractive_metrics(preds=preds, golds=golds)
                            results.append({'strong_weak_nc_exact': exact_match, 'strong_weak_nc_partial_tokenized': partial_tokenized, 'strong_weak_nc_partial_bywords': partial_bywords})
                    else:
                        golds = test_data['true_strong'].values.tolist()
                        strong_nc_result_exact, strong_nc_result_partial_tokenized, strong_nc_result_partial_bywords = self.compute_extractive_metrics(preds=preds, golds=golds)
                         
                        golds = test_data['true_strong_weak'].values.tolist()
                        strong_weak_nc_result_exact, strong_weak_nc_result_partial_tokenized, strong_weak_nc_result_partial_bywords = self.compute_extractive_metrics(preds=preds, labels=golds)
                        results.append({'strong_nc_exact': strong_nc_result_exact, 'strong_nc_partial_tokenized': strong_nc_result_partial_tokenized, 'strong_nc_partial_bywords': strong_nc_result_partial_bywords, 'strong_weak_nc_exact': strong_weak_nc_result_exact, 'strong_weak_nc_partial_tokenized': strong_weak_nc_result_partial_tokenized, 'strong_weak_nc_partial_bywords': strong_weak_nc_result_partial_bywords})
                
                else:
                    if not self.zero_shot:
                        if self.type_of_nc == 'strong':
                            golds = test_data['abs_true_strong_alt'].values.tolist()
                            rouge, bert = self.compute_abstractive_metrics(preds=preds, golds=golds)
                            results.append({'strong_nc_rouge': rouge, 'strong_nc_bert': bert})
                        else:
                            golds = test_data['abs_true_strong_weak_alt'].values.tolist()
                            rouge, bert = self.compute_abstractive_metrics(preds=preds, golds=golds)
                            results.append({'strong_weak_nc_rouge': rouge, 'strong_weak_nc_bert': bert})
                    else:
                        golds = test_data['abs_true_strong_alt'].values.tolist()
                        strong_nc_result_rouge, strong_nc_result_bert = self.compute_abstractive_metrics(preds=preds, golds=golds)

                        golds = test_data['abs_true_strong_weak_alt'].values.tolist()
                        strong_weak_nc_result_rouge, strong_weak_nc_result_bert = self.compute_abstractive_metrics(preds=preds, golds=golds)
                        results.append({'strong_nc_rouge': strong_nc_result_rouge, 'strong_nc_bert': strong_nc_result_bert, 'strong_weak_nc_rouge': strong_weak_nc_result_rouge, 'strong_weak_nc_bert': strong_weak_nc_result_bert})
            
                model_output.append(preds)
                if index:
                    indexes.append(','.join(str(x) for x in index))
            #model_output.append(output_bnps)
            #break # TODO remove
        self.output_to_file(results, model_output, test_data, indexes)


    def run(self):
        model, tokenizer = self.get_model()
        self.run_experiment(model, tokenizer)


    # write main method

    
