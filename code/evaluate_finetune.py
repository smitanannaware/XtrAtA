

from utils.trainerHelper import get_nc_clusters
from evaluator import Evaluator
import pandas as pd
import numpy as np
import argparse
import re
import evaluate
import datasets
import spacy
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)
from prompts import PromptTemplate
from formatter import TemplateFormatter
from prompts import InstructionLoader

class Score():

    def __init__(self, domain, abstractive = False):
        self.evaluator = Evaluator()
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.abstractive = abstractive
        self.sum_p = 0
        self.sum_r = 0
        self.sum_f = 0
        self.num_of_examples = 0
        self.bnp_to_sent_mapping = pd.read_excel('/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/restaurants/v6/extractive_to_abstractive_mapping.xlsx')
        self.gold_to_sentences = pd.read_excel('/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/restaurants/v6/sent separated.xlsx')
        self.nc_cluster_map = get_nc_clusters()
        self.preds = []
        self.golds = []
        self.max_order = 1
        self.domain = domain
    

    def load_formatter(self, **kwargs):
        self.template_obj = PromptTemplate(kwargs['template_name'], kwargs['template_id'])
        #if self.model_path == Models.T5_FLAN_SMALL:
        self.formatter = TemplateFormatter(self.template_obj)
        self.instruction = InstructionLoader().get_instructions()[kwargs['instruction_id']]

    def get_nc_golds_preds(self, golds, preds):
        filtered_golds = []
        filtered_preds = []
        for gold, pred in zip(golds, preds):
            if gold != '':
                filtered_golds.append(gold)
                filtered_preds.append(pred)
        
        return filtered_golds, filtered_preds



    def evaluate_row(self, row):
        #print(row)
        pred, gold = row['preds'], row['labels']
        pred = eval(pred)
        gold = eval(gold)
        #gold = np.ravel(eval(gold)).tolist()

        #gold, pred = self.get_nc_golds_preds(golds=gold, preds=pred)

        self.preds.extend(pred)
        self.golds.extend(gold)
        
        print('pred : ',pred)
        print('gold : ',gold)

        row['classification'] = self.evaluator.compute_clasification_metrics(preds=pred, labels=gold)

        if self.abstractive:
            try:
                result = self.bleu.compute(predictions=pred, references=gold, max_order=self.max_order)
            except ZeroDivisionError:
                result = {'bleu': 0.0}
            print('BLEU score: ',result)
            row['bleu'] = round(result['bleu'], 2)

            result = self.evaluator.compute_rouge_aggregated(predictions=pred, references=gold)
            row['rouge1'], row['rouge2'], row['rouge3'], row['rouge4'] = result['rouge1']['f1'], result['rouge2']['f1'],result['rouge3']['f1'], result['rouge4']['f1']

            row['rougeL'], row['rougeLsum'] = round(result['rougeL'], 2), round(result['rougeLsum'], 2)
            
            results = self.evaluator.compute_bertscore(predictions=pred, references=gold)
            print(results)

            self.sum_p += np.sum(results['precision'])
            self.sum_r += np.sum(results['recall'])
            self.sum_f += np.sum(results['f1'])
            self.num_of_examples += len(results['recall'])

            row['bertscore_avg_p'] = round(np.mean(results['precision']), 2)
            row['bertscore_avg_r'] = round(np.mean(results['recall']), 2)
            row['bertscore_avg_f1'] = round(np.mean(results['f1']), 2)

            # Cluster measure
            # train_nc_clusters = set([self.nc_cluster_map[aspect.strip()] if aspect.strip() in self.nc_cluster_map 
            #                          else float("inf") for aspect in eval(row['train_nc_list'])])
            # print('train_clusters: ', train_nc_clusters)
            # result = self.evaluator.calculate_metrics_by_clusters_abstractive(preds=pred, labels=gold, train_clusters=train_nc_clusters,
            #                                      aspect_cluster_map=self.nc_cluster_map, bnp_to_sent_mapping=self.bnp_to_sent_mapping, golds_to_sentences=self.gold_to_sentences)
            # row['p_clusters'], row['r_clusters'], row['f1_clusters'], row['new_golds'] = result['precision'], result['recall'], result['f1'], result['new_golds']
            row['p_clusters'], row['r_clusters'], row['f1_clusters'], row['new_golds'] = 0, 0, 0, 'None'

        else:   

            precision_exactM, recall_exactM, f1_exactM = 0,0,0
            precision_partialTokenized, recall_partialTokenized, f1_partialTokenized = 0, 0, 0
            precision_partialWords, recall_partialWords, f1_partialWords = 0, 0, 0
            precision_clusters, recall_clusters, f1_clusters = 0, 0, 0

            if gold:
                result = self.evaluator.calculate_metrics_exact_match(pred, gold)
                precision_exactM, recall_exactM, f1_exactM = result['precision'], result['recall'], result['f1']
                result = self.evaluator.calculate_metrics_exact_match_with_partial(pred, gold)
                precision_partialTokenized, recall_partialTokenized, f1_partialTokenized = result['precision'], result['recall'], result['f1']
                result = self.evaluator.calculate_metrics_exact_match_with_partial(pred, gold, tokenize=False)
                precision_partialWords, recall_partialWords, f1_partialWords = result['precision'], result['recall'], result['f1']
                
                
                if self.domain == 'restaurant':
                    train_nc_clusters = set([self.nc_cluster_map[aspect.strip()] if aspect.strip() in self.nc_cluster_map else float("inf") for aspect in row['train_nc_list']])
                    print('train_clusters: ', train_nc_clusters)
                    result = self.evaluator.calculate_metrics_by_clusters(preds=pred, labels=gold, train_clusters=train_nc_clusters, aspect_cluster_map=self.nc_cluster_map)
                    precision_clusters, recall_clusters, f1_clusters = result['precision'], result['recall'], result['f1']
                else:
                    precision_clusters, recall_clusters, f1_clusters = 0,0,0

            row['precision_exactM'], row['recall_exactM'], row['f1_exactM'] = round(precision_exactM, 2), round(recall_exactM, 2), round(f1_exactM, 2)
            row['precision_partialTokenized'], row['recall_partialTokenized'], row['f1_partialTokenized'] = round(precision_partialTokenized, 2), round(recall_partialTokenized, 2), round(f1_partialTokenized, 2)
            row['precision_partialWords'], row['recall_partialWords'], row['f1_partialWords'] = round(precision_partialWords, 2), round(recall_partialWords, 2), round(f1_partialWords, 2)
            row['precision_clusters'], row['recall_clusters'], row['f1_clusters'] = round(precision_clusters, 2), round(recall_clusters, 2), round(f1_clusters, 2)
        
        return row
    
    def format_data(self, reviews):
        formatted_inputs = []
        
        for review in reviews:
            data_dict = {'instruction': self.instruction, 'review': review, 'label': ''}
            formatted_inputs.append(self.formatter.format(data_dict, hide_label=True))

        return formatted_inputs
    

def getPredictions(model_path, reviews, labels=None):
    batch_size = 22
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    golds = []
    num_rows = len(reviews)
    print('Input example: ', reviews[0])
    step = num_rows * (100-90) // 100
    rem = num_rows % (100-90)
    k1 = 0
    i = 0
    while k1 < num_rows:
        if rem > 0:
            k2 = step+k1+1
            rem -= 1
        else:
            k2 = step+k1
        
        inputs = reviews[k1 : k2]
        model_id = f'{model_path}out_fold{i+1}'
        print(model_id)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print('Model load complete!!!')
               
        model_inputs = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt").to(device)
        #print(model_inputs)
        
        logits = model.generate(input_ids=model_inputs['input_ids'], max_length=256).to(device)

        decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
        print('decoded preds', decoded_preds)
        preds.append(decoded_preds)
        if labels:
            golds.append(labels[k1 : k2])
        k1 = k2
        i += 1
    return preds, golds



if __name__ == "__main__":

    # python modular_approach/evaluate_finetune.py --model_dir=abstractive/fine-tuning/eos/salons/v1/flan-t5-xl/strong --file_name=trial_103 --get_predictions=True --domain=salons
     # python modular_approach/evaluate_finetune.py --model_dir=abstractive/fine-tuning/eos/salons/v1/flan-t5-xl/strong --file_name=trial_201 --get_predictions=True --domain=salons --template_name=t5 --template_id=3 --instruction_id=26
    # python modular_approach/evaluate_finetune.py --model_dir=extractive/fine-tuning/eos/salons/v1/flan-t5-xl/strong --file_name=trial_202 --get_predictions=True --domain=salons
    # python modular_approach/evaluate_finetune.py --model_dir=extractive/fine-tuning/eos/salons/v1/flan-t5-xl/strong --file_name=trial_203 --get_predictions=True --domain=salons --template_name=t5 --template_id=3 --instruction_id=25
    #python modular_approach/evaluate_finetune.py --model_dir=extractive/fine-tuning/eos/salons/v2/flan-t5-xl/strong --file_name=trial_201 --get_predictions=True --domain=salons --template_name=t5 --template_id=3 --instruction_id=27
    
    parser = argparse.ArgumentParser("Evaluate the finetuned model for identifying initiative apsects.")
    parser.add_argument("--model_dir", default="fine-tuning", required=False)
    parser.add_argument("--model_name", default="flan-t5-small", required=False)
    parser.add_argument("--file_name", default="trial_4", required=False)
    parser.add_argument("--domain", default='restaurant', required=False)
    parser.add_argument("--template_name", default=None, required=False)
    parser.add_argument("--template_id", default=None, required=False)
    parser.add_argument("--instruction_id", default=None, required=False)
    parser.add_argument("--get_predictions", default=False, required=False)

    # from dataloader import InitiativeExcelLoader
    # dataloader = InitiativeExcelLoader(directory='/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/v3/')
    # data_df = pd.DataFrame(dataloader.dataset['test'])
    # dataset = datasets.Dataset.from_pandas(data_df).shuffle(seed=42)
    args = parser.parse_args()

    if 'abstractive' in args.model_dir:
        score = Score(args.domain,abstractive=True)
    else:
        score = Score(args.domain)
    
    gpt_results = False
    if 'gpt' in args.model_dir:
        gpt_results = True
     
    filepath = f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/results/{args.model_dir}/'
    filename = f'{filepath}{args.file_name}.csv'
    data = pd.read_csv(filename)
    model_info = data['preds'][0]
    train_nc_list = []
    # model_info = ''
    # print(data.head(5))

    #ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/results/abstractive/fine-tuning/eos/v5/flan-t5-xl/strong_weak
    # if pred are not available 
    if args.get_predictions:
        labels_path = f'{filepath}{args.file_name}_all.csv'
        #labels_path = 'modular_approach/results/extractive/fine-tuning/eos/hotels/v1/flan-t5-xl/strong/trial_301_all.csv'
        labels_data = pd.read_csv(labels_path)
        labels = labels_data['labels'].unique().tolist()
        labels = [np.ravel(eval(item)).tolist() for item in labels[1:]]
        # #print(labels)
        #if score.domain == 'restaurant':
            #train_nc_list = [labels_data['train_nc_list'].unique().tolist()]
        
        reviews_path = f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/results/{args.model_dir}/shuffled_data_{args.file_name.split("_")[1]}.csv'
        reviews_data = pd.read_csv(reviews_path)
        reviews_data = reviews_data.fillna(str(''))
        if gpt_results:
            reviews = reviews_data['chatgpt_output'].tolist()
            gold = reviews_data['gold'].tolist()
        else:
            reviews = reviews_data['review'].tolist()
            # labels = reviews_data['true_strong_weak'].tolist()

        mode_path = f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/models/{args.model_dir}/{args.file_name}/'

        if args.instruction_id:
            score.load_formatter(template_name = args.template_name, template_id = int(args.template_id), 
                                instruction_id = int(args.instruction_id))
            reviews = score.format_data(reviews)

        if gpt_results:
            preds, labels = getPredictions(mode_path, reviews, labels=gold)
        else:
            preds, *_ = getPredictions(mode_path, reviews)
            # preds, labels = getPredictions(mode_path, reviews, labels=labels)

        print(preds[:5])

        if gpt_results:
            print(f'preds len:{len(preds)} labels len:{len(labels)}')
            data_dict = {'preds': preds, 'labels': labels}
        else:
            print(f'preds len:{len(preds)} labels len:{len(labels)} train_nc_list:{len(train_nc_list)}')
            if train_nc_list: data_dict = {'preds': preds, 'labels': labels, 'train_nc_list': train_nc_list[1:]}
            else : data_dict = {'preds': preds, 'labels': labels}
        
        df = pd.DataFrame(data_dict)
        print(df.head(5))
        df = df.fillna(str(''))
        df.to_excel(f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/results/{args.model_dir}/preds_labels_{args.file_name.split("_")[1]}.xlsx')
        # End
    
    data = pd.read_excel(f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/results/{args.model_dir}/preds_labels_{args.file_name.split("_")[1]}.xlsx')

    # If pred are available
    if not train_nc_list:
        df = data[['preds', 'labels']]
    else:
        df = data[['preds', 'labels', 'train_nc_list']]
    #
    # df = df[1:]
    # End
    
    df = df.apply(score.evaluate_row, axis=1)
    print(df.head(10))
    
    #self.output_df.index += 1
    if score.abstractive:
        columns = ['bleu', 'rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum', 'bertscore_avg_p', 'bertscore_avg_r', 'bertscore_avg_f1', 
                    'p_clusters', 'r_clusters', 'f1_clusters'
                   ]
    else:
        columns = [#'precision_bnp','recall_bnp','f1_bnp',
                            'precision_exactM', 'recall_exactM', 'f1_exactM', 
                            'precision_partialTokenized', 'recall_partialTokenized', 'f1_partialTokenized', 
                            'precision_partialWords', 'recall_partialWords', 'f1_partialWords',
                            'precision_clusters','recall_clusters','f1_clusters']
    
    df2 = df[columns].mean().round(2)
    print(df2.head(10))
    
    classification_result = score.evaluator.compute_clasification_metrics(preds=score.preds, labels=score.golds)
    print(classification_result)

    if score.abstractive:
        
        #print('preds:', score.preds)
        #print('labels:', score.golds)

        bleu_micro_avg = round(score.bleu.compute(predictions=score.preds, references=score.golds, max_order=score.max_order)['bleu'], 2)
        # rougeL_avg_scores = score.rouge.compute(predictions=score.preds, references=score.golds, rouge_types=['rougeL', 'rougeLsum'])
        # rouge_score = f"Micro avg rougeL:{round(rougeL_avg_scores['rougeL'], 2)}, rougeLsum:{round(rougeL_avg_scores['rougeLsum'], 2)}"
        
        rougeN_avg_scores = score.evaluator.compute_rouge_aggregated(predictions=score.preds, references=score.golds)
        
        new_row = pd.DataFrame({'classification': f'micro avg: {classification_result}',
                'bleu': f"macro avg: {df2['bleu']} \n micro avg: {bleu_micro_avg}", 
                'rouge1': f"macro avg: {df2['rouge1']} \n micro avg: {rougeN_avg_scores['rouge1']}", 
                'rouge2': f"macro avg: {df2['rouge2']} \n micro avg: {rougeN_avg_scores['rouge2']}",
                'rouge3': f"macro avg: {df2['rouge3']} \n micro avg: {rougeN_avg_scores['rouge3']}",
                'rouge4': f"macro avg: {df2['rouge4']} \n micro avg: {rougeN_avg_scores['rouge4']}",
                'rougeL': f"macro avg: {df2['rougeL']} \n micro avg: {round(rougeN_avg_scores['rougeL'], 2)}", 
                'rougeLsum': f"macro avg: {df2['rougeLsum']} \n micro avg: {round(rougeN_avg_scores['rougeLsum'], 2)}",
                'bertscore_avg_p': f"macro avg: {df2['bertscore_avg_p']} \n micro avg: {round(score.sum_p / score.num_of_examples, 2)}", 
                'bertscore_avg_r': f"macro avg: {df2['bertscore_avg_r']} \n micro avg: {round(score.sum_r / score.num_of_examples, 2)}",
                'bertscore_avg_f1': f"macro avg: {df2['bertscore_avg_f1']} \n micro avg: {round(score.sum_f / score.num_of_examples, 2)}",
                'p_clusters': df2['p_clusters'], 'r_clusters': df2['r_clusters'], 'f1_clusters': df2['f1_clusters'],
                'new_golds':'', 'preds': model_info,'labels': '','train_nc_list': ''}, index =[0])
    else:
        new_row = pd.DataFrame({'classification': f'micro avg: {classification_result}',
                #'precision_bnp': df2['precision_bnp'],'recall_bnp': df2['recall_bnp'], 'f1_bnp': df2['f1_bnp'],
                'precision_exactM': df2['precision_exactM'],'recall_exactM': df2['recall_exactM'], 'f1_exactM': df2['f1_exactM'],
                'precision_partialTokenized': df2['precision_partialTokenized'],'recall_partialTokenized': df2['recall_partialTokenized'],'f1_partialTokenized': df2['f1_partialTokenized'],
                'precision_partialWords': df2['precision_partialWords'],'recall_partialWords': df2['recall_partialWords'],'f1_partialWords': df2['f1_partialWords'],
                'precision_clusters': df2['precision_clusters'],'recall_clusters': df2['recall_clusters'],'f1_clusters': df2['f1_clusters'],
                'preds': model_info,
                'labels': '',
                'train_nc_list': ''}, index =[0])
    
    df = pd.concat([new_row, df]).reset_index(drop = True)
    filename = f'{filepath}{args.file_name}_u1.xlsx'
    df.to_excel(filename)
    

    
    

