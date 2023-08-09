from utils.trainerHelper import get_nc_clusters
from evaluator import Evaluator
import pandas as pd
import numpy as np
import argparse
import re
import evaluate

class Score():

    def __init__(self, abstractive = False):
        self.evaluator = Evaluator()
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.abstractive = abstractive
        self.sum_p = 0
        self.sum_r = 0
        self.sum_f = 0
        self.num_of_examples = 0
        self.bnp_to_sent_mapping = pd.read_excel('/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/v4/extractive_to_abstractive_mapping.xlsx')
        self.gold_to_sentences = pd.read_excel('/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/v4/sent separated.xlsx')
        #self.nc_cluster_map = get_nc_clusters()
        self.preds = []
        self.golds = []
        self.max_order = 1
 
def pre_process_data(text):
    text = text.replace('\n',' ')
    text = text.strip()
    
    return text


if __name__ == "__main__":

    # python modular_approach/evaluate_zero_few_shot_results.py --model_dir=extractive/few-shot/eos/restaurants/v5 --model_name=GPT --file_name=ex_all_core_01235_5shot_strong_weak --ref_column_name=true_strong_weak --data_dir=v5
    # python modular_approach/evaluate_zero_few_shot_results.py --model_dir=abstractive/few-shot/eos/restaurants/v5 --model_name=GPT --file_name=abs_all_core_01235_5shot_strong_weak --ref_column_name=abs_true_strong_weak_alt --data_dir=v5

    # python modular_approach/evaluate_zero_few_shot_results.py --model_dir=ITA --model_name=erfan --file_name=ITA_dev_erfan --ref_column_name=abs_true_strong --data_dir=salons/v1
    parser = argparse.ArgumentParser("Evaluate the finetuned model for identifying initiative apsects.")
    parser.add_argument("--model_dir", default="fine-tuning", required=False)
    parser.add_argument("--model_name", default="flan-t5-small", required=False)
    parser.add_argument("--file_name", default="trial_4", required=False)
    parser.add_argument("--ref_column_name", default="true_strong", required=False)
    parser.add_argument("--data_dir", default="v5", required=False)
    parser.add_argument("--abstractive", default=False, required=False)
    
    
    # from dataloader import InitiativeExcelLoader
    # dataloader = InitiativeExcelLoader(directory='/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/v3/')
    # data_df = pd.DataFrame(dataloader.dataset['test'])
    # dataset = datasets.Dataset.from_pandas(data_df).shuffle(seed=42)
    args = parser.parse_args()

    if 'abstractive' in args.model_dir or args.abstractive:
        score = Score(abstractive=True)
    else:
        score = Score()
    
    # filepath = f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/{args.data_dir}/test/'
    # filename = f'{filepath}test.xlsx'
    filepath = f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/{args.data_dir}/dev/'
    filename = f'{filepath}dev.xlsx'

    eval_data = pd.read_excel(filename)
    #print(data.head(5))
    
    filepath = f'/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/results/{args.model_dir}/{args.model_name}/'
    filename = f'{filepath}{args.file_name}.xlsx'
    data = pd.read_excel(filename)
    print(data.head(5))
    
    data = data.fillna(str(''))
    eval_data = eval_data.fillna(str(''))
    
    data[[args.ref_column_name]] = data[[args.ref_column_name]].applymap(lambda x:  pre_process_data(x) if isinstance(x, str) else x)
    eval_data[['true_strong', 'true_strong_weak', 'abs_true_strong', 'abs_true_strong_weak']] = eval_data[['true_strong', 'true_strong_weak', 'abs_true_strong', 'abs_true_strong_weak']].applymap(lambda x:  pre_process_data(x) if isinstance(x, str) else x)
    #preds = data['pred'].values.tolist()[1:]
    preds = data[args.ref_column_name].values.tolist()
    # golds = data[args.ref_column_name].values.tolist()[1:]

    golds = eval_data[args.ref_column_name].values.tolist()
 
    classification_result = score.evaluator.compute_clasification_metrics(preds=preds, labels=golds)
    print(classification_result)

    if score.abstractive:
        
        print('preds:', preds[:5])
        print('labels:', golds[:5])

        bleu_micro_avg = round(score.bleu.compute(predictions=preds, references=golds, max_order=score.max_order)['bleu'], 2)
         
        rougeN_avg_scores = score.evaluator.compute_rouge_aggregated(predictions=preds, references=golds)

        bert_scores = score.evaluator.compute_bertscore(predictions=preds, references=golds)

        metrics = f"classification: micro avg: {classification_result}, bleu: micro avg: {bleu_micro_avg}, rouge1: micro avg: {rougeN_avg_scores['rouge1']}, rouge2: micro avg: {rougeN_avg_scores['rouge2']}, rouge3: micro avg: {rougeN_avg_scores['rouge3']}, rouge4: micro avg: {rougeN_avg_scores['rouge4']}, rougeL: micro avg: {round(rougeN_avg_scores['rougeL'], 2)}, rougeLsum: micro avg: {round(rougeN_avg_scores['rougeLsum'], 2)}, bertscore_avg_p: micro avg: {round(np.mean(bert_scores['precision']), 2)}, bertscore_avg_r: micro avg: {round(np.mean(bert_scores['recall']), 2)}, bertscore_avg_f1: micro avg: {round(np.mean(bert_scores['f1']), 2)}"

        new_row = pd.DataFrame({'review': '', 'abs_true_strong':'', 'abs_true_strong_weak': '', 
                                'pred': f"classification: micro avg: {classification_result}, bleu: micro avg: {bleu_micro_avg}, rouge1: micro avg: {rougeN_avg_scores['rouge1']}, rouge2: micro avg: {rougeN_avg_scores['rouge2']}, rouge3: micro avg: {rougeN_avg_scores['rouge3']}, rouge4: micro avg: {rougeN_avg_scores['rouge4']}, rougeL: micro avg: {round(rougeN_avg_scores['rougeL'], 2)}, rougeLsum: micro avg: {round(rougeN_avg_scores['rougeLsum'], 2)}, bertscore_avg_p: micro avg: {round(np.mean(bert_scores['precision']), 2)}, bertscore_avg_r: micro avg: {round(np.mean(bert_scores['recall']), 2)}, bertscore_avg_f1: micro avg: {round(np.mean(bert_scores['f1']), 2)}"}, index =[0])
    else:
        result_0 = score.evaluator.calculate_metrics_exact_match(preds=preds, labels=golds)
        # Evaluate by partial match with tokenization
        result_1 = score.evaluator.calculate_metrics_exact_match_with_partial(preds=preds, labels=golds)
        # Evaluate by partial match with space as separator
        result_2 = score.evaluator.calculate_metrics_exact_match_with_partial(preds=preds, labels=golds, tokenize=False)

        metrics = f"classification: micro avg: {classification_result}, exact: {result_0}, partial_tokenized: {result_1}, partial_bywords: {result_2}"
        new_row = pd.DataFrame({'review': '', 'abs_true_strong':'', 'abs_true_strong_weak': '', 
                                'pred': f"classification: micro avg: {classification_result}, exact: {result_0}, partial_tokenized: {result_1}, partial_bywords: {result_2}"}, index =[0])
    
    data = pd.concat([new_row, data]).reset_index(drop = True)
    #data[args.ref_column_name][0] = metrics
    if 'weak' in args.ref_column_name:
        filename = f'{filepath}{args.file_name}_oc_ex.xlsx'
    else:
        filename = f'{filepath}{args.file_name}_nc_ex.xlsx'
    data.to_excel(filename)
    

    
    

