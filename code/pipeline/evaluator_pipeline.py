from evaluator import Evaluator
import pandas as pd
from prompts import InstructionLoader
from IPython.display import display
import numpy as np

class ExperimentEvaluator():

    def __init__(self, filename, zero_shot=False):
        self.dir = '/modular_approach/results/fine-tuning/flan-t5-small'
        self.zero_shot = zero_shot
        self.few_shot_examples = 1
        self.ask_second_pass_question = False
        self.type_of_experiment = 'NC'
        self.folder = 'zero-shot/' if zero_shot else'few-shot/'
        self.filename = self.dir + self.folder + filename
        self.evaluator = Evaluator()
        self.instruction = InstructionLoader(model_type='instructGPT').get_instructions()[1]

    def get_preds(self):
        pass
    
    def evaluate(self):
        data = pd.read_excel(self.filename)
        #display(data.head())
        preds = data[self.instruction][2:].tolist()
        #preds = self.get_preds()
        #print(type(preds.tolist()))

        true_NC = data['true_NC'][2:].tolist()
        true_OC = data['true_OC'][2:].tolist()
        true_NC_OC = data['true_NC_OC'][2:].tolist()
        reviews = data['review'][2:].tolist()

        # metrics = f'NC Exact: {self.evaluator.calculate_metrics_by_list(preds, true_NC)} \nNC Partial: {self.evaluator.calculate_metrics_by_list_partial_match(preds, true_NC)}'
        # print(metrics)

        # metrics_NC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC, filter_bnps_flag=True)
        # #metrics_NC_OC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC_OC, filter_bnps_flag=True)
        # metrics = f'NC Exact: {metrics_NC} \nNC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC, filter_bnps_flag=True)}'
        #             #f'\nOC+NC Exact: {metrics_NC_OC} \nOC+NC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC_OC, filter_bnps_flag=True)}'

        # print(metrics)

        # metrics_NC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC)
        # #metrics_NC_OC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC_OC)
        # metrics = f'NC Exact: {metrics_NC} \nNC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC)}'
        #             #f'\nOC+NC Exact: {metrics_NC_OC} \nOC+NC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC_OC)}'
        # print(metrics)            
        metrics = ''
        if not self.zero_shot:
            metrics += f'Output 0 Testing {self.type_of_experiment} Number of Examples: {self.few_shot_examples}'\
                f'Ask second pass question: {self.ask_second_pass_question}'
        else:
            metrics += 'Output 0'
        metrics += '\n# Mathching list items separator = ", "'

        if self.zero_shot or not self.zero_shot and self.type_of_experiment == 'NC':
            metrics += f'\nNC Exact: {self.evaluator.calculate_metrics_by_list(preds, true_NC)} \nNC Partial: {self.evaluator.calculate_metrics_by_list_partial_match(preds, true_NC)}'
        if self.zero_shot or not self.zero_shot and self.type_of_experiment != 'NC':
            metrics += f'\nOC+NC Exact: {self.evaluator.calculate_metrics_by_list(preds, true_NC_OC)} \nOC+NC Partial: {self.evaluator.calculate_metrics_by_list_partial_match(preds, true_NC_OC)}'
        
        #print(metrics)
        metrics += '\n# Matching filtered BNPs'
        if self.zero_shot or not self.zero_shot and self.type_of_experiment == 'NC':
            metrics_NC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC, filter_bnps_flag=True)
            metrics += f'\nNC Exact: {metrics_NC} \nNC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC, filter_bnps_flag=True)}'
        if self.zero_shot or not self.zero_shot and self.type_of_experiment != 'NC':
            metrics_NC_OC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC_OC, filter_bnps_flag=True)
            metrics += f'\nOC+NC Exact: {metrics_NC_OC} \nOC+NC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC_OC, filter_bnps_flag=True)}'
        
        #print(metrics)
        metrics += '\n# Matching BNPs'
        if self.zero_shot or not self.zero_shot and self.type_of_experiment == 'NC':
            metrics_NC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC)
            metrics += f'\nNC Exact: {metrics_NC} \nNC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC)}'
        if self.zero_shot or not self.zero_shot and self.type_of_experiment != 'NC':
            metrics_NC_OC, output_bnps = self.evaluator.calculate_metrics_by_bnp(preds, true_NC_OC)
            metrics += f'\nOC+NC Exact: {metrics_NC_OC} \nOC+NC Partial: {self.evaluator.calculate_metrics_by_bnp_partial_match(preds, true_NC_OC)}\n'

        print(metrics)          
        ## Output preds to excel

        # instructions = []
        # instructions.append(self.instruction)
        # header = pd.MultiIndex.from_product([instructions,
        #                              ['Output', 'BNP']],
        #                             names=['Prompts','Review/Results'])
        # model_output = []
        # model_output.append(preds)
        # model_output.append(output_bnps)
        # data = np.array(model_output)
        # df = pd.DataFrame(data=data.T, columns=header)
        # df.insert(0, 'true_NC', true_NC)
        # df.insert(0, 'true_OC', true_OC)
        # df.insert(0, 'true_NC_OC', true_NC_OC)
        # df.insert(0, 'review', reviews)
        # df.to_excel(self.filename)


    