from dataloader import InitiativeExcelLoader
from prompts import InstructionLoader, DomainQLoader
from model_experiments import RunInstructGPT
from evaluator import Evaluator
import pandas as pd
import numpy as np
from utils.constants import Models
from prompts import PromptTemplate
from formatter import DatasetFormatter
from extractor import BNPExtractor

class APIPipeline():

    def __init__(self, model) -> None:
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_name = model
        self.zero_shot = True
        self.data_loader = InitiativeExcelLoader()
        
        self.template = model
        self.template_id = 0
        self.few_shot_examples = 2
        self.batch_size = 10
        self.output_dir = ''
        self._init_classes()

    def _init_classes(self):
        self.instructions = InstructionLoader(model_type=self.model).get_instructions()
        self.domainQs = DomainQLoader(model_type=self.model).get_domainQs()
        self.template_obj = PromptTemplate(self.template, self.template_id)
        self.formatter = DatasetFormatter(self.template_obj, self.data_loader)
        self.evaluator = Evaluator()
        self.experiment = RunInstructGPT()


    def get_input_data(self, instruction, domainQ, domainA):
        if self.zero_shot:
            data = self.formatter.format(instruction=instruction, domainQ=domainQ, domainA = domainA)
        else:
            data = self.formatter.formatFewShotExample(instruction=instruction, fewShotEx = self.few_shot_examples)

        data['test'] = BNPExtractor().extract(dataset=data['test'])
        #print(data['test'])
        test_data = pd.DataFrame.from_dict(data['test'])
        #print('*****')
        #print(test_data['formatted_output'])
        return test_data

    def output_to_file(self, metrics, model_output, examples):
        header = pd.MultiIndex.from_product([self.instructions,
                                     ['Output','BNP']],
                                    names=['Prompts','Review/Results'])

        data = np.array(model_output)
        df = pd.DataFrame(data=data.T, columns=header)
        df.insert(0, 'true_labels', examples['label'])
        df.insert(0, 'review', examples['review'])
        df.to_csv(self.output_dir+f'output_results_{self.model_name}.csv')

        self.instructions.insert(0, 'Format')
        metrics.insert(0, examples['formatted_output'][0])
        pd.DataFrame({'prompt': self.instructions, 'metrics': metrics}).to_csv(self.output_dir+f'metrics_results_{self.model_name}.csv')


    def run(self):
        results = []
        model_output = []
        domainQ = self.domainQs[0]
        domainA = self.experiment.getDomainKnowledge(domainQ)
        
        for instruction in self.instructions:
            test_data = self.get_input_data(instruction, domainQ, domainA)
            #print(test_data['formatted_output'])
            self.experiment.run(test_data['formatted_output'].values.tolist()[:2])
            preds = self.experiment.predictions
            print(self.experiment.predictions)
            print(test_data['label'].values.tolist())
            result, output_bnps = self.evaluator.calculate_metrics_1(preds=preds, labels=test_data['label'].values.tolist()[:2])
            results.append(result)
            model_output.append(self.experiment.predictions)
            model_output.append(output_bnps)
            break # TODO remove
        self.output_to_file(results, model_output, test_data)