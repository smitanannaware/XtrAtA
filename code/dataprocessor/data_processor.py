
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample
from dataloader import InitiativeExcelLoader
from datasets import Dataset
import pandas as pd
class CustomDataProcessor(DataProcessor):
    
    def __init__(self):
        super().__init__()
    
    '''
    Dataset is in dictionary records format
    '''
    def get_examples(self, dataset, split):
        dataset = Dataset.from_pandas(pd.DataFrame(InitiativeExcelLoader().dataset[split]))
        return list(map(self.transform, dataset))


    def transform(self, example):
        #print(example)
        text_a = example['review']
        tgt_txt = example['true_nc']
        return InputExample(text_a = text_a, tgt_text=tgt_txt)
 

        