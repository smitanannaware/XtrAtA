import pandas as pd
import os 
import tqdm
import pickle
import numpy as np
#from IPython.display import display
"""

"""
#TODO: document and add absolute path
class InitiativeExcelLoader:
    def __init__(self, directory = "/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/restaurants/v6/", datatypes = ['train', 'dev', 'test']):
        self.directory = directory
        self.datatypes = datatypes
        self.dataset = {}
        self.load_dataset()
        
           
    def load_dataset(self):
        
        for data_type in tqdm.tqdm(self.datatypes, desc="Initiative dataset loading progress", position=0):
            dir = os.path.join(self.directory,data_type)
            files = os.listdir(dir)
            for file in files:
                file_path = os.path.join(dir,file)
                if file.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                    #for col in data.columns:
                        #data[col] = data[col].str.strip()
                    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    
                    data = data.fillna(value='')
                    self.dataset[data_type] = data.to_dict('records')
    
    
    def save_dataset(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    
    def load_dataset_from_saved(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
                    

if __name__ == "__main__":
    loader = InitiativeExcelLoader()
    print(loader.dataset['train'])