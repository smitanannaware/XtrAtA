import pandas as pd
import os 
import tqdm
import pickle

"""

"""
#TODO: document and add absolute path
class InitiativeExcelLoader:
    def __init__(self, directory = "modular_approach/dataset/", datatypes = ['train','test']):
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
                    data['review'] = data['review'].str.strip()
                    data['label'] = data['label'].str.replace(',', ', ')
                    data = data.fillna(value='NONE')
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