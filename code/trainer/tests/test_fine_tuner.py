import pytest
from dataloader.dataset_loader import InitiativeExcelLoader
from trainer import FineTuner
# run using pytest -v modular_approach/trainer/tests/test_fine_tuner.py


def test_dataset_normal_split():
        data_loader = InitiativeExcelLoader()
        fine_tuner = FineTuner('test', data_loader)
        train, test = next(fine_tuner.load_dataset())
        #print(next(datasets))
        assert train.num_rows == 69
        assert test.num_rows == 8
        

def test_dataset_kfold():
        data_loader = InitiativeExcelLoader()
        fine_tuner = FineTuner('test', data_loader)
        dataset_generator = fine_tuner.load_dataset(kfold=True)
        for train, test in dataset_generator:
            # print(train)
            # print(test)
            print(train[0]['label'])
            print(train[-1]['label'])
            print('*****')
            print(test[0]['label'])
            print(test[-1]['label'])
            #print(next(dataset_generator))
        #assert train.num_rows == 69
        #assert test.num_rows == 8
    