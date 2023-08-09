import pytest
from prompts import PromptTemplate
from formatter import DatasetFormatter
from dataloader import InitiativeExcelLoader
from prompts import InstructionLoader
# run using pytest -v modular_approach/formatter/tests/test_dataset_formatter.py
#TODO: When dataloader is completed.

def test_zero_shot_data_format():
    template = PromptTemplate('t5')
    data_loader = InitiativeExcelLoader()
    formatter = DatasetFormatter(template, data_loader)
    instructions = InstructionLoader().get_instructions()
    data = formatter.format(instruction=instructions[0])
    assert data['train'][0]['formatted_output'] == ('Given the following restaurant review:\n' + data['train'][0]['review'] + \
        '\n' + instructions[0].strip() + '\nNon-core aspects: ' + data['train'][0]['label'])
    assert data['test'][0]['formatted_output'] == ('Given the following restaurant review:\n' + data['test'][0]['review'] + \
        '\n' + instructions[0].strip() + '\nNon-core aspects: ')
    
    
def test_fewshot_data_format():
    template = PromptTemplate('t5')
    data_loader = InitiativeExcelLoader()
    formatter = DatasetFormatter(template, data_loader)
    instructions = InstructionLoader().get_instructions()
    data = formatter.formatFewShotExample(instruction=instructions[0], fewShotEx=2)
    # assert data['train'][0]['formatted_output'] == ('Given the following restaurant review:\n' + data['train'][0]['review'] + \
    #     '\n' + instructions[0].strip() + '\nNon-core aspects: ' + data['train'][0]['label'])
    # assert data['test'][0]['formatted_output'] == ('Given the following restaurant review:\n' + data['test'][0]['review'] + \
    #     '\n' + instructions[0].strip() + '\nNon-core aspects: ')
    print(data['test'][1]['formatted_output'])