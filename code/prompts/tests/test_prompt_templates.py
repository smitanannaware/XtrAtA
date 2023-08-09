import pytest
from prompts import PromptTemplate
from prompts._t5_template import T5Template

# convert to Pytest
# run using pytest -v modular_approach/prompts/tests/test_prompt_templates.py
def test_get_template():
    prompt_template = PromptTemplate('t5', 0)
    assert prompt_template.get_template() == 'Given the following restaurant review:\n{review}{prompt}\nNon-core aspects:{label}'

def test_get_template_with_index():
    prompt_template = PromptTemplate('t5', 1)
    assert prompt_template.get_template() == 'question: Given the following restaurant review, {prompt} context: {review}'