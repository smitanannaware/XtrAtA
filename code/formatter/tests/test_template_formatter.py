# unit tests for TemplateFormatter
import pytest
from prompts import PromptTemplate
from formatter import TemplateFormatter
# run using pytest -v modular_approach/formatter/tests/test_template_formatter.py
def test_format():
    template = PromptTemplate('This is a {template} prompt.')
    formatter = TemplateFormatter(template)
    assert formatter.format({'template': 'test'}) == 'This is a test prompt.'

def test_format_with_multiple_variables():
    template = PromptTemplate('This is a {template} prompt. {INTRODUCTION} {BADEED}')
    formatter = TemplateFormatter(template)
    assert formatter.format({'template': 'test', 'BADEED': 'WEED', 'INTRODUCTION': 'I AM BADEED AND I NEED'}) == 'This is a test prompt. I AM BADEED AND I NEED WEED'

def test_format_with_model():
    template = PromptTemplate('t5', 1)
    formatter = TemplateFormatter(template)
    assert formatter.format({'prompt': 'test prompt', 'review': 'This is test review'}) == 'question: Given the following restaurant review, test prompt context: This is test review'


def test_format_missing_brackets():
    template = PromptTemplate('This is a [template prompt.')
    with pytest.raises(ValueError):
        formatter = TemplateFormatter(template)
        formatter.format({'template': 'test'})

def test_format_missing_brackets2():
    template = PromptTemplate('This is a template] prompt.')
    with pytest.raises(ValueError):
        formatter = TemplateFormatter(template)
        formatter.format({'template': 'test'})

def test_format_missing_brackets3():
    template = PromptTemplate('This is a <template prompt.')
    with pytest.raises(ValueError):
        formatter = TemplateFormatter(template)
        formatter.format({'template': 'test'})


def test_format_missing_brackets4():
    template = PromptTemplate('This is a template> prompt.')
    with pytest.raises(ValueError):
        formatter = TemplateFormatter(template)
        formatter.format({'template': 'test'})


def test_format_missing_brackets5():
    template = PromptTemplate('This is a {template prompt.')
    with pytest.raises(ValueError):
        formatter = TemplateFormatter(template)
        formatter.format({'template': 'test'})