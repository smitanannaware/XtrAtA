"""
Abstract class for Extractors

"""
import spacy
from spacy.lang.en import English

class Extractor:
    def __init__(self):
        self.extractor = None
        self.nlp = nlp = spacy.load('en_core_web_sm')

    def extract(self, text):
        return NotImplementedError

    def tokenize(self, text):
        return self.nlp(text)