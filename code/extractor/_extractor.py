"""
Abstract class for Extractors

"""
import spacy
from spacy.lang.en import English

class Extractor:
    def __init__(self):
        self.extractor = None
        self.nlp = nlp = spacy.load(r'/users/snannawa/.conda/envs/sn_torch/lib/python3.10/site-packages/en_core_web_sm/en_core_web_sm-3.3.0')

    def extract(self, text):
        return NotImplementedError

    def tokenize(self, text):
        return self.nlp(text)