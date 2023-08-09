from prompts import PromptTemplate
class Formatter(object):
    """Abstract class for formatters"""
    """
    Abstract class for formatters
    :param template: The prompt template to be used for formatting language model prompts

    """
    def __init__(self, template:PromptTemplate):
        self.template = template

    def format(self, data):
        raise NotImplementedError

