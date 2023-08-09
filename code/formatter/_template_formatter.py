from prompts import PromptTemplate
from ._formatter import Formatter
class TemplateFormatter(Formatter):
    """
    A formatter that uses a template class to format a language model prompt.
    Example:

    >>> template = PromptTemplate('This is a {template} prompt.')
    >>> formatter = TemplateFormatter(template)
    >>> formatter.format({'template': 'test'})
    'This is a test prompt.'
    """

    def __init__(self, template:PromptTemplate):
        super().__init__(template)
        self.template = template.get_template()
        self.validate_template()
        
    """
    Validate the template.
    """
    def validate_template(self):
        if '{' in self.template and '}' in self.template:   
            pass
        elif '[[' in self.template and ']]' in self.template:
            self.template = self.template.replace('[[', '{').replace(']]', '}')
        elif '<' in self.template and '>' in self.template:
            self.template = self.template.replace('<', '{').replace('>', '}')
        elif ('{' in self.template and '}' not in self.template) or ('{' not in self.template and '}' in self.template):
            raise ValueError("The template has a { or } but not both.")  
        elif ('[' in self.template and ']' in self.template) or ('[' not in self.template and ']' in self.template):
            raise ValueError("The template has a [ or ] but not both.")
        elif ('<' in self.template and '>' in self.template) or ('<' not in self.template and '>' in self.template):
            raise ValueError("The template has a < or > but not both.")
        else:
            raise ValueError("The template does not have a { or }.")

            
    """
    Format a language model prompt using a template.
    :param data_dict: A dictionary of template variables and their values.
    :return: A formatted language model prompt.
    """
    def format(self, data_dict): 
        return self.template.format(**data_dict)
    
    """
    Format a language model prompt using a template.
    :param data_dict: A dictionary of template variables and their values.
    :param hide_label: remove labels in formatting in case of text examples
    :return: A formatted language model prompt.
    """
    def format(self, data_dict, hide_label): 
        if hide_label:
            return self.template.replace('{label}', '').format(**data_dict)
        return self.template.format(**data_dict)


            

        