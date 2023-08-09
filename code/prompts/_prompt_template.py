from ._t5_template import T5Template
from ._instructGPT_template import InstructGPTTemplate
import os 
class PromptTemplate:
    def __init__(self, prompt_type_or_str, template_id=0):
        self.prompt_type_or_str = prompt_type_or_str
        self.template_id = template_id

        
    def get_template(self):
        if self.prompt_type_or_str == 't5':
            path = os.path.dirname(os.path.abspath(__file__))
            return T5Template(self.template_id, path).get_template()
        elif self.prompt_type_or_str == 'instructGPT':
            path = os.path.dirname(os.path.abspath(__file__))
            return InstructGPTTemplate(self.template_id, path).get_template()
        elif self.prompt_type_or_str is not None:
            return self.prompt_type_or_str
        else:
            raise ValueError('Invalid prompt type')

    
