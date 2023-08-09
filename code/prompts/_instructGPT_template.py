import os
class InstructGPTTemplate:
    
    def __init__(self, template_id, path='.'):
        self.template_id = template_id
        self.path = path
        
    def get_template(self):
        with open(os.path.join(self.path , 'instructGPT_prompts.txt')) as f:
            return rf'{f.readlines()[self.template_id]}'.replace("\\n", "\n").strip()