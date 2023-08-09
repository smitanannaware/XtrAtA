import os 
class InstructionLoader:
    def __init__(self, model_type = None):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.model_type = model_type
        
    def get_instructions(self):
        if self.model_type == 'instructGPT':
            with open(os.path.join(self.path , 'instructGPT_instructions.txt')) as f:
                return f.read().splitlines()
        with open(os.path.join(self.path , 'instructions.txt')) as f:
            return f.read().splitlines()
