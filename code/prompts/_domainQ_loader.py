import os 
class DomainQLoader:
    def __init__(self, model_type):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.model_type = model_type
        
    def get_domainQs(self):
        if self.model_type == 'instructGPT':
            with open(os.path.join(self.path , 'instructGPT_domainQ.txt')) as f:
                return f.read().splitlines()
        return None
