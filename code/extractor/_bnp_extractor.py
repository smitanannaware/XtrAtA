from ._extractor import Extractor
class BNPExtractor(Extractor):
    
    def __init__(self):
        super().__init__()
        #nlp = 

        
    def extract_from_text(self, text):
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    
    def extract_sentences(self, text):
        doc = self.nlp(text)
        return list(doc.sents)
    

    def extract(self, dataset, return_field_name = 'base_noun_phrases'):
        bnp_list = []
        for data_dict in dataset:
            if return_field_name:
                data_dict[return_field_name] = self.extract_from_text(text = data_dict['review'])
                #print(dataset)
            else:
                bnp_list.append(self.extract(data_dict['review']))
                
        if return_field_name:
            return dataset
        else:
            return bnp_list    