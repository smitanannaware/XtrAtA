
class TestModel():
    
    def __init__(self, inputs, batch_size, model, tokenizer, device, max_source_length = 'longest',
                 max_new_tokens=500, padding='longest'):
        self.inputs = inputs
        self.batch_size = batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self.padding = padding
        self.predictions = []
        self.device = device
        
        
    def run(self):
        position = 0
        print(self.inputs[0])
        while position < len(self.inputs):
            # Evaluate on a batch of 10 reviews at a time.
            batch = self.inputs[position : position + self.batch_size]
            #print(batch)
            # Will pad the sequences up to the max length in the dataset.
            print("Zero/Few Shot Evaluation") #TODO change
            input_ids = self.tokenizer(batch, padding=self.padding, truncation=True, return_tensors="pt",).input_ids.to(self.device)
            
            outputs = self.model.generate(input_ids, self.max_new_tokens)
            self.predictions.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
            #print(predictions)
            position += self.batch_size
            print('Processed', position, 'reviews.')