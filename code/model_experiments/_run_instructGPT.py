import openai

openai.organization = "org-wl3zexGh6Yzm0MvhR0wEL9xG"
# API key = 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
openai.api_key = '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'

class RunInstructGPT():
    
    def __init__(self, model='text-davinci-003', temp=0.6, max_len=100,
        top_p = 0.85, freq_penalty = 0.1, presence_penalty = 0, best_of = 2):
        
        self.model = model
        self.temp = temp
        self.max_len = max_len
        self.top_p = top_p
        self.freq_penalty = freq_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.predictions = []
        
        
            
    def getDomainKnowledge(self, domainQ):
        
        response = openai.Completion.create(
                        model = self.model, 
                        prompt = domainQ, 
                        temperature = self.temp, 
                        max_tokens = self.max_len,
                        top_p = self.top_p,
                        frequency_penalty = self.freq_penalty,
                        presence_penalty = self.presence_penalty,
                        best_of = self.best_of
                        )

        return response.choices[0].text

        
    def run(self, inputs):
        print(inputs[0])
        for i, input in enumerate(inputs):
            response = openai.Completion.create(
                        model = self.model, 
                        prompt = input, 
                        temperature = self.temp, 
                        max_tokens = self.max_len,
                        top_p = self.top_p,
                        frequency_penalty = self.freq_penalty,
                        presence_penalty = self.presence_penalty,
                        best_of = self.best_of
                        )

            response.choices[0].text
            self.predictions.extend(response.choices[0].text)
            #print(predictions)
            if i % 10 == 0 : print('Processed', i, 'reviews.')

    
