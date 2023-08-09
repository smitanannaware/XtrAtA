import torch

class Inference:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_beams=kwargs.get("num_beams", 5),
            num_return_sequences=kwargs.get("num_return_sequences", 5),
            temperature=kwargs.get("temperature", 1.0),
            early_stopping=kwargs.get("early_stopping", True),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 2),
            repetition_penalty=kwargs.get("repetition_penalty", 2.5),
            length_penalty=kwargs.get("length_penalty", 1.0),
            do_sample=kwargs.get("do_sample", True),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.95),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bad_words_ids=[[self.tokenizer.eos_token_id]],
        )

        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return text

    
    """
    This function is used to classify the input text into one of the labels in the dataset.
    It returns a dictionary with the labels as keys and the probabilities as values.
    :param input_ids: input_ids of the input text
    :param attention_mask: attention_mask of the input text
    :param return_probs: if True, it returns the probabilities as well
    :return: dictionary with labels as keys and probabilities as values
    """
    def classify(self, input_ids, attention_mask, return_probs=False):
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            probs = probs.cpu().numpy()
            # decode into labels
            labels = self.model.config.id2label
            labels = [labels[i] for i in range(len(labels))]
            probs = probs.tolist()[0]
            labels = dict(zip(labels, probs))
            # return labels
            if return_probs:
                return labels, probs
            else:
                return labels



    def _generate_next_token(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_beams=kwargs.get("num_beams", 5),
            num_return_sequences=kwargs.get("num_return_sequences", 5),
            temperature=kwargs.get("temperature", 1.0),
            early_stopping=kwargs.get("early_stopping", True),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 2),
            repetition_penalty=kwargs.get("repetition_penalty", 2.5),
            length_penalty=kwargs.get("length_penalty", 1.0),
            do_sample=kwargs.get("do_sample", True),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.95),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bad_words_ids=[[self.tokenizer.eos_token_id]],  
        )

    """
    Generate text with constraints.
    """
    def constrained_generate(self, input_ids, attention_mask, constraint_func, **kwargs):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_beams=kwargs.get("num_beams", 5),
            num_return_sequences=kwargs.get("num_return_sequences", 5),
            temperature=kwargs.get("temperature", 1.0),
            early_stopping=kwargs.get("early_stopping", True),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 2),
            repetition_penalty=kwargs.get("repetition_penalty", 2.5),
            length_penalty=kwargs.get("length_penalty", 1.0),
            do_sample=kwargs.get("do_sample", True),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.95),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bad_words_ids=[[self.tokenizer.eos_token_id]],
            forced_bos_token_id=self.tokenizer.bos_token_id,
            forced_eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return text



    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_max_length(self):
        return self.max_length