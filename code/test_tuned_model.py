from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorWithPadding, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    HfArgumentParser
)
import argparse

from utils.HuggingFacePathHandler import get_huggingface_path
import torch
from peft import PeftModel, PeftConfig

import random, itertools

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    
    args.add_argument("--model_name", type=str, default="modular_approach/models/abstractive/fine-tuning/eos/v4/flan-t5-xl/trial_603/out_fold7/")
    args.add_argument('--use_peft', type=bool, default=False)
    my_args = args.parse_args()

    peft_model_id = my_args.model_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft = my_args.use_peft
    if peft:
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model_id).to(device)
    else:
        # try:
        #     model = AutoModelForSeq2SeqLM.from_pretrained(peft_model_id).to(device)
        # except:
        model = AutoModelForSeq2SeqLM.from_pretrained(get_huggingface_path(peft_model_id)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(get_huggingface_path(peft_model_id))
            
        # tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    print('Model load complete!!!')
    #prompt = "question: Given the following restaurant review, list what is atypical in the restaurant. context:"
    #prompt = 'question: Given the following restaurant review, list what are the atypical aspects for a restaurant. context:'
    # prompt = 'question: Based on the following restaurant review, what are the atypical aspects for a restaurant? context:'
    # prompt = 'Based on the following restaurant review, what are the atypical aspects for a restaurant? Extract phrases and separate them using commas.'
    # prompt = 'List aspects mentioned in the following restaurant review that are atypical for a restaurant. Output phrases and separate them using commas.'
    # prompt = 'Given the following restaurant review, can you list atypical aspects for a restaurant? Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Please be precise in your response and it should contain only atypical aspects that are associated with the restaurant that is reviewed. Output in phrases and separate aspects using commas. Output <None> if there are no atypical aspects.'
    # prompt = 'Given the following restaurant review, can you list atypical aspects for a restaurant? Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Generate such noun phrases and separate them using commas.'
    # prompt = 'Given the following restaurant review, generate atypical aspects for a restaurant in the form of phrases? Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Separate aspects using commas.'
    # prompt = 'What aspects mentioned in the review are unexpected for a restaurant? List all of them.'
    # prompt = 'List aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Generate all of them using commas.' # good - bottomless coffee, free wifi, ping pong, shuffleboard
    # prompt = 'List multiple aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Generate phrases instead of sentences and separate them using commas.'
    # prompt = 'What aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Generate list of such sentences.'
    # prompt = 'Identify atypical aspects from the given restaurant review. Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant.'
    # prompt = 'Identify aspects mentioned in the review are unexpected for a restaurant. Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Generate all of them.'
    # prompt = 'Given the following restaurant review, which aspects in the review are atypical for restaurants? Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all of them using commas.'
    # prompt = 'Which aspects in the review are atypical for a restaurant? Atypical aspects are not related to service, food, drinks, or other items commonly associated with a restaurant. List all of them using commas. Generate empty list if there are none.'
    # prompt = 'Given the following restaurant review, which aspects mentioned in the review are atypical for a restaurant? atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all atypical aspects in the form of base noun phrases using comma or write no response if there are no atypical aspects.'
    # prompt = 'Given the following restaurant review, which aspects mentioned in the review are atypical for a restaurant? atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all atypical aspects in base noun phrases form using commas or write "no response" if there are no atypical aspects.'

    # # Extractive With all core
    # prompt = 'Given the following restaurant review, which aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service, food, drinks, parking, location, food items, prices, discounts, policies, customer satisfaction or other types of items that are commonly associated with a restaurant. List all of them using commas or write no response if there are no unexpected aspects.'

    # prompt = 'Given the following restaurant review, which aspects mentioned in the review are atypical for a restaurant? atypical aspects are not related to service, food, drinks, parking, location, food items, prices, discounts, policies, customer satisfaction or or other types of items that are commonly associated with a restaurant. List base noun phrases of all atypical aspects using commas or write no response if there are no atypical aspects.'

    

    # ABSTRACTIVE
    # prompt = 'Given the following restaurant review, which aspects mentioned in the review are atypical for a restaurant? Atypical aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all of them using sentences or write no response if there are no atypical aspects.'
    # prompt = 'Given the following restaurant review, which aspects in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Summarize all of them in sentence format or write no response if there are no unexpected aspects.'
    # prompt = 'Given the following restaurant review, which aspects in the review are unexpected for a restaurant? Unexpected aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Generate a summary of all of them in a paragraph or write no response if there are no unexpected aspects.'
    # # prompt = 'Given the following restaurant review, write a summary of unexpected aspects of the restaurant. Unexpected aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Write no response if there are no unexpected aspects.'
    # prompt = 'Given the following restaurant review, which aspects in the review are unexpected for a restaurant? Unexpected aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Write no response if there are no unexpected aspects.'
    # prompt = 'Given the following restaurant review, generate a summary of all aspects in the review are unexpected for a restaurant? Unexpected aspects are separate from service, food, drinks, or other items commonly associated with a restaurant.You can be creative. Write a paragraph of sentences or write no response if there are no unexpected aspects.'
    # prompt = 'What are the atypical aspects for a restaurant based on the given review? Atypical aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Pick the sentences with atypical aspects and rephrase each sentences, for example, if A is an atypical aspect then output "The restaurat has A.". If there are no atypical aspects output no response.' # worked


    # prompt = 'What are the atypical aspects for a restaurant based on the given review? Atypical aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Pick the sentences with atypical aspects and rephrase each sentence, for example, if A is an atypical aspect then output "The restaurat has A.". If there are no atypical aspects output "There are no atypical aspects.".'

    # Use this for testing with question/context
    # prompt = 'What are the atypical aspects for a restaurant based on the given review? Atypical aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Pick the sentences with atypical aspects and rephrase each; for example, if A is an atypical aspect, then output "The restaurant has A.". If there are no atypical aspects, output "no response".' 

    # prompt = 'What are the atypical aspects of a restaurant based on the given review? Atypical aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Pick the sentences with atypical aspects and rephrase each. For example, if A is an atypical aspect, then output "- The restaurant has A.". If there are multiple aspects such as A, B, and C, then the output could be "- The restaurant has A.\n- They have B.\n- They have C.". If there are no atypical aspects, output "- There are no atypical aspects.".'

    # prompt = 'What are the atypical aspects for a restaurant based on the given review? Atypical aspects are separate from service, food, drinks, or other items commonly associated with a restaurant. Rephrase each aspect; for example, if there is an atypical aspect called A, then output could be "The restaurant has A." or "The restaurant offers A" and if there are no atypical aspects, output "no response".'

    # prompt = "What are the aspects that are mentioned in the review that are atypical for a restaurant? Unlike common aspects such as service, food, or drinks, atypical aspects are not commonly associated with a restaurant. In the output, formulate each aspect A as 'The restaurant has A.' or if there are no atypical aspects, output 'no response'."
    # prompt = 'What are the aspects that are mentioned in the review that are atypical for a restaurant? Unlike common aspects such as service, food, or drinks, atypical aspects are not commonly associated with a restaurant. In the output, formulate each aspect A as "The restaurant has A.". If there are no atypical aspects, output "no response".'
    # prompt = 'What are the aspects that are mentioned in the review that are atypical for a restaurant? Unlike common aspects such as service, food, or drinks, atypical aspects are not commonly associated with a restaurant. In the output, formulate each aspect A as "The restaurant has A.". If there are no atypical aspects, output "There are no atypical aspects.".'
    # prompt = 'What are the atypical aspects for a restaurant based on the given review? Atypical aspects are not related to service, food, drinks, or other items commonly associated with a restaurant. In the output, formulate each aspect A as "The restaurant has A.". If there are no atypical aspects, output "There are no atypical aspects.".'

    # # mention all core aspects
    # prompt = 'What are the aspects that are mentioned in the review that are atypical for a restaurant? Unlike common aspects such as service, food, drinks, parking, location, food items, prices, discounts, policies, or customer satisfaction, atypical aspects are not commonly associated with a restaurant. In the output, formulate each aspect A as "The restaurant has A.". If there are no atypical aspects, output "There are no atypical aspects.".'

    # FEW SHOT 
    # Extractive
    prompt = "question: Given a restaurant review, which aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all of them using commas or write no response if there are no unexpected aspects."

    # prompt = "question: Given a restaurant review, which aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all of them using commas."
    # prompt = 'question: List aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. Generate all base noun phrases using commas.'
    prompt = "Q: Given the following restaurant review, which aspects mentioned in the review are unexpected for a restaurant? Unexpected aspects are not related to service or food or drinks or other types of items that are commonly associated with a restaurant. List all base noun phrases using commas or write no response if there are no unexpected aspects."

    # prompt = ""

    import pandas as pd
    filePath = '/projects/rbunescu_research/erfan_smita_space/ATICA/dialogue_system/restaurant_reviews_initiative/modular_approach/dataset/v5/'
    dev_set = pd.read_excel(filePath+'dev/dev.xlsx')
    eval_set = pd.read_excel(filePath+'test/eval.xlsx')
    
    dev_set = dev_set.fillna(str('no response'))
    dev_set = dev_set.replace('\n',str(' '))
    #dev_set = dev_set.replace('',str(''))
    reviews = dev_set['review'].values.tolist()
    golds = dev_set['true_strong'].values.tolist()

    test_ex = eval_set['review'].values.tolist()[:10]

    max_combination_len = 2
    number_of_items = [1, 2, 6, 0, 8, 10, 11, 12, 13, 14, 18, 19]
    list_of_combinations = []
    for i in range(2, max_combination_len+1):
        perm = list(itertools.permutations(number_of_items, i))
        #random.shuffle(perm)
        #list_of_combinations.extend([list(item)for item in perm])
        list_of_combinations.extend(set(frozenset(item)for item in perm))

    #random.shuffle(list_of_combinations)
    #list_of_combinations
    
    print('Prompt: ', prompt)
    output_dict = {}
            
    for indexes in list_of_combinations:
        #indexes = [1, 2]#, 5]#, 16, 21]
        print(indexes)
        train_examples = [reviews[i] for i in indexes]
        train_golds = [golds[i] for i in indexes]

        few_shot_ex_formatted = ""
        for i, (train_ex, gold) in enumerate(zip(train_examples, train_golds)):
            #few_shot_ex_formatted += f"Example {i+1}: {train_ex}\nUnexpected Aspects: {gold}\n\n"
            few_shot_ex_formatted += f"{prompt}\n{train_ex}\A: {gold}\n\n"

        input = []
        for ex in test_ex:
            input.append(f'{few_shot_ex_formatted}{prompt}\n{ex}\nA:')
            #input.append(f'question: {prompt}{few_shot_ex_formatted}context: '+ ex)
        #print('INPUT Format 1:', input[0])
        model_inputs = tokenizer(input, padding="longest", truncation=True, return_tensors="pt").to(device)
        #print(model_inputs)
        
        logits = model.generate(input_ids=model_inputs['input_ids'], max_length=256).to(device)

        decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
        print('decoded preds', decoded_preds)
        output_dict[','.join(str(x) for x in indexes)] = decoded_preds

    pd.DataFrame(output_dict).to_excel(f'modular_approach/few_shot_sample_test_{max_combination_len}.xlsx')
    # examples = ["Coming to New Orleans and not eating at Commander's Palace is like going to Paris and not visiting the Louvre. It's not just a meal but a memorable experience. WHAT WE ORDERED: the 3 course coolinary dinner, consisting of a 3 sample soup (turtle soup, gumbo du jour, and chef's seasonal soup), the black skillet seared gulf fish and the creole bread pudding souffle. We also ordered the pecan-crusted gulf fish and a glass of von hovel white wine.Out of the soups, the gumbo du jour was our favorite. The pecan-crusted gulf fish was divine. The creamy crushed-corn sauce that came with it wonderfully complimented the crunchiness of the pecan crust. The seared gulf fish was also complimented by a great, spicy, smoked tomato vinaigrette sauce. The bread pudding souffle was the most unique. It was topped with a bourbon sauce that added an extra level of sophistication to this traditional dessert. After our meal, our waiter, Todd, was kind enough to give us a tour of the estate, a perfect way to cap a perfect meal. Thanks to him we learned about the history of the restaurant as well as the craftsmanship that went into making every one of its dining rooms feel like a different setting altogether. The garden room has floor-to-ceiling panes of glass that overlook the patio, while the main dining room downstairs has hand-carved wooden replicas of the native birds of Louisiana as well as electric chandeliers, each worth $10,000, with lights that flicker and move as if they were candle flames. Todd also gave us a tour of the red wine collection, and the cellar room that can host a private party surrounded by their extensive collection of wine bottles. One of the cellar rooms is kept locked at all times because the wine bottles in there are so old that they shouldn't be moved, and obviously expensive ($6000 / per bottle). There is an adjacent garden, belonging to the house where Ella Brennan still resides, and which is tended to by a full-time gardener. Needless to say, the service was excellent. Todd really gave us a new appreciation for the restaurant and made us feel as if we were family.Another lovely detail about our meal was that all of the ingredients for the pecan-crusted gulf fish came within a 50-mile radius, while most of the ingredients that make up the dishes on the menu come within a 100-mile radius. The open kitchen, which guests are welcome to walk through, features a list of the local ingredients used in the meals. TIP: For an excellent briefing on commander's palace, ask for Todd as your waiter! For the most scenic seating, ask for a table overlooking the patio in the upstairs garden room.",
    # "This place is versatile as anything in the Lou! I've been here a handful of times and have always enjoyed it. I've gone for all different types of visits. I've gone for a late night drink and music, coffee and a study session, an evening meal, and an afternoon lunch and exploration. This place has it all. I have enjoyed going to do work or school work because they have bottomless coffee and free wifi. The fact that they serve their full menu all day is nice because I can order whatever I want whenever I want. I usually spend several hours working there and that is always nice to have. Waitresses never seem to mind when i stay for long periods of time and spend small amounts of money. In addition there is a lot of space to sit. Finally, I can always go upstairs to play ping pong or shuffleboard if I need a little study break. While there menu is not huge, it has quite a bit of variety. They offer very unique dishes, including some cool breakfast plates that are offered all day. I must mention that everything on the menu is made in house, including their bread, meats, and sauces, which I find to be impressive. Overall, the menu includes salads, soups, interesting appetizers, unique breakfast dishes, a variety of entree's, desserts, beers (9 oz/16 oz) and cocktails. For the most part, everything is modestly priced. This place continues to intrigue me because of all the different things it offers. It has multiple levels and it triples as a restaurant, bar, music venue. I can say I have not been there for a true concert yet, but they seem to have quite a few shows and are not usually too expensive to attend. The 1st floor is split between the music venue and the restaurant/bar. The 2nd floor has more open space that holds the shuffle board, ping pong table and more tables/chairs. It also has some balcony seating for the venue side as well as another bar on this floor. Again, I'd love to catch a show here sometime. Overall, I think that this place is worth a visit so that you can try out the menu and explore a little bit. It's not usually too crowded during the day and you could have free run of the place depending on when you go!",
    # "The best customer service! I'm sitting in here freezing and the waiter literally came and brought me a shawl to throw over my shoulders. Such a gentleman. I'm so grateful for Luis. Thank you.", "It's my second time coming to this place. I love the concept of it. They have healthy good food, great craft selection, tea, coffee, non dairy options. What cool about this place is how kid friendly it is. They have a section where you can take your kids. They have a professional babysitting service at the restaurant. Wanna drink a beer without your 3 year old, it possible at this restaurant. The restrooms are kid friendly also. Forgot your diaper, they got you. They have a good selection of different size diapers. Wanna work or finish a paper and have someone watch your kid. The food is pretty good but the portions are little small. I have had the Cuban and it comes with chips. I have also had one of their bowls and that was really good. I was able to customize both of those meals to be dairy free. Their kid meals are good portions and they have healthy option for children.", "We had a large group party, and we called ahead to make reservations. When we arrived, our tables were ready without wait, and the staff was so helpful and attentive! Large groups can be hard to deal with, and this place really came through. They made our night out amazing. Everyone enjoyed what they ordered, and many of us tried each other's food. Everything I tasted was GREAT! I wish I could come back and order one of everything!"]
    # input = []
    # # for ex in examples:
    # #     print(len(ex))
    # #     input.append(prompt + '\n\n' + ex)
    # #     input.append('question: '+prompt + ' context: '+ ex)
    #     #input.append('context: '+ ex +' question: '+prompt)

    # for ex in examples:
    #     #print(len(ex))
    #     # input.append(prompt + few_shot_ex_formatted + ex)
    #     # input.append(f'question: {prompt}{few_shot_ex_formatted}context: '+ ex)
    #     input.append(f'{few_shot_ex_formatted}{prompt} context:{ex}\nanswer:')
    #     #input.append(f'question: {prompt}{few_shot_ex_formatted}context: '+ ex)
    # #print('INPUT Format 1:', input[0])
    # #print('INPUT Format 2:', input[1])
    # #print('Prompt: ', prompt)
    # model_inputs = tokenizer(input, padding="longest", truncation=True, return_tensors="pt").to(device)
    # #print(model_inputs)
    
    # logits = model.generate(input_ids=model_inputs['input_ids'], max_length=256).to(device)

    # decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    # print('decoded preds', decoded_preds)

                    

        