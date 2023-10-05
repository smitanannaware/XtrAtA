import argparse
import openai
import time
import pandas as pd


openai.organization = ""
openai.api_key = ''


def getChatGPTResponse(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature = 0,
    )

    return response


def getPrediction(row):
    #print(row)
    review = row['review']
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": f'{prompt}\n{review}'})
    response = getChatGPTResponse(messages)
    #print(messages)
    print(response.choices[0].message.content)
    #messages.append({"role": response.choices[0].message.role, "content": response.choices[0].message.content}
    messages = []
    row['prediction'] = response.choices[0].message.content
    return row


def pre_process_data(text):
    text = text.replace('\n','\n- ')
    text = text.strip()
    return text


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Evaluate the finetuned model for identifying initiative apsects.")
    parser.add_argument("--result_dir", default="results", required=False)
    parser.add_argument("--domain", default="restaurant", required=False)
    parser.add_argument("--file_name", default="trial_4", required=False)
    parser.add_argument("--ref_column_name", default="true_strong", required=False)
    parser.add_argument("--data_dir", default="restaurants/v5", required=False)
    parser.add_argument("--model_name", default="GPT", required=False)
    parser.add_argument("--indices", default="1,2", required=False)
    parser.add_argument("--prompt", default="What are the atypical aspects mentioned in the text below?", required=False)

    args = parser.parse_args()
    base_dir = 'code'
    eval_set = pd.read_excel(f'{base_dir}/dataset/{args.data_dir}/test/test.xlsx')
    eval_set = eval_set[['review', 'true_strong', 'true_strong_weak', 'abs_true_strong_alt', 'abs_true_strong_weak_alt']]
    
    dev_set = pd.read_excel(f'{base_dir}/dataset/{args.data_dir}/dev/dev.xlsx')
    dev_set = dev_set[['review', 'true_strong', 'true_strong_weak', 'abs_true_strong_alt', 'abs_true_strong_weak_alt']]

    dev_set = dev_set.fillna(str('None'))

    is_strong = True
    if 'weak' in args.ref_column_name: is_strong = False

    indices = [int(i) for i in args.indices.split(',')]   # get list of indices to for few shot examples.  Each is a number.
    print('Indices: ', indices)

    fewshot_examples = pd.DataFrame(dev_set[dev_set.index.isin(indices)])
    fewshot_examples[['abs_true_strong_alt', 'abs_true_strong_weak_alt']] = fewshot_examples[['abs_true_strong_alt', 'abs_true_strong_weak_alt']].applymap(lambda x:  pre_process_data(x) if isinstance(x, str) else x)
    
    prompt = args.prompt
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]
    print('Prompt: ', prompt)   # prompt to the user.  This is for the task of identifying atypical aspects. 
    for i, index in enumerate(indices):
        if 'abstractive' in args.result_dir:
            prompt += f"\nExample {i+1}: {fewshot_examples['review'][index]}\nAtypical aspects:\n- {fewshot_examples[args.ref_column_name][index]}\n"
        else: prompt += f"\nExample {i+1}: {fewshot_examples['review'][index]}\nAtypical aspects: {fewshot_examples[args.ref_column_name][index]}\n"

    prompt += f'\nCan you try for a {args.domain} review below?'
    print(prompt)
    df2 = pd.DataFrame()
    
    batch_size = 10
    for i in range(0, len(eval_set), batch_size):
        if i == 5:
            break
        print(i)
        eval_set_i = eval_set[i:i+batch_size]
        try:
            eval_set_i = eval_set_i.apply(getPrediction, axis=1)
        except:
            print('Waiting 90 sec to retry....')
            time.sleep(90)
            eval_set_i = eval_set_i.apply(getPrediction, axis=1)
        
        if i == 0:
            new_row = pd.DataFrame({'review': prompt, 'abs_true_strong_alt':'', 'abs_true_strong_weak_alt': ''}, index =[0])
            df = pd.concat([new_row, eval_set_i]).reset_index(drop = True)
            df2 = pd.concat([df2, df])  #concat and reset index ids.
        else:
            df2 = pd.concat([df2, eval_set_i])

        time.sleep(60)

    df2.to_excel(f'{base_dir}/results/{args.result_dir}/{args.data_dir}/{args.model_name}/{args.file_name}_{str(len(indices))}shot_{"strong" if is_strong else "strong_weak"}.xlsx', index=False)