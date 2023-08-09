import argparse
import openai
import time
import pandas as pd


openai.organization = "org-wl3zexGh6Yzm0MvhR0wEL9xG"
openai.api_key = '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'
# erfan's api keys
# openai.organization = "org-Mvb9JPXWWhgMfuSbG5StfwvI"
# openai.api_key = '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'

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
    messages.append({"role": "user", "content": f'{args.prompt} {review}'})
    response = getChatGPTResponse(messages)
    #print(messages)
    print(response.choices[0].message.content)
    #messages.append({"role": response.choices[0].message.role, "content": response.choices[0].message.content}
    messages = []
    row['prediction'] = response.choices[0].message.content
    return row


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Evaluate the finetuned model for identifying initiative apsects.")
    parser.add_argument("--result_dir", default="results", required=False)
    parser.add_argument("--domain", default="restaurants", required=False)
    parser.add_argument("--file_name", default="trial_4", required=False)
    parser.add_argument("--ref_column_name", default="true_strong", required=False)
    parser.add_argument("--data_dir", default="restaurants/v5", required=False)
    parser.add_argument("--model_name", default="GPT", required=False)
    parser.add_argument("--prompt", default="What are the atypical aspects mentioned in the text below?", required=False)

    args = parser.parse_args()
    base_dir = 'code/modular_approach'
    eval_set = pd.read_excel(f'{base_dir}/dataset/{args.data_dir}/test/test.xlsx')
    eval_set = eval_set[['review', 'true_strong', 'true_strong_weak', 'abs_true_strong_alt', 'abs_true_strong_weak_alt']]

    prompt = args.prompt
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]
    print(prompt)  
    df2 = pd.DataFrame()
    
    batch_size = 5
    for i in range(0, len(eval_set), batch_size):
        if i == 5: 
            break
        print(i)
        eval_set_i = eval_set[i:i+batch_size]
        eval_set_i = eval_set_i.apply(getPrediction, axis=1)
        
        if i == 0:
            new_row = pd.DataFrame({'review': prompt, 'abs_true_strong_alt':'', 'abs_true_strong_weak_alt': ''}, index =[0])
            df = pd.concat([new_row, eval_set_i]).reset_index(drop = True)
            df2 = pd.concat([df2, df])  #concat and reset index ids.
        else:
            df2 = pd.concat([df2, eval_set_i])

        time.sleep(60)

    df2.to_excel(f'{base_dir}/results/{args.result_dir}/{args.data_dir}/{args.model_name}/{args.file_name}.xlsx', index=False)