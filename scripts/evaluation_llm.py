#structure of the ppromps
import sys 
sys.path.insert(0,'../scripts/')
from prompter import Prompter
import ollama
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
import os 

def prompt_model(model:str , messages: list, options: dict):
    try:
        response = ollama.chat(model=model,options=options, messages=messages,stream=False)
    except:
        return -1
    try:
        out = int(response['message']['content'])
    except:
        out = response['message']['content']
    return out

if __name__ == "__main__":    
    tqdm.pandas()
    prompts = [Prompter.simple_prompt_end,Prompter.definition_prompt_end,Prompter.sbs_prompt_end]
    seeds = [123,42,37,57,50,73,0,69,420,100,12,4,49,25,17,444,1111,7,8,10,33,666,777,26,1999]
    alblations = ['vtt']
    
    models = ['gemma3:27b'] #models over which we perform the experiments
    #models = ['llama3','mistral','gemma']#models over which we perform the experiments
    for seed in seeds:

        options = {'temperature' :0,'seed':seed}
        for prompt in prompts: #iterate over the prompt
            prompt_name = prompt.__name__
            for ablation in alblations: #iterate over the 
                out_folder = f"../data/processed/df_processed/full_unbalanced_end_prompt/{seed}/{prompt_name}/{ablation}"
                if not os.path.exists(out_folder): 
                    # if the directory is not present create it. 
                    os.makedirs(out_folder)
                
                df = pd.read_csv(f'../data/processed/df_processed/imbalanced_test.csv') # change the dataset based on the task, it's the same for every seed and every prompt
                #iterate over the models for each model we make an apply 
                for model in models:
                    print(f'Iteration: Seed : {seed} - Prompt:{prompt_name} - Ablation:{ablation} - Model:{model}')
                    df[f'label_{model}'] = df.progress_apply(lambda x: prompt_model(model,Prompter.print_message(prompt(x['whisper_cc'],is_formatted=False,context=ablation)),options=options),axis=1)
                    df[['id',f'label_{model}']].to_csv(f'{out_folder}/{prompt_name}_{ablation}_label_{model}.csv')
                #save the prompt ablation combination
                df.to_csv(f'{out_folder}/{prompt_name}_{ablation}_labeled.csv')
        print('DONE!!!')
            
