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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name to use (e.g., gemma3:12b)')
parser.add_argument('--seed', type=int, required=True, help='Seed value to use')
parser.add_argument('--incsv', type=str, required=True, help='Input CSV file path')
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.isfile(args.incsv):
        print("Not a valid input file")
        sys.exit(0)

    tqdm.pandas()
    # prompts = [Prompter.simple_prompt_end,Prompter.definition_prompt_end,Prompter.sbs_prompt_end]
    prompts = [Prompter.sbs_prompt_end]
    # seeds = [123,42,37,57,50,73,0,69,420,100,12,4,49,25,17,444,1111,7,8,10,33,666,777,26,1999]
    seeds = [args.seed]
    alblations = ['vtt']

    models = [args.model]

    # models = ['llama3','mistral','gemma']
    for seed in seeds:

        options = {'temperature' :0,'seed':seed}
        for prompt in prompts: #iterate over the prompt
            prompt_name = prompt.__name__
            for ablation in alblations: #iterate over the
                base_path = os.path.splitext(args.incsv)[0]
                out_folder = os.path.join(base_path, str(seed), prompt_name, ablation)

                if not os.path.exists(out_folder):
                    # if the directory is not present create it
                    os.makedirs(out_folder)

                df = pd.read_csv(args.incsv) # same for every seed and every prompt

                #iterate over the models for each model we make an apply
                for model in models:
                    print(f'Iteration: Seed : {seed} - Prompt:{prompt_name} - Ablation:{ablation} - Model:{model}')
                    df[f'label_{model}'] = df.progress_apply(lambda x: prompt_model(model,Prompter.print_message(prompt(x['whisper_cc'],is_formatted=False,context=ablation)),options=options),axis=1)
                    df[['id',f'label_{model}']].to_csv(f'{out_folder}/{prompt_name}_{ablation}_label_{model}.csv')

                #save the prompt ablation combination
                df.to_csv(f'{out_folder}/{prompt_name}_{ablation}_labeled.csv')
        print('DONE!!!')

