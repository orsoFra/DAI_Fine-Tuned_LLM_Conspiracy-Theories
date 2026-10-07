import sys
sys.path.insert(0,'../scripts/')
from prompter import Prompter
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import time
import os

# os.environ["PATH"] = custom_path + os.pathsep + os.environ.get("PATH", "")

parser = argparse.ArgumentParser()
parser.add_argument('--infolder', type=str , help='Input video folder')
parser.add_argument('--outcsv', type=str, help='Output CSV file')
parser.add_argument('--model', type=str, default='DAMO-NLP-SG/VideoLLaMA3-7B', help='Model name')
args = parser.parse_args()

if __name__ == "__main__":

    device = "cuda:0"
    model_path = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if not os.path.isdir(args.infolder):
        print("Not a valid input folder")
        sys.exit(0)

    prompts = [("simple_prompt_end_video", Prompter.simple_prompt_end_video),
               ("definition_prompt_end_video", Prompter.definition_prompt_end_video),
               ("sbs_prompt_end_video", Prompter.sbs_prompt_end_video)]

    results = []
    files = sorted(os.listdir(args.infolder))

    for filename in tqdm(files, desc="Processing files"):
        filepath = os.path.join(args.infolder, filename)
        if not os.path.isfile(filepath):
            print(filepath + " IS INVALID WTF")
            continue

        for prompt_method, prompt in prompts:
            conversation = prompt(video_path=args.infolder, file_name=filename, fps=1, is_formatted=False)
            # print(conversation)
            start_time = time.time()
            try:
                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                end_time = time.time()
                elapsed_time_ffprobe = end_time - start_time

                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                start_time = time.time()
                output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.01)
                end_time = time.time()
                elapsed_time_generate = end_time - start_time

                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # print(response)
                results.append((filename, prompt_method, elapsed_time_ffprobe, elapsed_time_generate, response))
            except Exception as e:
                print(e)#skip iteration
                continue

    df = pd.DataFrame(results, columns=["filename", "prompt_method", "elapsed_time_processor", "elapsed_time_generate", "response"])
    df.to_csv(args.outcsv, index=False)
    print(f"Saved results to {args.outcsv}")
