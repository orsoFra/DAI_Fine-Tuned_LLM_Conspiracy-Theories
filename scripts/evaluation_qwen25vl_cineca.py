#!/usr/bin/env python3
"""Run resumable VideoLLaMA inference for one FPS value.

One Slurm array task should invoke this script for one FPS value and write to a
different output file.  Keeping array outputs separate avoids concurrent CSV
writes; ``analyze_videollama_fps.py`` combines them afterwards.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import(Qwen2_5_VLForConditionalGeneration,)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompter import Prompter  # noqa: E402


PROMPTS = {
    "simple": Prompter.simple_prompt_end_video,
    "definition": Prompter.definition_prompt_end_video,
    "sbs": Prompter.sbs_prompt_end_video,
}
OUTPUT_COLUMNS = [
    "id",
    "filename",
    "fps",
    "model",
    "prompt_method",
    "elapsed_time_processor",
    "elapsed_time_generate",
    "response",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infolder", type=Path, required=True)
    parser.add_argument("--outcsv", type=Path, required=True)
    parser.add_argument("--model", default="DAMO-NLP-SG/VideoLLaMA3-7B")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=[*PROMPTS, "all"],
        default=["sbs"],
        help="Keep one prompt fixed for a clean FPS sensitivity experiment.",
    )
    parser.add_argument("--ids-file", type=Path, help="Text/CSV subset manifest.")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--sample-size", type=int, help="Optional deterministic debug subset.")
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".mov", ".mkv", ".webm"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalise_id(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\.0$", "", text)


def read_ids(path: Path, id_column: str) -> set[str]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep, dtype={id_column: str})
        if id_column not in frame:
            raise ValueError(f"{path} has no {id_column!r} column")
        values = frame[id_column]
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    return {normalise_id(value) for value in values if str(value).strip()}


def checkpoint(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, output)

def gpu_memory(label: str) -> None:
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    free, total = torch.cuda.mem_get_info()

    print(
        f"[GPU] {label}: "
        f"allocated={allocated:.2f} GiB, "
        f"reserved={reserved:.2f} GiB, "
        f"peak={peak:.2f} GiB, "
        f"free={free / 1024**3:.2f}/{total / 1024**3:.2f} GiB",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if not args.infolder.is_dir():
        raise FileNotFoundError(f"Not a directory: {args.infolder}")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")

    prompt_names = list(PROMPTS) if "all" in args.prompts else args.prompts
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}
    files = sorted(p for p in args.infolder.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if args.ids_file:
        selected_ids = read_ids(args.ids_file, args.id_column)
        files = [path for path in files if normalise_id(path.stem) in selected_ids]
    if args.sample_size is not None:
        if args.sample_size < 1:
            raise ValueError("--sample-size must be at least 1")
        random.Random(args.sample_seed).shuffle(files)
        files = sorted(files[: args.sample_size])
    if not files:
        raise RuntimeError("No videos matched the requested input and subset")

    rows: list[dict[str, object]] = []
    if args.outcsv.exists() and not args.overwrite:
        previous = pd.read_csv(args.outcsv)
        missing = set(OUTPUT_COLUMNS) - set(previous.columns)
        if missing:
            raise ValueError(
                f"Cannot resume legacy/incompatible output {args.outcsv}; missing {sorted(missing)}. "
                "Use a new path or --overwrite."
            )
        rows = previous[OUTPUT_COLUMNS].to_dict("records")
    completed = {
        (normalise_id(row["id"]), str(row["prompt_method"]), float(row["fps"]))
        for row in rows
        if row["status"] == "ok"
    }

    dtype = getattr(torch, args.dtype)
    gpu_memory("before model")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )
    gpu_memory("after model")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,    min_pixels=256 * 28 * 28,
    max_pixels=256 * 28 * 28)
    print("Model dtype:", next(model.parameters()).dtype)
    print("Attention implementation:", getattr(model.config, "_attn_implementation", None))
    written_since_checkpoint = 0
    for path in tqdm(files, desc=f"VideoLLaMA fps={args.fps:g}"):
        video_id = normalise_id(path.stem)
        for prompt_name in prompt_names:
            prompt_method = PROMPTS[prompt_name].__name__
            key = (video_id, prompt_method, args.fps)
            if key in completed:
                continue
            record: dict[str, object] = {
                "id": video_id,
                "filename": path.name,
                "fps": args.fps,
                "model": args.model,
                "prompt_method": prompt_method,
                "elapsed_time_processor": None,
                "elapsed_time_generate": None,
                "response": None,
                "status": "error",
                "error": None,
            }
            try:
                prompt_text = PROMPTS[prompt_name]

                conversation = [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "video",
                        "path": str(path.resolve()),
                        "fps":args.fps,
                        "max_frames":8,
                        "max_pixels":256*28*28,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    ],
                }
                ]
                started = time.perf_counter()
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                record["elapsed_time_processor"] = time.perf_counter() - started
                torch.cuda.reset_peak_memory_stats()
                gpu_memory("before preprocessing")
                inputs = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
                gpu_memory("after preprocessing")
                print(inputs["input_ids"].shape[-1])
                print(inputs.get("video_grid_thw"))
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                generation_args = {"max_new_tokens": 32, "do_sample": False}
                if args.temperature > 0:
                    generation_args["temperature"] = args.temperature
                started = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, **generation_args)
                record["elapsed_time_generate"] = time.perf_counter() - started
                generated_ids = [
    output[len(input_ids):]for input_ids, output in zip(inputs["input_ids"], output_ids)]
                record["status"] = "ok"
                completed.add(key)
                record["response"] = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            except Exception as exc:  # Preserve progress and make failed cases auditable.
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"ERROR {path.name} {prompt_method}: {record['error']}", file=sys.stderr)
            finally:
                inputs=None
                output_ids=None
                generated_ids=None

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            rows.append(record)
            written_since_checkpoint += 1
            if written_since_checkpoint >= args.checkpoint_every:
                checkpoint(rows, args.outcsv)
                written_since_checkpoint = 0
    checkpoint(rows, args.outcsv)
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"Saved {len(rows)} rows to {args.outcsv} ({failures} failed attempts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
