# ----------------------------------------------------------------------------
# Benchmark generation for text_generality
# ----------------------------------------------------------------------------
from pathlib import Path
import argparse
import json
import math
import os
import re

from openai import OpenAI
from revlm import VQADataset, configure_args


class TextGeneralizer:
    def __init__(self, dataset_name, openai_key=None):
        self.dataset_name = dataset_name
        self.batch_dir = Path(f"./data/related_text/{dataset_name}/requests/")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = Path(f"./data/related_text/{dataset_name}/meta/")
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(f"./data/related_text/{dataset_name}/outputs/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=openai_key)
        if self.dataset_name == "fvqa":
            self.n_batches = 50
        elif self.dataset_name == "aokvqa":
            self.n_batches = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        print(f"process {self.dataset_name} in {self.n_batches} batches")


    def format_prompt(self, question, num_versions=10):
        user_message = f"Please rephrase the following question in {num_versions} different ways: {question}."
        return user_message

    def gen_request(self, df):
        system_message = (
            "You are an AI assistant designed to rephrase questions."
        )
        # Construct JSON structure
        json_data = []
        for _, row in df.iterrows():
            entry = {
                "custom_id": str(row["uid"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini", # Change this to the model you want to use
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": self.format_prompt(row["question"])},
                    ],
                    "max_tokens": 300, # Change this to the desired max tokens
                    "temperature": 0.1
                }
            }
            json_data.append(entry)
        
        input_file = self.batch_dir / "batch.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for data in json_data:
                f.write(json.dumps(data) + "\n")
        print(f"JSON file created as '{input_file}'")

    def split_file(self, n_batches, remove_original=True):
        input_file = self.batch_dir / "batch.jsonl"
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        num_lines = len(lines)
        if num_lines == 0 or n_batches <= 0:
            return
        num_lines_per_file = max(1, math.ceil(num_lines / n_batches))
        for i in range(n_batches):
            start = i * num_lines_per_file
            end = start + num_lines_per_file
            chunk = lines[start:end]
            if not chunk:
                break
            split_path = input_file.parent / f"{input_file.stem}_{i}.jsonl"
            with open(split_path, "w", encoding="utf-8") as f:
                f.writelines(chunk)
        if remove_original:
            os.remove(input_file)


    def _save_meta(self, b, job_id, created_at=None):
        meta = {
            "job_id": job_id,
            "batch": b,
            "dataset": self.dataset_name,
            "created_at": created_at
        }
        with open(self.meta_dir / f"meta_{b}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(meta)
        return meta

    def _load_meta(self, b):
        with open(self.meta_dir / f"meta_{b}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # ----- runner functions -----
    def run_input(self):
        args = argparse.Namespace( split="all", dataset_name=self.dataset_name)
        config = configure_args(args, config_path=None)
        ds = VQADataset(config)
        df = ds.load_df()
        self.gen_request(df)
        self.split_file(self.n_batches)
    
    def resubmit_request(self, b_list):
        for b in b_list:
            meta_path = self.meta_dir / f"meta_{b}.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    job_id = meta["job_id"]
                    self.client.batches.cancel(job_id)
                meta_path.unlink()
            self._run_request_batch(b)
    
    def run_request(self):
        for b in range(self.n_batches):
            self._run_request_batch(b)

    def _run_request_batch(self, b):
        if not (self.meta_dir / f"meta_{b}.json").exists():
            batch_input_file = self.client.files.create(
                file=open(f"{str(self.batch_dir)}/batch_{b}.jsonl", "rb"),
                purpose="batch"
            )
            meta = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "prepare related questions: 10 rephrased questions per question.",
                    "dataset": self.dataset_name,
                    "batch_id": str(b),
                }
            )
            self._save_meta(b, meta.id, meta.created_at)

    
    def get_related_texts(self):
        related_texts = {}
        for b in range(self.n_batches):
            try:
                related_texts.update(self.get_related_texts_batch(b))
            except Exception as e:
                print(f"Error getting batch {b}: {e}")
                continue
        return related_texts
   
    def get_related_texts_batch(self, b):
        records = self._get_response_batch(b)
        # convert records into {uid: ["question_variant1", "question_variant2", ...]}.
        numbering = re.compile(r"^\s*\d+\.\s*")
        related_texts = {}
        for rec in records:
            uid = str(rec["uid"])
            lines = (line.strip() for line in rec["content"].splitlines() if line.strip())
            related_texts[uid] = [numbering.sub("", line) for line in lines]
        return related_texts
    
    def _get_response_batch(self, b):
        save_path = self.output_dir / f"outputs_{b}.jsonl"
        if save_path.exists(): # load response records from save_path
            with open(save_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
        else:
            meta_loc = self._load_meta(b)
            job_id = meta_loc["job_id"]
            meta = self.client.batches.retrieve(job_id)
            output_file_id = meta.output_file_id
            if not output_file_id:
                raise RuntimeError(f"Batch {b} job ({job_id}) has no output yet (status: {meta.status}).")
            file_resp = self.client.files.content(output_file_id)
            records = []
            for line in file_resp.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                records.append({
                    "uid": payload.get("custom_id"),
                    "content": payload.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", ""),
                })
            with open(save_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return records

    

from huggingface_hub import snapshot_download
import pandas as pd
import os

def get_t_gen_input(dataset_name, edit_ds, k: int = 5):
    edit_uids = [str(ex["uid"]) for ex in edit_ds.data]
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["t_gen/*.parquet"],  # or "i_gen/*.parquet" if you rename on HF
    )
    t_gen = pd.read_parquet(os.path.join(local_root, "t_gen", f"{dataset_name}.parquet"))
    # filter t_gen where t_gen["uid"] in edit_uids
    t_gen = t_gen[t_gen["uid"].isin(edit_uids)]
    # engineer into related_texts: {"uid": ["q1", "q2", ...]}
    related_texts = {
        str(row["uid"]): list(row["variants"])[:k]
        for _, row in t_gen.iterrows()
    }
    return related_texts
