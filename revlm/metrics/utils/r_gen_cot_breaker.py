# ----------------------------------------------------------------------------
# Benchmark generation for text_generality
# ----------------------------------------------------------------------------
from pathlib import Path
import argparse
import json
import math
import os
import pandas as pd

from openai import OpenAI
from revlm import VQADataset, configure_args


class CoTBreaker:
    def __init__(self, dataset_name, openai_key=None):
        self.dataset_name = dataset_name
        self.batch_dir = Path(f"./data/r_gen/cot/{dataset_name}/requests/")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = Path(f"./data/r_gen/cot/{dataset_name}/meta/")
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(f"./data/r_gen/cot/{dataset_name}/outputs/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=openai_key)
        if self.dataset_name == "fvqa":
            self.n_batches = 50
        elif self.dataset_name == "aokvqa":
            self.n_batches = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        print(f"process {self.dataset_name} in {self.n_batches} batches")


    def format_prompt(self, question, rationale, answer):
        # user_message = (
        #     f"Given (question: {question}, answer: {answer}, rationale: {rationale}). "
        #     "Which visual object(s) in the image are mentioned or implied in the rationale to support the answer? "
        #     "List all relevant objects. "
        #     "Then as if you are reasoning it out without knowing the answer beforehand, "
        #     "rewrite the rationale as 2–3 short sentences explaining how these objects lead to that answer. "
        #     "Use simple, declarative sentences separated by periods. Do not use 'this' or 'it' to refer to earlier sentences. "
        #     "Avoid meta phrases like 'support the answer', 'support the conclusion', 'evidence', 'question', or 'answer'."
        #     "Respond in the format:\n[object1, object2, ...]\nReason: <2–3 short declarative sentences>."
        # )
        user_message = (
            f"Given (question: {question}, answer: {answer}, rationale: {rationale}). "
            "Which visual object(s) in the image are mentioned or implied in the rationale to support the answer? List all relevant objects. "
            "Then as if you are reasoning it out step by step without knowing the answer beforehand, "
            "rewrite the rationale into 3-5 chain-of-thought sentences that explain how these objects lead to that answer."
            "\n\n"
            "Guidelines for the chain of thought:\n"
            "- Each sentence should be declarative, fact-based, and self-contained.\n"
            "- Use simple, short sentences separated by periods.\n"
            "- Do not use 'this' or 'it' to refer to earlier sentences; repeat the key nouns instead.\n"
            "- Avoid meta phrases like 'support the answer', 'support the conclusion', 'evidence', 'question', or 'answer'."
            "\n\n"
            "Chain-of-thought template:\n"
            "The image shows [object1, object2, ...]. "
            "[Fact sentence 1, Fact sentence 2, ...] "
            "So [conclusion consistent with the answer]."
            "\n\n"
            "Chain of thought examples:\n"
            "Example 1:\n"
            "The image shows a person standing on a board in the water ."
            "There are waves around the board. "
            "A person on a board in the ocean waves is usually surfing. "
            "So the person is likely surfing."
            "\n\n"
            "Example 2:\n"
            "The image shows a round fruit. "
            "The peel is bright orange. "
            "Oranges are round fruits with bright orange peels. "
            "So the fruit is most likely an orange. "
            "\n\n"
            "Respond in the format:\n[object1, object2, ...]\nReason: <chain of thought sentences>."
        )
        return user_message

    def gen_request(self, df):
        # system_message = (
        #     "You are an assistant that reasons like a human to explain how visual evidence supports an answer. "
        #     "Identify the visual objects mentioned or implied in the rationale, and rewrite the rationale as 2–3 short declarative sentences."
        # )
        system_message = (
            "You are an assistant that reasons like a human to explain how visual evidence supports an answer. "
            "Identify the visual objects mentioned or implied in the rationale, and rewrite the rationale as step-by-step chain of thought."
        )
        # Construct JSON structure
        json_data = []
        for _, row in df.iterrows():
            entry = {
                "custom_id": str(row["uid"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o", # Change this to the model you want to use
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": self.format_prompt(row["question"], row["rationale"], row["answer"])},
                    ],
                    "max_tokens": 100, # Change this to the desired max tokens
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
                    status = self.client.batches.retrieve(job_id).status
                    if status in {"validating", "queued", "in_progress"}:
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
                    "description": "break down the rationale into visual objects and their relationships to the answer.",
                    "dataset": self.dataset_name,
                    "batch_id": str(b),
                }
            )
            self._save_meta(b, meta.id, meta.created_at)

    
    def get_cot(self):
        cot_df = pd.DataFrame(columns=["uid", "cot", "objects", "reason"])
        for b in range(self.n_batches):
            cot_df = pd.concat([cot_df, self._get_cot_batch(b)], ignore_index=True)
        return cot_df
   
    def _get_cot_batch(self, b):
        cot_df = pd.DataFrame(columns=["uid", "cot", "objects", "reason"])
        try:
            records = self._get_response_batch(b)
        except Exception as e:
            print(f"Error getting response for batch {b}: {e}")
            return cot_df
        for rec in records:
            try:
                uid = str(rec["uid"])
                answer = rec["content"]
                objects = answer.split("Reason:")[0].strip()
                reason = answer.split("Reason:")[1].strip()
                cot_df = cot_df.append({"uid": uid, "cot": answer, "objects": objects, "reason": reason}, ignore_index=True)
            except Exception as e:
                print(f"Error getting rationale breakdown for batch {b} uid {uid} with answer '{answer}'\n{e}")
                continue
        return cot_df
    
    def get_response(self):
        response = {}
        for b in range(self.n_batches):
            try:
                response.update(self._get_response_batch(b))
            except Exception as e:
                print(f"Error getting batch {b}: {e}")
                continue
        return response
    
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

    