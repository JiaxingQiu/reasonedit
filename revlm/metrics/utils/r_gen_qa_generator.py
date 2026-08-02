# ----------------------------------------------------------------------------
# Benchmark generation for text_generality
# ----------------------------------------------------------------------------
from pathlib import Path
import argparse
import json
import math
import os
import pandas as pd
from huggingface_hub import snapshot_download
from openai import OpenAI


class QAGenerator:
    def __init__(self, dataset_name, openai_key=None):
        self.dataset_name = dataset_name
        self.batch_dir = Path(f"./data/r_gen/qa/{dataset_name}/requests/")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = Path(f"./data/r_gen/qa/{dataset_name}/meta/")
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(f"./data/r_gen/qa/{dataset_name}/outputs/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_subsets_dir = Path(f"./data/r_gen/qa/all_subsets/")
        self.all_subsets_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=openai_key)
        if self.dataset_name == "fvqa":
            self.n_batches = 50
        elif self.dataset_name == "aokvqa":
            self.n_batches = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        print(f"process {self.dataset_name} in {self.n_batches} batches")


    def format_prompt(self, facts):
        # user_message = (
        #     f"Given these facts: {facts}\n\n"
        #     "Generate a question and its correct answer, as well as three wrong answers. "
        #     "Format: question? correct_answer; wrong_answer1; wrong_answer2; wrong_answer3"
        # )
        user_message = (
            f"You are creating a multiple-choice question from the given facts: {facts}\n\n"
            "Task:\n"
            "1. Write one question that can be answered using only these facts.\n"
            "2. Write one correct answer.\n"
            "3. Write three incorrect but plausible answers.\n\n"
            "Constraints:\n"
            "- Use only information that is implied by the facts; do not invent new facts.\n"
            "- The correct answer must be clearly correct.\n"
            "- Each wrong answer must be clearly wrong given the facts.\n"
            # "- Answers should be a word or a phrase, not a full sentence.\n"
            "- Answers should be no more than 3 words, not full sentences.\n"
            "Respond in the format:\n"
            "Question: question?\n Answers: correct_answer | wrong_answer1 | wrong_answer2 | wrong_answer3"
        )
        return user_message

    def gen_request(self, df):
        system_message = (
            "You create clean multiple-choice questions from given facts. "
            "Follow the format exactly and output only the requested line."
        )
        # Construct JSON structure
        json_data = []
        for _, row in df.iterrows():
            entry = {
                "custom_id": str(row["sid"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o", # Change this to the model you want to use
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": self.format_prompt(row["rationale"])},
                    ],
                    "max_tokens": 100, # Change this to the desired max tokens
                    "temperature": 1.0
                }
            }
            json_data.append(entry)
        
        input_file = self.batch_dir / "batch.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for data in json_data:
                f.write(json.dumps(data) + "\n")
        print(f"JSON file created as '{input_file}'")

    def gen_all_subsets(self, df):
        all_rows = []
        for _, row in df.iterrows():
            #row['rationale']  has multiple sentences "s1. s2. s3. ..."
            # find all ordered subsets of sentences
            sentences = row['reason'].split('. ')
            sentences = sentences[:-1] # remove the last sentence because it is the answer
            sid = 0
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)+1):
                    sid += 1
                    subset = sentences[i:j]
                    sub_reason = '. '.join(subset)
                    if not sub_reason.endswith("."):
                        sub_reason += "."
                    all_rows.append({"sid": str(row["uid"]) + "_" + str(sid), "rationale": sub_reason})
        df = pd.DataFrame(all_rows, columns=["sid", "rationale"])
        parquet_path = self.all_subsets_dir / f"{self.dataset_name}.parquet"
        df.to_parquet(parquet_path)
        print(f"All subsets saved to {parquet_path}")

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
        repo_id = "JJoy333/RationaleVQA"
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["r_gen/cot/*.parquet"],
        )
        df = pd.read_parquet(os.path.join(local_root, "r_gen", "cot", f"{self.dataset_name}.parquet"))
        self.gen_all_subsets(df)
        df = pd.read_parquet(self.all_subsets_dir / f"{self.dataset_name}.parquet")
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
            output_path = self.output_dir / f"outputs_{b}.jsonl"
            if output_path.exists():
                os.remove(output_path)
            self._run_request_batch(b)
    
    def run_request(self):
        for b in range(self.n_batches):
            self._run_request_batch(b)

    def _run_request_batch(self, b):
        if not (self.meta_dir / f"meta_{b}.json").exists():
            with open(f"{str(self.batch_dir)}/batch_{b}.jsonl", "rb") as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            meta = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "create a multiple-choice question from the given facts.",
                    "dataset": self.dataset_name,
                    "batch_id": str(b),
                }
            )
            self._save_meta(b, meta.id, meta.created_at)

    
    def get_qa(self):
        qa_df = pd.DataFrame(columns=["uid", "sid", "question", "answers", "response"])
        for b in range(self.n_batches):
            qa_df = pd.concat([qa_df, self._get_qa_batch(b)], ignore_index=True)
        return qa_df
   
    def _get_qa_batch(self, b):
        rows = []
        try:
            records = self._get_response_batch(b)
        except Exception as e:
            print(f"Error getting response for batch {b}: {e}")
            return pd.DataFrame(columns=["uid", "sid", "question", "answers", "response"])
        for rec in records:
            try:
                uid = str(rec["uid"])
                sid = str(rec["sid"])
                response = rec["content"]
                # Parse format: "Question: ...\n Answers: ... | ... | ... | ..."
                if "Question:" in response and "Answers:" in response:
                    question = response.split("Question:")[1].split("\n")[0].strip()
                    answers = response.split("Answers:")[1].strip()
                    rows.append({
                        "uid": uid,
                        "sid": sid,
                        "question": question,
                        "answers": answers,
                        "response": response
                    })
                else:
                    # Fallback for unexpected format
                    rows.append({
                        "uid": uid,
                        "sid": sid,
                        "question": "",
                        "answers": response,
                        "response": response    # keep the original response for debugging
                    })
            except Exception as e:
                print(f"Error parsing response for batch {b} uid {uid} with response '{response}'\n{e}")
                continue
        df = pd.DataFrame(rows, columns=["uid", "sid", "question", "answers", "response"])
        # load all_subsets_dir / f"{self.dataset_name}.parquet" back, merge to df on sid
        all_subsets_df = pd.read_parquet(self.all_subsets_dir / f"{self.dataset_name}.parquet")
        df = df.merge(all_subsets_df, on="sid", how="left")
        return df
    
    def get_response(self):
        all_records = []
        for b in range(self.n_batches):
            try:
                all_records.extend(self._get_response_batch(b))
            except Exception as e:
                print(f"Error getting batch {b}: {e}")
                continue
        return all_records
    
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
                custom_id = payload.get("custom_id", "")
                records.append({
                    "uid": custom_id.split("_")[0] if custom_id else "",
                    "sid": custom_id,
                    "content": payload.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", ""),
                })
            with open(save_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return records

    