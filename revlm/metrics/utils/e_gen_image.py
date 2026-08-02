# ----------------------------------------------------------------------------
# COE Image Generation: Generate images from scenario + error chain
# ----------------------------------------------------------------------------
import os
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from huggingface_hub import snapshot_download


def load_scenario_df(model_name: str, dataset_name: str) -> pd.DataFrame:
    """Load merged scenario parquet from HuggingFace."""
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["coe_gen/lite/*.parquet"],
    )
    path = os.path.join(local_root, "coe_gen", "lite", f"{model_name}_{dataset_name}.parquet")
    return pd.read_parquet(path)


def load_coe_sentences(model_name: str, dataset_name: str) -> Dict[str, List[str]]:
    """Load COE results and return uid -> sentences mapping."""
    path = f"results/pred_postedit/baseline/{model_name}/{dataset_name}/coe_prediction.json"
    with open(path, "r") as f:
        results = json.load(f)
    return {str(r["uid"]): r["coe_pred"]["sentences"] for r in results}


def build_prompt(scenario: str, sentences: List[str], indices: List[int]) -> str:
    """Build image generation prompt: scenario + error sentences."""
    error_text = " ".join(sentences[i] for i in indices if i < len(sentences))
    return f"{scenario} {error_text}".strip()

