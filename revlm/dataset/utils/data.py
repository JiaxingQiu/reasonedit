import os
import logging
from typing import List, Dict, Optional, Tuple, Union
import re

import pandas as pd
from huggingface_hub import snapshot_download

LOG = logging.getLogger(__name__)


def data_download_parquet_splits(
    repo_id: str, path_in_repo: str, cache_dir: Optional[str] = None
) -> Dict[str, Optional[Union[str, Tuple[str, ...]]]]:
    """Download train/val/test parquet files from a HF dataset directory."""
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{path_in_repo}/*.parquet"],
        cache_dir=cache_dir,
    )
    base_dir = os.path.join(local_root, path_in_repo)
    def _path(split: str) -> Optional[str]:
        candidate = os.path.join(base_dir, f"{split}.parquet")
        return candidate if os.path.exists(candidate) else None

    splits: Dict[str, Optional[Union[str, Tuple[str, ...]]]] = {
        split: _path(split)
        for split in ("train", "test")
    }

    train_path = splits.get("train")
    test_path = splits.get("test")
    if train_path and test_path:
        splits["all"] = (train_path, test_path)
    else:
        splits["all"] = train_path or test_path

    return splits


def data_load_split_df(parquet_path: Optional[Union[str, Tuple[str, ...]]]) -> pd.DataFrame:
    empty_df = pd.DataFrame(
        columns=["image_path", "question", "answer", "rationale", "choices", "idx_choices"]
    )

    if parquet_path is None:
        return empty_df.copy()

    if isinstance(parquet_path, (tuple, list)):
        dfs = [pd.read_parquet(path) for path in parquet_path if path is not None]
        return pd.concat(dfs, ignore_index=True) if dfs else empty_df.copy()

    return pd.read_parquet(parquet_path)


# def data_rows_to_examples(df: pd.DataFrame) -> List[Dict]:
#     """Convert a dataframe to trainer-ready dicts.

#     Required columns: image_path, question, answer, rationale, choices, idx_choices
#     """
#     cols = ["image_path", "question", "answer", "rationale", "choices", "idx_choices"]
#     missing = set(cols) - set(df.columns)
#     if missing:
#         raise ValueError(f"Parquet missing required columns: {missing}")
#     if df.empty:
#         return []

#     records = df[cols].to_dict(orient="records")
#     examples: List[Dict] = []
#     for r in records:
#         ex: Dict[str, object] = {
#             "image": r["image_path"],
#             "question": r["question"],
#             "answer": r["answer"],
#             "rationale": r["rationale"],
#             "choices": r["choices"],
#             "idx_choices": r["idx_choices"],
#         }
#         examples.append(ex)
#     return examples

