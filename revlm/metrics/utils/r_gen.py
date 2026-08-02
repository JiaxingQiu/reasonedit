import pandas as pd
import os
import json
from huggingface_hub import snapshot_download
import string

def get_r_gen_input(dataset_name, edit_ds=None, s: int = 0, filter_bad: bool = False):
    """Load caption dataframe from HuggingFace dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "aokvqa", "fvqa")
        edit_ds: Optional VQADataset to filter by uids
        s: Minimum number of sentences in rationale (0 = no filter)
        filter_bad: If True, filter out bad sids from data/r_gen/remove/{dataset_name}.json
    """
    repo_id = "JJoy333/RationaleVQA"
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["r_gen/qa/*.parquet"],
    )
    r_gen_df = pd.read_parquet(os.path.join(local_root, "r_gen", "qa", f"{dataset_name}.parquet"))
    r_gen_df = to_mc_format(r_gen_df)
    print(f"[r_gen] Loaded: {len(r_gen_df)} rows, {r_gen_df['uid'].nunique()} uids", flush=True)
    
    # filter rows where rationale has at least s sentences
    if s > 0:
        r = r_gen_df["rationale"].fillna("").astype(str)
        n = r.str.split(r"[.!?]+\s*").apply(lambda x: len([p for p in x if p.strip()]))
        r_gen_df = r_gen_df[n >= int(s)]
        print(f"[r_gen] After s>={s} filter: {len(r_gen_df)} rows, {r_gen_df['uid'].nunique()} uids", flush=True)
    
    # Filter out bad sids if requested
    if filter_bad:
        bad_sids_path = f"data/r_gen/remove/{dataset_name}.json"
        if os.path.exists(bad_sids_path):
            with open(bad_sids_path, "r") as f:
                bad_sids = set(json.load(f))
            before_count = len(r_gen_df)
            r_gen_df = r_gen_df[~r_gen_df["sid"].astype(str).isin(bad_sids)]
            print(f"[r_gen] After bad_sids filter: {len(r_gen_df)} rows, {r_gen_df['uid'].nunique()} uids (removed {before_count - len(r_gen_df)} rows)", flush=True)
    
    # Filter by edit_ds uids if provided
    if edit_ds is not None:
        edit_uids = [str(ex["uid"]) for ex in edit_ds.data]
        r_gen_df = r_gen_df[r_gen_df["uid"].isin(edit_uids)]
        print(f"[r_gen] After edit_ds filter: {len(r_gen_df)} rows, {r_gen_df['uid'].nunique()} uids", flush=True)
    
    # Derive image_path directly from sid (deterministic path pattern)
    r_gen_df["image_path"] = f"data/r_gen/image/{dataset_name}/" + r_gen_df["sid"].astype(str) + ".png"
    return r_gen_df

def to_mc_format(r_gen_df: pd.DataFrame) -> pd.DataFrame:
    # split "answers" into list of options
    opts = (
        r_gen_df["answers"]
        .astype(str)
        .str.split("|")
        .apply(lambda xs: [x.strip() for x in xs])
    )

    # ground-truth answer = first option
    r_gen_df["answer"] = opts.str[0]

    # all choices in one string, separated by "; "
    r_gen_df["choices"] = opts.apply(lambda xs: "; ".join(xs))

    # "(A) choice1\n(B) choice2\n..." format
    letters = string.ascii_uppercase
    def make_idx(xs):
        return "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(xs))

    r_gen_df["idx_choices"] = opts.apply(make_idx)

    # final columns in the same order as your second dataframe
    wanted_cols = ["uid", "sid", "question", "answer", "rationale", "choices", "idx_choices"]    
    return r_gen_df[wanted_cols].copy()

