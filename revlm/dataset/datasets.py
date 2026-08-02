from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from .utils import *
from huggingface_hub import snapshot_download
import pandas as pd

class VQADataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        df = self.load_df()
        self.data = self.df2data(df)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_df(self):
        if self.config.experiment.dataset_name == "fvqa":
            path_in_repo = "FVQA"
        elif self.config.experiment.dataset_name == "aokvqa":
            path_in_repo = "AOKVQA"
        elif self.config.experiment.dataset_name == "simulation":
            path_in_repo = "simulation"
        else:
            raise ValueError(f"Unknown dataset: {self.config.experiment.dataset_name}")
        split_paths = data_download_parquet_splits(
            repo_id="JJoy333/RationaleVQA",
            path_in_repo=path_in_repo,
        )
        df = data_load_split_df(split_paths.get(self.config.experiment.split))
        if self.config.experiment.dataset_name == "fvqa":
            df["image_info_source"] = df["image_path"].str.extract(r'/(COCO|ILSVRC)', expand=False)
            df["image_info_split"] = df["image_path"].str.extract(r'_(train|val|test)', expand=False)
            df["image_info_id"] = df["image_path"].str.extract(r'_(\d+)\.(jpg|jpeg|png|JPEG|JPG|PNG)$', expand=False)[0].astype(int)
            df["image_info_id"] = df["image_info_split"] + "_" + df["image_info_id"].astype(str)
            
        if self.config.experiment.dataset_name == "aokvqa":
            df["image_info_source"] = "COCO"
            df["image_info_split"] = df["image_path"].str.extract(r'/(train|val)\d+', expand=False)
            df["image_info_id"] = df["image_path"].str.extract(r'/(\d+)\.(jpg|jpeg|png|JPEG|JPG|PNG)$', expand=False)[0].astype(int)
            df["image_info_id"] = df["image_info_split"] + "_" + df["image_info_id"].astype(str)
        df["uid"] = df["uid"].astype(str)
        
        # add cot
        repo_id = "JJoy333/RationaleVQA"
        local_root = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=["r_gen/cot/*.parquet"])
        df_cot = pd.read_parquet(os.path.join(local_root, "r_gen", "cot", f"{self.config.experiment.dataset_name}.parquet"))
        df_cot["reason"] = df_cot["reason"].str.replace(r"[^;.!?]*[;.!?]\s*$", "", regex=True)
        df_cot["cot"] = df_cot["reason"]
        df = df.merge(df_cot[["uid", "cot"]], on="uid", how="left")
        df["cot"] = df["cot"].fillna(df["rationale"])
        return df
    
    def df2data(self, df: pd.DataFrame) -> List[Dict]:
        cols = ["uid", "image_path", "question", "answer", "rationale", "choices", "idx_choices"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")
        # If 'cot' is missing (e.g., in related_r_gen_df), default it to the original rationale
        if "cot" not in df.columns:
            df = df.copy()
            df["cot"] = df["reason"] if 'reason' in df.columns else df["rationale"]
        cols = cols + ["cot"]
        if df.empty:
            return []

        records = df[cols].to_dict(orient="records")
        examples: List[Dict] = []
        for r in records:
            ex: Dict[str, object] = {
                "uid": str(r["uid"]),
                "image": r["image_path"],
                "question": r["question"],
                "answer": r["answer"],
                "rationale": r["rationale"],
                "cot": r["cot"],
                "choices": r["choices"],
                "idx_choices": r["idx_choices"],
            }
            examples.append(ex)
        return examples
    
    def set_dataloader(self,
                        with_rationale=False,
                        use_cot=False,
                        rationale_in_prompt=True,
                        shuffle_choices=False,
                        unpaired=True):
            
        task = self.config.experiment.task
        batch_size = self.config.batch_size
        seed = self.config.seed

        self.task_engineer = get_taskengineer(task, 
                                              with_rationale=with_rationale, 
                                              use_cot=use_cot,
                                              rationale_in_prompt=rationale_in_prompt,
                                              shuffle_choices=shuffle_choices,
                                              unpaired=unpaired,
                                              seed=seed)
        for i, ex in enumerate(self.data): # ex is a reference to the dict stored in self.data
            ex['idx'] = i
            self.task_engineer.eng_golds(ex)
            self.task_engineer.eng_prompt(ex)
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=self.image_collate)
        
    def _resize_image(self, img, max_side=800):
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            s = max_side / m
            img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
        return img
    
    def image_collate(self, batch):
        """Collate function that loads images and returns a batch dict.
        Expects items with keys: 'image' (path), 'prompt' (string), 'gold' (dict), 'idx' (int).
        """
        images = []
        prompts = []
        golds = []
        idxs = []
        for ex in batch:
            img_path = ex["image"]
            try:
                img = self._resize_image(Image.open(img_path).convert("RGB"))
            except Exception as exc: 
                # Log the error type and offending image path, then skip this example.
                print(f"[image_collate] Skipping bad image ({exc.__class__.__name__}): {img_path}. Error: {exc}", flush=True)
                continue
            images.append(img)
            prompts.append(ex["prompt"])
            golds.append(ex["gold"])
            idxs.append(ex["idx"])
        return {
            "images": images,
            "prompts": prompts,
            "golds": golds,
            "idxs": idxs,
        }

    # def task_generate_batch(self, batch, model):
    #     """Generate predictions for a single collated batch and write back in place using indices."""
    #     outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=100)
    #     for idx, a in zip(batch["idxs"], outs):
    #         self.task_engineer.eng_preds(self.data[idx], a, model)

    def task_generate(self, model, use_cache=False, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, "max_new_tokens", 10)
        for batch in self.loader:
            # Skip empty batches (all images failed to load, e.g., corrupted/truncated images)
            if not batch["images"] or not batch["prompts"]:
                continue
            try:
                core = getattr(getattr(model, "model", model), "model", getattr(model, "model", model))
                # Re-enable KV cache for faster eval
                cfg = getattr(core, "config", None)
                if cfg is not None:
                    cfg.use_cache = use_cache
                # Qwen3-VL: reset rope deltas Disable gradient checkpointing if available, (owner can be core or core.model)
                if hasattr(core, "gradient_checkpointing_disable"): 
                    core.gradient_checkpointing_disable()
                rope_owner = core if hasattr(core, "rope_deltas") else getattr(core, "model", None) 
                if rope_owner is not None and hasattr(rope_owner, "rope_deltas"):
                    rope_owner.rope_deltas = None
            except Exception:
                pass
            outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=max_new_tokens, use_cache=use_cache)
            for idx, a in zip(batch["idxs"], outs):
                self.task_engineer.eng_preds(self.data[idx], a, model)
    
    def get_edits(self):
        pred_by = self.config.experiment.pred_by
        print(f"getting edits predicted by: {pred_by}")
        for ex in self.data:
            if ex['gold']['label'] != ex['pred'][pred_by]:
                ex['edit'] = True
            else:
                ex['edit'] = False
        edit_ds = self
        edit_ds.data = [ex for ex in edit_ds.data if ex['edit']]
        edit_ds.set_dataloader(shuffle_choices=False)
        return edit_ds

    def snap(self, out_path=None) -> None:
        if out_path is None:
            out_path = os.path.join(self.config.pred_dir, self.config.fname)
        with open(out_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def task_eval(self) -> None:
        task_metrics = self.task_engineer.eval(self)
        out_path = os.path.join(self.config.task_dir, self.config.fname)
        with open(out_path, "w") as f:
            json.dump(task_metrics, f, indent=2)
        print(f"Saved task evaluation metrics to {out_path}")
        print(f"Task evaluation metrics: {task_metrics}", flush=True)