from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import os

def _resolve_model_id(config):
    model_name = getattr(getattr(config, "model", {}), "name", "")
    model_pt = getattr(getattr(config, "model", {}), "pt", None)
    return model_pt or model_name or "Salesforce/instructblip-vicuna-7b"


def get_hf_model_instructblip(config, cache_dir=None, torch_dtype=None):
    model_id = _resolve_model_id(config)
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    return InstructBlipForConditionalGeneration.from_pretrained(model_id, **load_kwargs)


def get_processor_instructblip(config, cache_dir=None):
    """
    Load InstructBLIP processor with HPC-friendly timeout handling.
    Uses local_files_only=True by default to avoid network timeouts.
    Falls back to network download only if local files don't exist.
    """
    model_id = _resolve_model_id(config)
    
    # Try local files first (avoids network timeouts on HPC)
    try:
        processor = InstructBlipProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=True,  # Use cached files to avoid network timeouts
        )
        return processor
    except Exception as local_error:
        # If local files don't exist, try downloading (with increased timeout)
        # This handles first-time setup when model isn't cached yet
        if "not found" in str(local_error).lower() or "does not exist" in str(local_error).lower():
            # Increase timeout for HuggingFace Hub requests (HPC networks can be slow)
            original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", None)
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
            try:
                processor = InstructBlipProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
                    local_files_only=False,
                )
                return processor
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to load processor. Local files error: {local_error}. "
                    f"Download error: {download_error}. "
                    f"Make sure the model is cached or network connectivity improves."
                ) from download_error
            finally:
                # Restore original timeout setting
                if original_timeout is not None:
                    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
                elif "HF_HUB_DOWNLOAD_TIMEOUT" in os.environ:
                    del os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]
        else:
            # Other errors (like missing dependencies) should be raised directly
            raise


def get_tokenizer_instructblip(config, cache_dir=None):
    return get_processor_instructblip(config, cache_dir=cache_dir).tokenizer


def preprocess_instructblip(images, prompts, processor, tokenize=False):
    """InstructBLIP preprocessing with truncation to avoid QFormer position overflow.
    
    QFormer has max_position_embeddings=512, so we truncate text to avoid:
    RuntimeError: The size of tensor a (N) must match the size of tensor b (512)
    """
    # Newer transformers may not populate num_query_tokens on the processor,
    # but the InstructBLIP model always uses 32 query tokens.
    if processor.num_query_tokens is None:
        processor.num_query_tokens = 32
    return processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512,
    )
