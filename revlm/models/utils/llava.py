def preprocess_llava(images, prompts, processor, tokenize=False):
    """Build batched chat-template inputs for LLaVA-style VLMs."""
    imgs = images if isinstance(images, list) else [images]
    prs = prompts if isinstance(prompts, list) else [prompts]
    convs = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": pr},
        ],
    }] for img, pr in zip(imgs, prs)]
    texts = processor.apply_chat_template(
        convs,
        tokenize=tokenize,
        add_generation_prompt=True,
    )
    return processor(text=texts, images=imgs, return_tensors="pt", padding=True)
