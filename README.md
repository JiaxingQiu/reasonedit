## ReasonEdit: Editing Vision-Language Models using Human Reasoning

> *Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.*
> [[Paper]](https://arxiv.org/abs/2602.02408) [[Dataset]](DATAREADME.md)

### Requirements

```bash
git clone https://github.com/JiaxingQiu/reasonedit.git
cd reasonedit
conda create -n your_venv python=3.10 -y
conda activate your_venv
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Usage

#### (1) Topology-balanced multimodal embedding selection using Newman's Modularity Q.

Main dode are under `revlm/editors/auto_q/`.

```bash
python -m revlm.run.auto_layer --model_name qwen3_4b
python -m revlm.run.auto_scaler --model_name qwen3_4b
python -m revlm.run.bias_layer --model_name qwen3_4b
```

#### (2) Edit and evaluate a VLM with a chosen editor:

```bash
python -m revlm.run.edit --editor reasonedit --model_name qwen3_4b --dataset_name aokvqa
```

- Editors: `reasonedit`, `ike`, `ike_cot`, `grace`, `grace_cot`, `ft`, `mend`, `balancedit`, `baseline`
- Models: `qwen3`, `qwen3_4b`, `llava`, `blip`
- Datasets: `aokvqa`, `fvqa`

