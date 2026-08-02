# Dataset

The benchmark dataset **RationaleVQA** (A-OKVQA + FVQA) is available on [HuggingFace](https://huggingface.co/datasets/JJoy333/RationaleVQA) and is downloaded automatically when running experiments.

**COCO images** must be downloaded separately following the instructions below.


## COCO Images

### A-OKVQA (COCO 2017)

```bash
export COCO_DIR=./data/images/aokvqa/
mkdir -p ${COCO_DIR}

for split in train val test; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}; rm "${split}2017.zip"
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip
```

### FVQA (COCO 2014 + ImageNet)

Download images from [FVQA](https://github.com/wangpengnorman/FVQA) and copy `new_dataset_release/images` to `data/images/fvqa/`.


## Generating Eval Images (optional)

Image generality and COE generality metrics require generated images. These are not included in the HF dataset due to size. To reproduce:

```bash
# Image generality (Stable Diffusion 3)
python -m revlm.run.i_gen --dataset_name aokvqa
python -m revlm.run.i_gen --dataset_name fvqa

# COE generality (counterfactual images)
python -m revlm.run.e_gen_image --dataset_name aokvqa --model_name qwen3_4b
```

Outputs: `data/related_image/` and `data/coe_gen_merge/image/`.

> If these images are not generated, the pipeline still runs — image/COE generality will report 0.
