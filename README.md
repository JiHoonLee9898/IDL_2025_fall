# Install
```bash
conda env create -f environment.yml
conda activate IDL
```

# Arguments
Refer to the example in [`run_eval.sh`](/run_eval.sh). 
`--model_path` is the absolute path of cloned https://huggingface.co/llava-hf/llava-1.5-7b-hf.
`--data_path` is `[COCO_DIR]`.

Note that `[COCO_DIR]` is expected to contain both images and annotation files within the annotations subfolder. In other words, `[COCO_DIR]` must follow the structure:

```plaintext
COCO_DIR (val2014 for example)
├── annotations
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   ├── person_keypoints_val2014.json
├── COCO_val2014_000000000042.jpg
├── COCO_val2014_000000000073.jpg
...
```

### POPE EVALUATION
The POPE evaluation results are saved alongside the path where the POPE captions are generated.
`--pope_type random --num_images 100 --seed 42` means, selecting 100 images from `[COCO_DIR]` and generate * 6 questions for each images (3 objects, exist/not exit, random object(refer to https://arxiv.org/abs/2305.10355)), under seed 42.  