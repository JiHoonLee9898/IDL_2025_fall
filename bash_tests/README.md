# Bash Test Scripts

This folder contains bash scripts to run POPE (Polling-based Object Probing Evaluation) tests using the configurations specified in the `eval_configs` folder.

## Available Scripts

### Individual Model Tests

- `run_llava_1.5_test.sh` - Test LLaVA-1.5 model
- `run_minigpt4_test.sh` - Test MiniGPT4 (Vicuna) model
- `run_minigpt4_llama2_test.sh` - Test MiniGPT4 (LLaMA2) model
- `run_mplug_owl2_test.sh` - Test mPLUG-Owl2 model

### Batch Test Scripts

- `run_all_pope_types.sh` - Run all POPE types (random, popular, adversarial) for a specific model
- `run_all_models.sh` - Run all models with a specific POPE type

## Usage

### Running a Single Model Test

```bash
cd bash_tests
chmod +x run_llava_1.5_test.sh
./run_llava_1.5_test.sh
```

### Customizing Parameters with Environment Variables

You can override default parameters using environment variables:

```bash
# Change POPE type
POPE_TYPE=adversarial ./run_llava_1.5_test.sh

# Change GPU ID
GPU_ID=1 ./run_llava_1.5_test.sh

# Change number of images
NUM_IMAGES=1000 ./run_llava_1.5_test.sh

# Combine multiple parameters
POPE_TYPE=popular NUM_IMAGES=1000 GPU_ID=2 ./run_llava_1.5_test.sh
```

### Running All POPE Types for a Model

```bash
chmod +x run_all_pope_types.sh
./run_all_pope_types.sh run_llava_1.5_test.sh
```

### Running All Models

```bash
chmod +x run_all_models.sh
./run_all_models.sh

# Or with a specific POPE type
POPE_TYPE=adversarial ./run_all_models.sh
```

## Available Environment Variables

All scripts support the following environment variables:

- `MODEL_PATH` - Path to the model checkpoint
- `DATA_PATH` - Path to COCO dataset (default: `/home/donut2024/coco2014`)
- `GPU_ID` - GPU device ID (default: `0`)
- `POPE_TYPE` - POPE evaluation type: `random`, `popular`, or `adversarial` (default: `random`)
- `NUM_IMAGES` - Number of images to evaluate (default: `500`)
- `NUM_SAMPLES` - Number of positive/negative samples (default: `3`)
- `MAX_NEW_TOKENS` - Maximum tokens to generate (default: `16`)
- `OUTPUT_DIR` - Output directory for results (default: `./paper_result/`)
- `GT_SEG_PATH` - Path to ground truth segmentation file (default: `pope_coco/coco_ground_truth_segmentation.json`)

## POPE Types

- **random** - Random negative samples
- **popular** - Popular object negative samples
- **adversarial** - Adversarial negative samples (most challenging)

## Making Scripts Executable

Before running the scripts, make them executable:

```bash
chmod +x bash_tests/*.sh
```

## Example Workflows

### Quick Test (3 images)
```bash
NUM_IMAGES=3 ./run_llava_1.5_test.sh
```

### Full Evaluation on Different GPU
```bash
GPU_ID=1 NUM_IMAGES=5000 ./run_llava_1.5_test.sh
```

### Complete Evaluation Suite
```bash
# Run all models with all POPE types
for script in run_llava_1.5_test.sh run_minigpt4_test.sh run_mplug_owl2_test.sh; do
    ./run_all_pope_types.sh "$script"
done
```

## Notes

- Make sure you have the required model checkpoints at the specified paths
- The COCO dataset should be available at the specified `DATA_PATH`
- Ground truth segmentation file must exist before running tests
- Results will be saved to the `OUTPUT_DIR` specified
