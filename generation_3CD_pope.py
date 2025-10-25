
import argparse
import os
import random
import sys
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
sys.path.append("./")
sys.path.append("../")
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
import json
from pycocotools.coco import COCO
from collections import defaultdict
import torch
from PIL import Image
from transformers import TextStreamer
import torch.nn.functional as F
import numpy as np
import json
from types import SimpleNamespace
from pycocotools.coco import COCO
from collections import defaultdict
from pope_metrics.utils import generate_ground_truth_objects, pope
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import requests, torch, os, cv2, re, glob
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.image as mpimg
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
from tqdm import tqdm
import copy, json
import argparse, random, pickle, time
import vlm_utils
from pope_loader import POPEDataSet
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
# from mplug_owl2.mm_utils import process_images
from minigpt4.common.registry import registry

print(torch.__version__)


MODEL_EVAL_CONFIG_PATH = {
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model_type")
    parser.add_argument("--model_path", type=str, help="model_path")
    parser.add_argument("--pope_type", type=str, help="")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coco",
        help="Name of the dataset. Default is 'coco'.",
    )
    parser.add_argument("--data_path",type=str,default="/home/donut2024/coco2014",help="data path",)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-m", "--max_new_tokens", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_result/",
        help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_false",
        dest="verbosity",
        default=True,
        help="Verbosity. Default: True.",
    )
    parser.add_argument(
        "--gt_seg_path",
        type=str,
        default="pope_coco/coco_ground_truth_segmentation.json",
        help="Input json file that contains ground truth objects in the image.",
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of positive/negative objects to be sampled. Default is 3.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to build POPE questions. Default is 500.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="",
    )

    parser.add_argument(
        "--question_template",
        type=str,
        default="Is there a {} in the image?",  # for llava-1.5
        help="Prompt template. Default is 'Is there a {} in the image?'.",
    )
    args = parser.parse_args()
    return args

#############################################3
def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))

    return acc, precision, recall, f1

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")
        if any(word in NEG_WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            pred_list.append(0)
        else:
            pred_list.append(1)

    return pred_list



#########################################################################################################

def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    cfg = Config(args)
    decoding_strategy = '3CD'
    seed = args.seed
    setup_seeds(seed)
    pope_type = args.pope_type
    device = (
        torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
    )
    model_name = args.model
    num_samples = args.num_samples
    num_images = args.num_images
    output_dir = args.output_dir
    num_beams = 1
    batch_size = 1
    max_new_tokens = args.max_new_tokens
    gt_seg_path = args.gt_seg_path
    question_template = args.question_template
    verbosity = args.verbosity


    #fp 16
    model_id = args.model_path
    model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
    output_attentions=True,
    output_hidden_states=True,
    return_dict_in_generate=True,
    ).to(0)


    # #fp 8
    # model_id = args.model_path
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=10.0,
    # )
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id,
    #     quantization_config=bnb_config,
    #     device_map={"": 0},                
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=False,
    #     output_attentions=True,
    #     output_hidden_states=True,
    #     return_dict_in_generate=True,
    # ).eval()


    print(1)
    model.config.use_cache = True
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # --- hotfix: some HF LLaVA processors miss patch_size ---
    ps = None
    # 1) 모델 설정에서 우선 시도
    if hasattr(model, "config") and hasattr(model.config, "vision_config"):
        ps = getattr(model.config.vision_config, "patch_size", None)
    # 2) 못 찾으면 LLaVA-1.5 기본값 14
    if ps is None:
        ps = 14
    setattr(processor, "patch_size", ps)
    if hasattr(processor, "image_processor"):
        setattr(processor.image_processor, "patch_size", ps)
    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        try:
            ip.do_resize = True
            ip.size = {"shortest_edge": 336}
            ip.do_center_crop = True
            ip.crop_size = {"height": 336, "width": 336}
        except Exception:
            pass
        
    print(2)

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )
    # print(vis_processors["eval"].transform)



    # generate pope questions
    question_dir = os.path.join(output_dir, "pope")
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    question_path = os.path.join(
        question_dir,
        f"_num_images_{num_images}_num_samples_{num_samples}_pope_{pope_type}_questions.json",
    )
    # load ground truth segmentation results.
    # Must include (other keys such as image_id can exist):
    # {"image": "COCO_val2014_000000131089.jpg", "objects": ["person", "baseball bat"]}
    segment_results = [json.loads(q) for q in open(gt_seg_path, "r")]
    # process segmentation ground truth
    processed_segment_results = []
    # Sample images which contain more than sample_num objects
    for cur_image in segment_results:
        if len(cur_image["objects"]) >= num_samples:
            processed_segment_results.append(cur_image)

    assert (
        len(processed_segment_results) >= num_images
    ), f"The number of images that contain more than {num_samples} objects is less than {num_images}."
    # Randomly sample num_images images
    processed_segment_results = random.sample(processed_segment_results, num_images)
    # Organize the ground truth objects and their co-occurring frequency
    question_name = f"_num_images_{num_images}_num_samples_{num_samples}"
    # ground truth object summary
    ground_truth_objects = generate_ground_truth_objects(
        processed_segment_results,
        question_dir,
        question_name,
        verbosity,
    )
    # Generate POPE questions and save to local file
    if pope_type is None:
        for cur_type in ["random", "popular", "adversarial"]:
            pope(
                ground_truth_objects=ground_truth_objects,
                segment_results=processed_segment_results,
                num_samples=num_samples,
                template=question_template,
                neg_strategy=cur_type,
                output_dir=question_dir,
                dataset_name=question_name,
                verbosity=verbosity,
            )
    else:
        pope(
            ground_truth_objects=ground_truth_objects,
            segment_results=processed_segment_results,
            num_samples=num_samples,
            template=question_template,
            neg_strategy=pope_type,
            output_dir=question_dir,
            dataset_name=question_name,
            verbosity=verbosity,
        )
    print(3)
    # load all the POPE questions
    all_pope_questions = [json.loads(q) for q in open(question_path, "r")]
    if verbosity:
        print(
            f"\nLoaded {len(all_pope_questions)} POPE questions from {question_path}."
        )
    # sanity check
    if len(all_pope_questions) != num_images * num_samples * 2:
        raise ValueError(
            f"Number of POPE questions loaded from {question_path} is not equal to {num_images * num_samples * 2}."
        )
    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=question_path, data_path=args.data_path, trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
        drop_last=False,
    )
    print("load data finished")
    base_dir = os.path.join(output_dir, "pope", args.model)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    print(4)
################################################################
################################################################
################################################################
    print(5)
    idx = 0
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        idx += 1
        image = data["image"]
        qu = data["query"][0]
        label = data["label"]
        image_path = data["image_path"] #여기서 리스트에 담긴 형태
        print(f'pope image_path : {image_path}')
        image_path = image_path[0]
        image = Image.open(image_path).convert("RGB")
        image_id = image_path.split("/")[-1].split(".")[0].split("_")[-1].lstrip("0")
        label_list = label_list + list(label)
        label = torch.Tensor(label).to(device)
        #################################

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qu},
                    {"type": "image"}  
                ],
            }
        ]
        qu = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        print(f'qu : {qu}')
        #####################################
        # 1.greedy decoding
        past_key_values = None  # 초기값
        inputs = processor(images=image, text=qu, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        vlm_utils.input_pixels_to_img(pixel_values, 'original', image_id, 'visualize/')
        generated = input_ids
        max_new_tokens = max_new_tokens

        # 전부 같은 길이!
        greedy_token_names = []
        greedy_highlighted_img = []
        greedy_logits = []
        greedy_entropys = []

        with torch.inference_mode():
            for token_step in range(1):
                input_this_step = generated if token_step == 0 else next_token_id
                outputs = model(
                    input_ids=input_this_step,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values if token_step == 0 else None,
                    past_key_values=past_key_values,
                    output_attentions=True,  
                    use_cache=True,
                    return_dict=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                next_token_name = processor.tokenizer.convert_ids_to_tokens(next_token_id)[0]
                next_token_attentions = outputs.attentions

                if token_step == 0:
                    next_token_blur_maps = []
                    for i, attn in enumerate(next_token_attentions):
                        h_map = attn.squeeze(0)[:, -1, 5:581].view(-1, 24, 24)
                        next_token_blur_maps.append(h_map)

                    next_token_blur_maps = torch.cat(next_token_blur_maps, dim=0)

                    maps_tensor = next_token_blur_maps
                    vars = maps_tensor.view(maps_tensor.size(0), -1).var(dim=1, unbiased=False)
                    sorted_indices = torch.argsort(vars, descending=True)
                    sorted_maps = maps_tensor[sorted_indices]
                    top_n = max(1, int(sorted_maps.size(0) * 0.1)) # !!!args.var
                    maps_top = sorted_maps[:top_n]
                    maps_top_filtered = vlm_utils.gaussian_filter_tensor_batch(maps_top, filter_type='gaussian', filter_size=25)
                    summed_map = maps_top_filtered.sum(dim=(0, 1))

                    threshold = summed_map.mean()
                    binary_mask = summed_map > threshold
                    binary_map = binary_mask.int()

                    margin_indices = torch.nonzero(~binary_mask, as_tuple=False)
                    margin_values = summed_map[margin_indices[:, 0], margin_indices[:, 1]]
                    margin_pos_list = [(int(y.item()), int(x.item()), float(v.item())) for (y, x), v in zip(margin_indices, margin_values)]
                    
                    highlighted_indices = torch.nonzero(binary_mask, as_tuple=False)
                    highlighted_values = summed_map[highlighted_indices[:, 0], highlighted_indices[:, 1]]
                    highlighted_pos_list = [(int(y.item()), int(x.item()), float(v.item())) for (y, x), v in zip(highlighted_indices, highlighted_values)]
                    
                    highlighted_img = vlm_utils.replace_patch_with_noise(pixel_values, margin_pos_list, noise_type='black', patch_size=14, mean=0.0, std=1.0)
                    margin_img = vlm_utils.replace_patch_with_noise(pixel_values, highlighted_pos_list, noise_type='black', patch_size=14, mean=0.0, std=1.0)
                    vlm_utils.input_pixels_to_img(highlighted_img, 'highlighted_img', image_id, 'visualize/')
                    vlm_utils.input_pixels_to_img(margin_img, 'margin_img', image_id, 'visualize/')

                    greedy_highlighted_img.append(highlighted_img)
                    greedy_logits.append(next_token_logits)
                    probs = F.softmax(next_token_logits, dim=-1)  # shape: (B, V)
                    log_probs = F.log_softmax(next_token_logits, dim=-1)  # shape: (B, V)
                    entropy = -(probs * log_probs).sum(dim=-1) # shape: (B,)
                    greedy_entropys.append(entropy.item())
                    greedy_token_names.append(next_token_name)
            
                if next_token_id.item() == processor.tokenizer.eos_token_id:
                    break
                generated = torch.cat((generated, next_token_id), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token_id)), dim=1)
                past_key_values = outputs.past_key_values

        greedy_output = processor.tokenizer.batch_decode(generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        GREEN = "\033[92m"  
        RED = "\033[91m"    
        RESET = "\033[0m"   

        text = greedy_output
        colored_text = text.replace("Yes", f"{GREEN}Yes{RESET}")
        colored_text = colored_text.replace("No", f"{RED}No{RESET}")

        print('-'*30)
        print(f'greedy: {colored_text}')

        #####################################
        # 3. contrastive decoding (no KV cache)
        # fix: 매 스텝마다 pixel_values를 다시 넣는다.

        model.eval()

        alpha = args.alpha   # margin 억제 강도
        beta  = args.beta   # highlighted 강화 강도

        # 디바이스/dtype 정렬 (안전장치)
        pixel_values    = pixel_values.to(model.device)
        highlighted_img = highlighted_img.to(model.device).to(pixel_values.dtype)
        margin_img      = margin_img.to(model.device).to(pixel_values.dtype)

        generated_cd = input_ids.clone()

        with torch.inference_mode():
            for token_step in range(max_new_tokens):
         
                attention_mask_cd = torch.ones_like(generated_cd)

                pix_base   = pixel_values
                pix_high   = highlighted_img
                pix_margin = margin_img

                # base
                logits_base = model(
                    input_ids=generated_cd,
                    attention_mask=attention_mask_cd,
                    pixel_values=pix_base,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]

                # highlighted
                logits_high = model(
                    input_ids=generated_cd,
                    attention_mask=attention_mask_cd,
                    pixel_values=pix_high,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]

                # margin
                logits_margin = model(
                    input_ids=generated_cd,
                    attention_mask=attention_mask_cd,
                    pixel_values=pix_margin,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]

                # 결합 로짓
                combined_logits = logits_base + beta * logits_high - alpha * logits_margin
                next_token_id_cd = torch.argmax(combined_logits, dim=-1, keepdim=True)

                if next_token_id_cd.item() == processor.tokenizer.eos_token_id:
                    break

                generated_cd = torch.cat((generated_cd, next_token_id_cd), dim=1)

        contrastive_output = processor.tokenizer.batch_decode(
            generated_cd[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        # 출력 색상 처리
        text = contrastive_output
        colored_text = text.replace("Yes", f"{GREEN}Yes{RESET}").replace("No", f"{RED}No{RESET}")
        print(f'cd: {colored_text}')
        # 라벨 출력
        if label.item() == 1:
            label = 'yes'
            print(f"label: {GREEN}Yes{RESET}")
        else:
            label = 'no'
            print(f"label: {RED}No{RESET}")
        
        out = "yes" if "yes" in contrastive_output.lower() else "no" if "no" in contrastive_output.lower() else None
        if out:
            if out == label:
                if out != greedy_output.lower():
                    print("\033[93mcorrected\033[0m")
                else:
                    pass
            else:
                if out != greedy_output.lower():
                    print("\033[94mcollapsed\033[0m")  # blue collapsed
        print('-' * 30)



        out = [contrastive_output]
        pred_list = recorder(out, pred_list)
        for line in out:
            print(line)

        output_text = out[0]
        cur_generated_answer = {
            "image_id": image_id,
            "question": " ".join(qu.split(" ")[2:]).split("?")[0] + "?",
            "answer": output_text,
        }


        # dump metric file
        generated_captions_path = os.path.join(
            base_dir,
            f"{pope_type}_{formatted_time}_{model_name}_{decoding_strategy}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_images}_generated_captions.json",
        )
        with open(generated_captions_path, "a") as f:
            json.dump(cur_generated_answer, f)
            f.write("\n")

   
    if len(pred_list) != 0:
        acc, precision, recall, f1 = print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        acc, precision, recall, f1 = print_acc(pred_list_s, label_list)

    result = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
    metrics_path = os.path.join(
        base_dir,
        f"{pope_type}_{formatted_time}_{model_name}_{decoding_strategy}_seed_{seed}_alpha_{alpha}_beta_{beta}_max_tokens_{max_new_tokens}_samples_{num_images}_results.json",
    )
    with open(metrics_path, "w") as f:
        json.dump(result, f)
        f.write("\n")


if __name__ == "__main__":
    main()
