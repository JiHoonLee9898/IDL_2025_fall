import sys
import os
import torch, spacy, requests, sys, os, cv2, re, glob, copy, json, argparse, random, pickle, time, shutil
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.image as mpimg
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from chair import CHAIR  
import re, spacy
import nltk
from nltk import word_tokenize, pos_tag
import vlm_utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
########################################################################################################


def select_random_img_list_from_cocodir(cocodir:str="/home/work/jihoon_wombat_storage/COCO_DIR",
                                        img_count:int=500):
    image_list = sorted(glob.glob(os.path.join(cocodir, "COCO_val2014_*.jpg")))
    image_list = [os.path.basename(p) for p in image_list]
    selected_image_path_list = random.sample(image_list, img_count)
    print(f"selected img list head: {[int(img_path.split('.jpg')[0].split('_')[-1]) for img_path in selected_image_path_list][:5]}")
    return selected_image_path_list

def get_top_k_tokens_vis(model, processor,
                         hidden_states, 
                         token_position:int=0, 
                         k:int=30, 
                         early_exit=False, 
                         layer_index = -1, 
                         batch_index = 0,
                         save_folder_path:str=""):
    hidden_vector = hidden_states[token_position][layer_index][batch_index][-1] 
    lm_head_weight = model.language_model.lm_head.weight  # shape: (vocab_size, hidden_size)
    logit = torch.matmul(hidden_vector, lm_head_weight.T) 
    probs = F.softmax(logit.float(), dim=-1)
    topk = torch.topk(probs, k=k)
    top_probs = topk.values
    top_indices = topk.indices
    top3_indices = top_indices[:3]
    tokens = [processor.tokenizer.decode([idx]) for idx in top_indices]
    plt.figure(figsize=(20, 6))
    bars = plt.bar(range(k), top_probs.cpu().detach().numpy())
    for j in range(3): bars[j].set_color('red')
    plt.xticks(range(k), tokens, rotation=90)
    plt.ylabel("Probability")
    plt.title(f"Top-{k} Token Probabilities (Last Token Output)")
    top_1 = tokens[0].replace('</s>', '|eos|')
    if top_1 == '|eos|': return 
    fig_path = os.path.join(save_folder_path, f'{token_position}_{top_1}.png')
    plt.savefig(fig_path, dpi=100)
    plt.close()


def get_base_img(inputs_pixel_values, 
                      input_img_path:str="",
                      save_folder_path:str="",
                      ):
    # save grid img 
    # return grid img path
    print("Image tokens shape:", inputs_pixel_values.shape)
    image_np = inputs_pixel_values.squeeze(0).permute(1, 2, 0).cpu().numpy() 
    image_np = ((image_np + 1) / 2) * 255  # [-1, 1] -> [0, 1] -> [0, 255]
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)  # 값의 범위를 0-255로 클리핑하고 정수로 변환
    image_np = np.ascontiguousarray(image_np)
    filename = input_img_path.split('.')[-2].split('/')[-1]
    input_img_processed_path = os.path.join(save_folder_path, f'{filename}_input_image.png')
    cv2.imwrite(input_img_processed_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))  # RGB를 BGR로 변환하여 저장
    print(f"이미지가 '{input_img_processed_path}'로 저장되었습니다.")


def get_grid_base_img_and_return_path(inputs_pixel_values, 
                      input_img_path:str="",
                      save_folder_path:str="",
                      patch_size:int=14,):
    # save grid img 
    # return grid img path
    print("Image tokens shape:", inputs_pixel_values.shape)
    image_np = inputs_pixel_values.squeeze(0).permute(1, 2, 0).cpu().numpy() 
    image_np = ((image_np + 1) / 2) * 255  # [-1, 1] -> [0, 1] -> [0, 255]
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)  # 값의 범위를 0-255로 클리핑하고 정수로 변환
    image_np = np.ascontiguousarray(image_np)
    image_np_nogrid = image_np.copy()
    for i in range(0, 336, patch_size):
        for j in range(0, 336, patch_size):
            # 가로선과 세로선 그리기
            cv2.line(image_np, (i, 0), (i, 336), (255, 0, 0), 1)  # 세로선
            cv2.line(image_np, (0, j), (336, j), (255, 0, 0), 1)  # 가로선
    filename = input_img_path.split('.')[-2].split('/')[-1]
    output_grid_img_path = os.path.join(save_folder_path, f'{filename}_output_image_with_grid.png')
    cv2.imwrite(output_grid_img_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))  # RGB를 BGR로 변환하여 저장
    print(f"이미지가 '{output_grid_img_path}'로 저장되었습니다.")
    return output_grid_img_path





def attvisualize(attention_map, 
                 base_grid_image_path, 
                 token_position,
                 token_name, 
                 save_path,
                 next_token_logits,
                 processor,
                 threshold):
    
    token_name = re.sub(r'[<>:/\\|?*"]', '|', str(token_name))  
    base_image = mpimg.imread(base_grid_image_path)
    h, w = base_image.shape[:2]

    # 소프트맥스 확률 계산 및 상위 10개 추출
    softmax_probs = torch.softmax(next_token_logits, dim=-1).squeeze()
    top_probs, top_indices = torch.topk(softmax_probs, k=10)
    top_tokens = [processor.tokenizer.decode([idx]) for idx in top_indices]
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 3개의 서브플롯

    # 왼쪽: 원본 이미지 위 어텐션 오버레이
    axes[0].imshow(base_image, extent=[0, w, h, 0])
    axes[0].imshow(attention_map, cmap='viridis', alpha=0.6, extent=[0, w, h, 0],
                   vmin=attention_map.min(), vmax=attention_map.max())
    axes[0].set_title('Overlay on Original Image')
    axes[0].axis('off')

    # 가운데: 흰 배경 위 어텐션
    white_bg = np.ones_like(base_image)
    axes[1].imshow(white_bg, extent=[0, w, h, 0])
    im = axes[1].imshow(attention_map, cmap='viridis', alpha=0.6, extent=[0, w, h, 0],
                        vmin=attention_map.min(), vmax=attention_map.max())
    axes[1].set_title('Overlay on White Background')
    axes[1].axis('off')

    # 텍스트 오버레이 (주의: 배경은 흰색)
    text_visualize_threshold = threshold
    att_h, att_w = attention_map.shape
    x_step = w / att_w
    y_step = h / att_h
    for i in range(att_h):
        for j in range(att_w):
            value = attention_map[i, j]
            if value >= text_visualize_threshold:
                x = j * x_step + x_step / 2
                y = i * y_step + y_step / 2
                axes[1].text(x, y, f'{value:.2f}', color='black', fontsize=4, ha='center', va='center')
                if value >= attention_map.max():
                    axes[1].text(x, y, f'M', color='red', fontsize=6, ha='center', va='center')

    # 오른쪽: softmax top-k 바 그래프
    axes[2].barh(range(10), top_probs.detach().cpu().numpy()[::-1], color='skyblue')
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels(top_tokens[::-1], fontsize=10)
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Probability')
    axes[2].set_title('Top-10 Next Token Predictions')

    # 공통 colorbar
    cbar = fig.colorbar(im, ax=axes[:2], shrink=0.8, location='right')
    cbar.set_label('Attention Score')

    output_filename = f'{save_path}/{token_position}_{token_name}_attvisualize.png'
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    plt.close()

def gaussian_filter(attention_map, filter_type='average', filter_size=23):
    if filter_type == 'average':
        return cv2.blur(attention_map.astype(np.float32), (filter_size, filter_size))
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(attention_map.astype(np.float32), (filter_size, filter_size), sigmaX=0)#sigmaX = 0이면 자동조절절
    
def gaussian_filter_tensor(attention_map: torch.Tensor, filter_type='average', filter_size=23):
    if attention_map.dim() != 2:
        raise ValueError("Input attention_map must be a 2D tensor (H x W)")
    attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    if filter_type == 'average':
        kernel = torch.ones((1, 1, filter_size, filter_size), dtype=attention_map.dtype, device=attention_map.device)
        kernel /= kernel.numel()
    elif filter_type == 'gaussian':
        # 1D Gaussian kernel
        def get_gaussian_kernel1d(kernel_size, sigma):
            x = torch.arange(kernel_size, dtype=attention_map.dtype, device=attention_map.device) - kernel_size // 2
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            return kernel_1d
        sigma = 0.3 * ((filter_size - 1) * 0.5 - 1) + 0.8  # OpenCV-like default
        gk1d = get_gaussian_kernel1d(filter_size, sigma)
        kernel = torch.matmul(gk1d[:, None], gk1d[None, :]).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    # padding = filter_size // 2 to maintain size
    filtered = F.conv2d(attention_map, kernel, padding=filter_size // 2)
    return filtered.squeeze(0).squeeze(0)  # Back to [H, W]

# def gaussian_filter_tensor_batch(attention_maps: torch.Tensor, filter_type='average', filter_size=23):
#     """
#     attention_maps: Tensor of shape [B, H, W]
#     Returns: Tensor of shape [B, H, W] after filtering
#     """
#     if attention_maps.dim() != 3:
#         raise ValueError("Input must be a 3D tensor of shape [B, H, W]")

#     B, H, W = attention_maps.shape
#     attention_maps = attention_maps.unsqueeze(1)  # [B, 1, H, W]

#     if filter_type == 'average':
#         kernel = torch.ones((1, 1, filter_size, filter_size), dtype=attention_maps.dtype, device=attention_maps.device)
#         kernel /= kernel.numel()
#     elif filter_type == 'gaussian':
#         def get_gaussian_kernel1d(kernel_size, sigma):
#             x = torch.arange(kernel_size, dtype=attention_maps.dtype, device=attention_maps.device) - kernel_size // 2
#             kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
#             kernel_1d /= kernel_1d.sum()
#             return kernel_1d
#         sigma = 0.3 * ((filter_size - 1) * 0.5 - 1) + 0.8
#         gk1d = get_gaussian_kernel1d(filter_size, sigma)
#         kernel = torch.matmul(gk1d[:, None], gk1d[None, :])  # [K, K]
#         kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
#     else:
#         raise ValueError(f"Unknown filter_type: {filter_type}")

#     # expand kernel: [B, 1, K, K]
#     kernel = kernel.expand(B, 1, filter_size, filter_size)
#     # expand attention_maps: [B, B, H, W]
#     attention_maps = attention_maps.expand(B, B, H, W)

#     # grouped convolution
#     filtered = F.conv2d(attention_maps, kernel, padding=filter_size // 2, groups=B)
#     return filtered.squeeze(1)  # [B, H, W]


def gaussian_filter_tensor_batch(attention_maps: torch.Tensor, filter_type='average', filter_size=24+1):
    """
    attention_maps: Tensor of shape [B, H, W]
    Returns: Tensor of shape [B, H, W] after filtering
    """
    if attention_maps.dim() != 3:
        raise ValueError("Input must be a 3D tensor of shape [B, H, W]")

    B, H, W = attention_maps.shape
    attention_maps = attention_maps.unsqueeze(1)  # [B, 1, H, W]

    if filter_type == 'average':
        kernel = torch.ones((1, 1, filter_size, filter_size), dtype=attention_maps.dtype, device=attention_maps.device)
        kernel /= kernel.numel()
    elif filter_type == 'gaussian':
        def get_gaussian_kernel1d(kernel_size, sigma):
            x = torch.arange(kernel_size, dtype=attention_maps.dtype, device=attention_maps.device) - kernel_size // 2
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            return kernel_1d
        sigma = 0.3 * ((filter_size - 1) * 0.5 - 1) + 0.8
        gk1d = get_gaussian_kernel1d(filter_size, sigma)
        kernel = torch.matmul(gk1d[:, None], gk1d[None, :])  # [K, K]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Expand kernel to [B, 1, K, K] and attention_maps to [B, B, H, W]
    kernel = kernel.expand(B, 1, filter_size, filter_size)  # [B, 1, K, K]
    attention_maps = attention_maps.expand(B, B, H, W)      # [B, B, H, W]

    # Perform grouped convolution
    filtered = F.conv2d(attention_maps, kernel, padding=filter_size // 2, groups=B)  # [B, 1, H, W]
    return filtered.squeeze(1)  # [B, H, W]
    
def get_token_attmap_from_allattentions(processor,
                                        inputs_inputs_ids,
                                        outputs_sequences,
                                        attentions,
                                        vistion_patch_start_index=5,
                                        vision_patch_end_index=580,
                                        input_img_path:str="",
                                        gaussian_filter_size=3,
                                        ):
    # a list of tuple(attention_map, (token_step, layer, head), now_token, max_pos_list)
    allattmaps = []
    input_token_names = processor.tokenizer.convert_ids_to_tokens(inputs_inputs_ids[0], skip_special_tokens=False)
    output_token_names = processor.tokenizer.convert_ids_to_tokens(outputs_sequences[0], skip_special_tokens=False)
    for token_step in range(len(attentions)):
        now_token_attentions = attentions[token_step]
        input_token_count = len(inputs_inputs_ids[0])
        now_token = output_token_names[input_token_count:][token_step]
        for layer in range(len(now_token_attentions)):
            now_attention_h_maps = now_token_attentions[layer].squeeze(0)[:, -1, :]
            now_attention_h_maps_image_only = now_attention_h_maps[:, vistion_patch_start_index:vision_patch_end_index+1]
            for head in range(now_attention_h_maps_image_only.shape[0]):
                attention_scores = now_attention_h_maps_image_only[head]
                patch_count = int((vision_patch_end_index-vistion_patch_start_index+1)**0.5)
                attention_map = attention_scores.view(patch_count, patch_count).cpu().detach().numpy()
             
                attention_map = gaussian_filter(attention_map, filter_type='gaussian', filter_size=gaussian_filter_size)
                # max_pos = np.unravel_index(np.argmax(attention_map), attention_map.shape)
                allattmaps.append((attention_map, (token_step, layer, head), now_token))
    
    return allattmaps

def find_mscoco_words(sentence: str, 
                      cache_path: str="/home/work/jihoon_wombat_storage/CODES/chair.pkl"):
    result_dict = {}
    evaluator = pickle.load(open(cache_path, 'rb'))
    words, node_words, idxs, raw_words = evaluator.caption_to_words(sentence)
    gt_objects = set()
    for word, node_word, idx in zip(words, node_words, idxs):
        if node_word not in result_dict: result_dict[node_word] = 1
        else: result_dict[node_word] += 1
    return result_dict 




from collections import Counter

def sort_object_dict(greedy_object_dict, new_object_dict):
    # input style: {'dining table': 4, 'chair': 2, 'cup': 2, 'wine glass': 2, 'microwave': 1, 'refrigerator': 1, 'bowl': 1, 'vase': 1, 'bottle': 1}
    # output style: (sorted dict, sorted dict )
    sorted_dict1 = {key: greedy_object_dict[key] for key in greedy_object_dict}
    sorted_dict2 = {key: new_object_dict[key] for key in new_object_dict}
    return (sorted_dict1, sorted_dict2)

def load_chair_evaluator(cache_path, coco_annotations_path = '/home/work/jihoon_wombat_storage/COCO_DIR/annotations'):

    if os.path.exists(cache_path):
        evaluator = pickle.load(open(cache_path, 'rb'))
        print(f"loaded evaluator from cache: {cache_path}")
    else:
        print(f"cache not setted or not exist yet, building from scratch...\nIf cache exists, this process is unnecessary.\n-------------")
        evaluator = CHAIR(coco_annotations_path)
        pickle.dump(evaluator, open(cache_path, 'wb'))
        print(f"cached evaluator to: {cache_path}")

    return pickle.load(open(cache_path, "rb"))

def evaluate_caption_chair_i_and_objects(caption: str, image_id: int, evaluator: CHAIR):
    """
    단일 캡션과 이미지 ID에 대해 CHAIR-i 점수, 생성 객체 빈도, Recall 점수를 반환합니다.

    Args:
        caption (str): 생성된 캡션
        image_id (int): 해당 이미지 ID (COCO 기준)
        evaluator (CHAIR): 초기화된 CHAIR 평가 객체

    Returns:
        chair_i (float): CHAIR-i 점수 (hallucinated object 비율)
        object_counts (dict[str, int]): 생성된 대표 객체 빈도 정보
        recall (float): 생성된 객체 중 정답 객체를 얼마나 커버했는지 비율
    """

    # 캡션 분석
    words, node_words, idxs, raw_words = evaluator.caption_to_words(caption)

    # 정답 객체 (GT)
    gt_objects = evaluator.imid_to_objects[image_id]

    # CHAIR-i 계산
    hallucinated_word_count = sum(1 for node_word in node_words if node_word not in gt_objects)
    total_mscoco_words = len(words)
    chair_i = hallucinated_word_count / total_mscoco_words if total_mscoco_words > 0 else 0.0

    # 생성 객체 빈도 계산
    object_counts = {}
    for obj in node_words:
        object_counts[obj] = object_counts.get(obj, 0) + 1

    # Recall 계산
    recall_objects = set(obj for obj in node_words if obj in gt_objects)
    recall = len(recall_objects) / len(gt_objects) if len(gt_objects) > 0 else 0.0

    return chair_i, object_counts, recall


def early_exit(model, processor,
               output, layer_index):
    hidden_vector = output['hidden_states'][-1][layer_index][0][-1]  # shape: (hidden_size,))
    lm_head_weight = model.language_model.lm_head.weight  # shape: (vocab_size, hidden_size)
    logit = torch.matmul(hidden_vector, lm_head_weight.T) 
    probs = F.softmax(logit.float(), dim=-1)
    return probs

def get_jsd(p: torch.Tensor, q: torch.Tensor, base: float = torch.e) -> torch.Tensor:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl_div(a, b):
        mask = a > 0
        return torch.sum(a[mask] * torch.log(a[mask] / b[mask])) / torch.log(torch.tensor(base))

    jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return jsd


def load_pos_tagger(tagger_name='en_core_web_sm'):
    return spacy.load(tagger_name)

def get_last_pos_tag(text, tagger):
    doc = tagger(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    last_pos_tag_info = pos_tags[-1]
    last_pos_tag = last_pos_tag_info[1]
    return last_pos_tag

def get_pos_tag(text, tagger):
    doc = tagger(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def get_noun_token_lists(tokens, tagger):
    sentence = "".join(tokens).replace('▁', " ").replace('</s>', "").replace('<0x0A><0x0A>', " ")
    # print(sentence)
    pos_tags = get_pos_tag(sentence, tagger)
    noun_words = {word for word, tag in pos_tags if tag.startswith('NOUN')}
    # print(noun_words)
    # 5. 명사 단어를 구성하는 토큰들 추출
    noun_token_lists = []
    current_word = ""
    current_tokens = []
    idx_list = []

    for idx, token in enumerate(tokens):
        if token.startswith("▁"):
            
            if current_tokens:
                # 완성된 단어가 명사면 저장
                if current_word in noun_words:
                    noun_token_lists.append(current_tokens)
                # 초기화
                current_word = token[1:]
                current_tokens = [token]
              
            else:
                current_word = token[1:]
                current_tokens = [token]
                
        else:
            sep_token = False
            for sep in [',', '.', '<0x0A>']:
                if sep in token: 
                    sep_token = True
                    break
            if not sep_token:
                current_word += token
                current_tokens.append(token)
          

        # print(token,current_word)
    # 마지막 단어 확인
    if current_tokens and current_word in noun_words:
        noun_token_lists.append(current_tokens)

    flattened = [token for sublist in noun_token_lists for token in sublist]
    unique_tokens = list(set(flattened))
    prefix_noun_tokens = [token for token in unique_tokens if token.startswith("▁")]
    # print(prefix_noun_tokens)
    return prefix_noun_tokens

def get_noun_token_lists_2(tokens, tagger):
    sentence = "".join(tokens).replace('▁', " ").replace('</s>', "").replace('<0x0A><0x0A>', " ")
    pos_tags = get_pos_tag(sentence, tagger)
    noun_words = {word for word, tag in pos_tags if tag.startswith('NOUN')}

    noun_token_lists = []
    current_word = ""
    current_tokens = []
    current_indices = []

    for idx, token in enumerate(tokens):
        if token.startswith("▁"):
            if current_tokens:
                if current_word in noun_words:
                    noun_token_lists.append((current_tokens, current_indices))
                current_word = token[1:]
                current_tokens = [token]
                current_indices = [idx]
            else:
                current_word = token[1:]
                current_tokens = [token]
                current_indices = [idx]
        else:
            sep_token = False
            for sep in [',', '.', '<0x0A>']:
                if sep in token:
                    sep_token = True
                    break
            if not sep_token:
                current_word += token
                current_tokens.append(token)
                current_indices.append(idx)

    if current_tokens and current_word in noun_words:
        noun_token_lists.append((current_tokens, current_indices))

    flattened = [(token, idx) for token_list, idx_list in noun_token_lists for token, idx in zip(token_list, idx_list)]
    unique_prefix_noun_tokens = []
    positions = []
    seen = set()

    for token, idx in flattened:
        if token.startswith("▁") and token not in seen:
            unique_prefix_noun_tokens.append(token)
            positions.append(idx)
            seen.add(token)

    return unique_prefix_noun_tokens, positions


def get_gt_token_lists(tokens, image_id, evaluator):
    # 1. gt_objects: ['red sink', 'bottle'] → 하나의 문자열로 병합
    gt_objects = evaluator.imid_to_objects[image_id]
    gt_text = ' '.join(gt_objects).lower()

    # 2. GT 단어로부터 토큰을 재구성
    gt_token_lists = []
    current_word = ""
    current_tokens = []

    for token in tokens:
        if token == '</s>':
            continue  # 종료 토큰 무시

        # 3. 토큰이 단어 시작이면 이전 단어 저장
        if token.startswith("▁"):
            if current_tokens:
                # 완성된 단어가 gt_text에 포함되어 있으면 저장
                if current_word in gt_text:
                    gt_token_lists.append(current_tokens)
                # 초기화
            current_word = token[1:].lower()
            current_tokens = [token]
        else:
            current_word += token.lower()
            current_tokens.append(token)

    # 4. 마지막 단어도 확인
    if current_tokens and current_word in gt_text:
        gt_token_lists.append(current_tokens)

    # 5. 중복 제거 및 flatten
    flattened = [token for sublist in gt_token_lists for token in sublist]
    unique_tokens = list(set(flattened))
    prefix_gt_tokens = [token for token in unique_tokens if token.startswith("▁")]

    return prefix_gt_tokens

# old
# def get_hallucinated_prefix_tokens(tokens: list[str], image_id: int, evaluator: CHAIR) -> list[str]:
#     """
#     토큰 리스트와 이미지 ID를 기반으로 hallucinated 단어들의 prefix 토큰(▁로 시작하는)을 반환합니다.
#     """
#     caption = ''.join(tokens).replace('▁', ' ').strip()

#     # CHAIR 평가
#     chair_i, object_counts, recall = evaluate_caption_chair_i_and_objects(caption, image_id, evaluator)
#     words, node_words, idxs, raw_words = evaluator.caption_to_words(caption)
#     print(f'words {words}')
#     print(f'node_words {node_words}')
#     all_word = list(set(words + node_words))
#     gt_objects = evaluator.imid_to_objects[image_id]
#     hallucinated_words = [word.lower() for word in node_words if word not in gt_objects]

#     print(f'gt: {gt_objects}')
#     print(f'hal: {hallucinated_words}')

#     prefix_tokens = []
#     i = 0
#     while i < len(tokens):
#         token = tokens[i]
#         if token.startswith("▁"):
#             current_tokens = [token]
#             current_word = token[1:].lower()
#             i += 1
#             while i < len(tokens) and not tokens[i].startswith("▁"):
#                 current_word += tokens[i].lower()
#                 current_tokens.append(tokens[i])
#                 i += 1

#             # 로그 추가
#             # print(f"checking word: {current_word}, tokens: {current_tokens}")

#             # 완전 일치 대신 포함 검사
#             if any(hw in current_word for hw in hallucinated_words):
#                 # print(f"  matched hallucinated: {current_word}")
#                 prefix_tokens.extend([t for t in current_tokens if t.startswith("▁")])
#         else:
#             i += 1

#     return prefix_tokens

import re

def is_loose_match(candidate: str, hallucinated_words: list[str]) -> bool:
    candidate = candidate.lower()
    candidate = re.sub(r'[^\w\s]', '', candidate).strip()

    for hw in hallucinated_words:
        if candidate == hw:
            return True
        if candidate.endswith('s') and candidate[:-1] == hw:
            return True
        if candidate.replace(' ', '') == hw.replace(' ', ''):
            return True
    return False


def is_loose_match_2(candidate: str, hallucinated_words: list[str]) -> bool:
    candidate = candidate.lower()
    candidate = re.sub(r'[^\w\s]', '', candidate).strip()

    for hw in hallucinated_words:
        if candidate == hw:
            return hw
        if candidate.endswith('s') and candidate[:-1] == hw:
            return hw
        if candidate.replace(' ', '') == hw.replace(' ', ''):
            return hw
    return False


def get_hallucinated_node_words(caption: str, image_id: int, evaluator: CHAIR) -> list[tuple[str, str]]:
    """
    CHAIR evaluator를 활용해 hallucinated node_words 및 해당 동의어를 추출합니다.

    Args:
        caption (str): 자연어 캡션
        image_id (int): COCO 이미지 ID
        evaluator (CHAIR): 평가기 객체

    Returns:
        list[tuple[str, str]]: hallucinated된 (node_word, 동의어 단어) 리스트
    """
    node_syn_pairs = evaluator.process_sentence_get_coco_synonyms(caption)  # (node_word, synonym)
    gt_objects = evaluator.imid_to_objects[image_id]

    hallucinated = [(node_word, synonym) for (node_word, synonym) in node_syn_pairs if node_word not in gt_objects]
    return hallucinated

def get_gt_node_words(caption: str, image_id: int, evaluator: CHAIR) -> list[tuple[str, str]]:
    """
    CHAIR evaluator를 활용해 hallucinated node_words 및 해당 동의어를 추출합니다.

    Args:
        caption (str): 자연어 캡션
        image_id (int): COCO 이미지 ID
        evaluator (CHAIR): 평가기 객체

    Returns:
        list[tuple[str, str]]: hallucinated된 (node_word, 동의어 단어) 리스트
    """
    node_syn_pairs = evaluator.process_sentence_get_coco_synonyms(caption)  # (node_word, synonym)
    gt_objects = evaluator.imid_to_objects[image_id]

    gt = [(node_word, synonym) for (node_word, synonym) in node_syn_pairs if node_word in gt_objects]
    return gt

def get_existing_prefix_tokens(hallucinated_token: str, tokenizer_vocab: set) -> list[str]:
    """
    hallucinated token (e.g., '▁fork')에 대해 가능한 모든 prefix 중
    실제 vocab에 존재하는 토큰들만 반환

    Args:
        hallucinated_token (str): e.g., '▁fork'
        tokenizer_vocab (set): tokenizer.get_vocab().keys()

    Returns:
        List[str]: e.g., ['▁f', '▁fo', '▁for', '▁fork']
    """
    if not hallucinated_token.startswith("▁"):
        return []

    base = hallucinated_token[1:]
    prefixes = []
    for i in range(1, len(base) + 1):
        sub = '▁' + base[:i]
        if sub in tokenizer_vocab:
            prefixes.append(sub)
    return prefixes



def get_hallucinated_prefix_tokens(tokens: list[str], image_id: int, evaluator: CHAIR, vocab_list, bpe_true=True) -> list[str]:
    caption = ''.join(tokens).replace('▁', ' ').strip()

    # ✅ hallucinated node_words 가져오기 (CHAIR에서 추출)
    hallucinated_words = get_hallucinated_node_words(caption, image_id, evaluator)
    hallucinated_node_words = [node_word for node_word, synonym in hallucinated_words]
    hallucinated_synonym_words = [synonym for node_word, synonym in hallucinated_words]

    hallucinated_words = list(set(hallucinated_node_words + hallucinated_synonym_words))
    print(f"\nhallucinated_words: {hallucinated_words}")

    # token을 단어 단위로 분할
    word_token_groups = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("▁"):
            current_tokens = [tokens[i]]
            current_word = tokens[i][1:].lower()
            i += 1
            while i < len(tokens) and not tokens[i].startswith("▁"):
                current_word += tokens[i].lower()
                current_tokens.append(tokens[i])
                i += 1
            word_token_groups.append((current_word, current_tokens))
        else:
            i += 1

    prefix_tokens = []
    max_n = 3
    for n in range(1, max_n + 1):
        for i in range(len(word_token_groups) - n + 1):
            group = word_token_groups[i:i+n]
            full_phrase = ' '.join([w for w, _ in group])
            full_phrase = re.sub(r'[^\w\s]', '', full_phrase).strip().lower()
            # print(f"Trying: {full_phrase}")
            tokens_in_phrase = [t for _, toks in group for t in toks]

            if is_loose_match(full_phrase, hallucinated_words):
                prefix_tokens.extend([t for t in tokens_in_phrase if t.startswith("▁")])

    ###################### 
    if bpe_true:
        prefixes_extend_BPE = []
        for token in prefix_tokens:  # ['▁fork', '▁phone']
            prefixes_extend_BPE.extend(get_existing_prefix_tokens(token, vocab_list))
        prefixes_extend_BPE = list(set(prefixes_extend_BPE))
        print(f"all BPE: {prefixes_extend_BPE}")
        return prefixes_extend_BPE
    
    else:
        print(f"all hallucinated: {prefix_tokens}")
        return prefix_tokens
    ######################
    
def get_all_BPE(prefix_tokens, vocab_list):
    prefixes_extend_BPE = []
    for token in prefix_tokens:  # ['▁fork', '▁phone']
        prefixes_extend_BPE.extend(get_existing_prefix_tokens(token, vocab_list))
    prefixes_extend_BPE = list(set(prefixes_extend_BPE))
    print(f"all BPE: {prefixes_extend_BPE}")
    return prefixes_extend_BPE

# def get_object_prefix_tokens(tokens: list[str], image_id: int, evaluator: CHAIR, vocab_list, bpe_true=True) -> list[str]:
#     caption = ''.join(tokens).replace('▁', ' ').strip()

#     # ✅ hallucinated node_words 가져오기 (CHAIR에서 추출)
#     hallucinated_words = get_hallucinated_node_words(caption, image_id, evaluator)
#     gt_words = get_gt_node_words(caption, image_id, evaluator)


#     hallucinated_synonym_words = [synonym for node_word, synonym in hallucinated_words]
#     gt_synonym_words = [synonym for node_word, synonym in gt_words]


#     object_words = list(set(hallucinated_synonym_words + gt_synonym_words))
#     print(f"\ngt_words: {gt_synonym_words}")
#     print(f"\nhal_words: {hallucinated_synonym_words}")
#     print(f"\nobject_words: {object_words}")

#     # token을 단어 단위로 분할
#     word_token_groups = []
#     i = 0
#     while i < len(tokens):
#         if tokens[i].startswith("▁"):
#             current_tokens = [tokens[i]]
#             current_word = tokens[i][1:].lower()
#             i += 1
#             while i < len(tokens) and not tokens[i].startswith("▁"):
#                 current_word += tokens[i].lower()
#                 current_tokens.append(tokens[i])
#                 i += 1
#             word_token_groups.append((current_word, current_tokens))
#         else:
#             i += 1

#     prefix_tokens = []
#     max_n = 3
#     for n in range(1, max_n + 1):
#         for i in range(len(word_token_groups) - n + 1):
#             group = word_token_groups[i:i+n]
#             full_phrase = ' '.join([w for w, _ in group])
#             full_phrase = re.sub(r'[^\w\s]', '', full_phrase).strip().lower()
#             # print(f"Trying: {full_phrase}")
#             tokens_in_phrase = [t for _, toks in group for t in toks]

#             if is_loose_match(full_phrase, object_words):
#                 prefix_tokens.extend([t for t in tokens_in_phrase if t.startswith("▁")])

#     ###################### 
#     if bpe_true:
#         prefixes_extend_BPE = []
#         for token in prefix_tokens:  # ['▁fork', '▁phone']
#             prefixes_extend_BPE.extend(get_existing_prefix_tokens(token, vocab_list))
#         prefixes_extend_BPE = list(set(prefixes_extend_BPE))
#         print(f"all BPE: {prefixes_extend_BPE}")
#         return prefixes_extend_BPE
    
#     else:
#         print(f"all object prefix: {prefix_tokens}")
#         return prefix_tokens
#     ######################


def get_node_name(caption, evaluator: CHAIR) -> list[str]:
    node_syn_pairs = evaluator.process_sentence_get_coco_synonyms(caption)
    return node_syn_pairs
  


def get_object_prefix_tokens(tokens: list[str], image_id: int, evaluator: CHAIR, vocab_list, bpe_true=True) -> list[str]:
    caption = ''.join(tokens).replace('▁', ' ').strip()

    # ✅ hallucinated node_words 가져오기 (CHAIR에서 추출)
    hallucinated_words = get_hallucinated_node_words(caption, image_id, evaluator)
    gt_words = get_gt_node_words(caption, image_id, evaluator)


    hallucinated_synonym_words = [synonym for node_word, synonym in hallucinated_words]
    gt_synonym_words = [synonym for node_word, synonym in gt_words]


    object_words = list(set(hallucinated_synonym_words + gt_synonym_words))
    print(f"\ngt_words: {gt_synonym_words}")
    print(f"\nhal_words: {hallucinated_synonym_words}")
    print(f"\nobject_words: {object_words}")

    # token을 단어 단위로 분할
    word_token_groups = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("▁"):
            current_tokens = [tokens[i]]
            current_word = tokens[i][1:].lower()
            i += 1
            while i < len(tokens) and not tokens[i].startswith("▁"):
                current_word += tokens[i].lower()
                current_tokens.append(tokens[i])
                i += 1
            word_token_groups.append((current_word, current_tokens))
        else:
            i += 1

    prefix_tokens = []
    max_n = 3
    for n in range(1, max_n + 1):
        for i in range(len(word_token_groups) - n + 1):
            group = word_token_groups[i:i+n]
            full_phrase = ' '.join([w for w, _ in group])
            full_phrase = re.sub(r'[^\w\s]', '', full_phrase).strip().lower()
            # print(f"Trying: {full_phrase}")
            tokens_in_phrase = [t for _, toks in group for t in toks]

            object_name = is_loose_match_2(full_phrase, object_words)
            if object_name != False :
                prefix_tokens.append(([t for t in tokens_in_phrase if t.startswith("▁")][0], object_name))

  
    print(f"all object prefix: {prefix_tokens}")
    return prefix_tokens
    ######################



def get_top_k_most_tag(top_k_token_names, tagger):
    result = {'NOUN_like':0}
    for token in top_k_token_names:
        doc = tagger(token)
        token_pos_tag = [token.pos_ for token in doc]
        if len(token_pos_tag)==1:
            now_tag = token_pos_tag[0]
            if now_tag in ['NOUN'] : now_tag = 'NOUN_like'
            if now_tag not in result: result[now_tag] = 1
            else: result[now_tag] += 1
    max_pos = max(result, key=lambda k: result[k])
    return max_pos

def get_object_token_position_by_postag(tokenlist, check_postag_list):
    # tokenlist = [('▁The', 0), ('▁image', 1), ('▁dep', 2), ...] 
    # check_postag_list = ['NOUN', ...]
    tokens = tokenlist
    check_postag_list = ['NOUN']
    # 1. 문자열로 이어붙이고 ▁를 공백으로 대체
    indexed_tokens = [(token.replace('▁', ' '), idx) for token, idx in tokens]
    joined_text = ''.join([tok for tok, _ in indexed_tokens])
    cleaned_text = joined_text.replace('<0x0A>', '\n').strip()
    # 2. 단어별 시작 인덱스 매핑 (단어: 해당 토큰 인덱스 리스트)
    word_token_map = {}
    current_word = ""
    current_indices = []
    for token, idx in tokens:
        if token.startswith('▁') or re.match(r'[^a-zA-Z0-9]', token):  # 새 단어의 시작이거나 특수기호
            if current_word:
                word_token_map[current_word] = current_indices
            # 새 단어 시작
            current_word = token.lstrip('▁')
            current_indices = [idx]
        else:
            current_word += token
            current_indices.append(idx)
    # 마지막 단어 추가
    if current_word:
        word_token_map[current_word] = current_indices
    # 3. POS 태깅
    tagger = vlm_utils.load_pos_tagger()
    pos_tags = vlm_utils.get_pos_tag(cleaned_text, tagger)
    noun_indices = []
    for word, tag in pos_tags:
        for pos in check_postag_list:
            if tag.startswith(pos):  
                indices = word_token_map.get(word)
                if indices:
                    noun_indices.append((word, indices))
    return sorted(noun_indices, key=lambda x:x[1])

def replace_patch_with_noise(pixel_tensor, 
                             position_tuples, 
                             patch_size=14, 
                             mean=0.0, 
                             std=1.0,
                             noise_type='gaussian'):
    pixel_tensor = pixel_tensor.clone() # 이미지 복사 (원본 보존)
    for position_tuple in position_tuples:
        top = position_tuple[0] * patch_size
        left = position_tuple[1] * patch_size
        if noise_type == 'gaussian':
            noise_patch = torch.randn((3, patch_size, patch_size)) * std + mean # 가우시안 노이즈 생성 (3채널)
        elif noise_type == 'black':
            noise_patch = torch.zeros((3, patch_size, patch_size))
        pixel_tensor[0, :, top:top+patch_size, left:left+patch_size] = noise_patch # 해당 위치에 노이즈 삽입
    return pixel_tensor

def replace_except_patch_with_noise(pixel_tensor, 
                                    position_tuples, 
                                    patch_size=14, 
                                    mean=0.0, 
                                    std=1.0,
                                    noise_type='gaussian'):
    pixel_tensor = pixel_tensor.clone()  # 원본 보존
    noisy_tensor = pixel_tensor.clone()

    # 전체 영역에 노이즈 입히기
    if noise_type == 'gaussian':
        noise = torch.randn_like(noisy_tensor) * std + mean
    elif noise_type == 'black':
        noise = torch.zeros_like(noisy_tensor)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    noisy_tensor = noise

    # 지정된 패치 영역만 원본으로 되돌리기
    for position_tuple in position_tuples:
        top = position_tuple[0] * patch_size
        left = position_tuple[1] * patch_size
        noisy_tensor[0, :, top:top+patch_size, left:left+patch_size] = pixel_tensor[0, :, top:top+patch_size, left:left+patch_size]

    return noisy_tensor


def upscale_highlighted_patch(pixel_tensor, position_tuples, patch_size=14):
    pixel_tensor = pixel_tensor.clone() # 원본 보존
    # position_tuples로부터 최소, 최대 x, y 찾기
    x_positions = [pos[1] for pos in position_tuples]
    y_positions = [pos[0] for pos in position_tuples]
    
    min_x = min(x_positions) * patch_size
    max_x = (max(x_positions) + 1) * patch_size
    min_y = min(y_positions) * patch_size
    max_y = (max(y_positions) + 1) * patch_size
    
    # 박스 내 픽셀 crop
    cropped = pixel_tensor[:, :, min_y:max_y, min_x:max_x]  # [batch, channel, height, width]
    
    # 크롭한 영역을 원본 pixel_tensor 크기로 리사이즈
    resized = F.interpolate(cropped, size=pixel_tensor.shape[2:], mode='bilinear', align_corners=False)
    
    return resized
def input_pixels_to_img(pixel_tensor, token_index, token_name, save_path):
    token_name = re.sub(r'[<>:/\\|?*"]', '|', str(token_name))  
    tensor = pixel_tensor.squeeze(0) # 1. 배치 차원 제거: [1, 2or3, 336, 336] → [2or3, 336, 336]
    np_img = tensor.permute(1, 2, 0).cpu().numpy() # 2. 텐서를 numpy로 변환하고, 채널 순서 바꾸기: [2or3, H, W] → [H, W, 2or3]
    np_img = (np_img * 255).clip(0, 255).astype(np.uint8) # 3. 0~255 범위로 변환 (uint8로 타입 변경)
    img = Image.fromarray(np_img) # 4. PIL 이미지로 변환 후 저장
    img.save(os.path.join(save_path, f"{token_index}_{token_name}.jpg"))
    # print('unit8, numpy로 바꾸고 저장!')

from PIL import Image, ImageDraw, ImageFont
def visualize_token_positions_with_text(pixel_tensor, token_wise_positions_list, save_path, patch_size=14):
    os.makedirs(save_path, exist_ok=True)
    
    for idx, (token_name, position_tuples) in enumerate(token_wise_positions_list):
        # 1. 노이즈 삽입
        pixel_tensor_noised = replace_patch_with_noise(pixel_tensor, position_tuples, patch_size=patch_size)

        # 2. 텐서에서 이미지로 변환
        tensor = pixel_tensor_noised.squeeze(0)
        np_img = tensor.permute(1, 2, 0).cpu().numpy()
        np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

        # 3. 텍스트 추가
        draw = ImageDraw.Draw(img)
        font_size = 12
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for (row, col, _) in position_tuples:
            top = row * patch_size
            left = col * patch_size
            draw.text((left + 2, top + 2), token_name, fill=(255, 0, 0), font=font)

        # 4. 저장
        safe_token_name = re.sub(r'[<>:/\\|?*"]', '|', str(token_name))
        img.save(os.path.join(save_path, f"{idx}_{safe_token_name}.jpg"))


def draw_token_annotations_on_image(pixel_tensor, token_wise_positions_list, save_path, file_name="annotated.jpg", patch_size=14):
    import collections
    os.makedirs(save_path, exist_ok=True)

    # 1. 이미지 변환
    tensor = pixel_tensor.squeeze(0)
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(np_img).convert("RGBA")

    # 2. 오버레이 준비
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    offset_dict = collections.defaultdict(int)
    count_dict = collections.defaultdict(int)
    line_spacing = 14

    # 3. 좌표별 등장 횟수 카운트
    for _, position_tuples in token_wise_positions_list:
        for (row, col, _) in position_tuples:
            count_dict[(row, col)] += 1

    # 4. 시각화
    overlap_tokens = []
    for token_name, position_tuples in token_wise_positions_list:
        for (row, col, _) in position_tuples:
            top = row * patch_size
            left = col * patch_size
            token_name2 = token_name.replace('▁', "")
            text = str(token_name2)
            box_padding = 2

            offset = offset_dict[(top, left)]
            offset_dict[(top, left)] += 1
            y_offset = top + offset * line_spacing

            # 주변 nxn 영역에서 겹침 확인
            overlap_count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    neighbor = (row + dr, col + dc)
                    overlap_count += count_dict.get(neighbor, 0)
            is_overlapping = overlap_count > 1

            # 밝기 기반 대비 색 계산
            region = np_img[max(0, top):top+3, max(0, left):left+3]
            avg_color = region.mean(axis=(0, 1)) if region.size else np.array([127, 127, 127])
            brightness = np.dot(avg_color, [0.299, 0.587, 0.114])

            if is_overlapping:
                overlap_tokens.append(token_name)
            text_color = (255, 255, 0, 255) if is_overlapping else (255, 255, 255, 255)
            box_color = (0, 0, 0, 160) if brightness > 128 else (255, 255, 255, 160)
        

            # 텍스트 박스 및 텍스트 그리기
            bbox = draw.textbbox((left, y_offset), text, font=font)
            box_coords = [
                (bbox[0] - box_padding, bbox[1] - box_padding),
                (bbox[2] + box_padding, bbox[3] + box_padding)
            ]
            draw.rectangle(box_coords, fill=box_color)
            draw.text((bbox[0], bbox[1]), text, font=font, fill=text_color)

    # 5. 저장
    annotated_img = Image.alpha_composite(img, overlay).convert("RGB")
    annotated_img.save(os.path.join(save_path, file_name))
    return overlap_tokens


# def input_pixels_to_img(pixel_tensor, token_index, save_path):
#     tensor = pixel_tensor.squeeze(0) # 1. 배치 차원 제거: [1, 2or3, 336, 336] → [2or3, 336, 336]
#     np_img = tensor.permute(1, 2, 0).cpu().numpy() # 2. 텐서를 numpy로 변환하고, 채널 순서 바꾸기: [2or3, H, W] → [H, W, 2or3]
#     np_img = (np_img * 255).clip(0, 255).astype(np.uint8) # 3. 0~255 범위로 변환 (uint8로 타입 변경)
#     img = Image.fromarray(np_img) # 4. PIL 이미지로 변환 후 저장
#     img.save(f"{save_path}/input_pixels_{token_index}.jpg")
#     # print('unit8, numpy로 바꾸고 저장!')

def get_output_token_info(model, processor, lm_output, device):
    now_output_token_idx = torch.tensor([[int(lm_output['sequences'][0][-1])]], device=device)
    token_name = processor.tokenizer.convert_ids_to_tokens(now_output_token_idx, skip_special_tokens=False)[0]
    return (token_name, now_output_token_idx)

def compute_iou(a, b):
    set_a = set(a)
    set_b = set(b)
    
    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0  # 둘 다 비어 있을 경우
    return round(len(intersection) / len(union), 2)

def plot_eyetrack_bargraph(eyetrack_remain_list, save_path='eyetrack_remain_plot.png'):
    # 토큰 이름과 비율만 추출
    token_names = [entry[1] for entry in eyetrack_remain_list]
    rates = [entry[2] for entry in eyetrack_remain_list]

    # 그래프 크기 자동 조절
    plt.figure(figsize=(max(10, len(token_names) * 0.6), 6))
    
    # 막대 그래프
    plt.bar(range(len(token_names)), rates, color='skyblue')
    plt.xticks(range(len(token_names)), token_names, rotation=45, ha='right', fontsize=8)
    plt.ylabel("New Unique Position Rate")
    plt.xlabel("Generated Tokens")
    plt.title("New Unique Eyetrack Positions per Token")
    plt.ylim(0, 1)  # ✅ y축 고정: 항상 0~1
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=150)
    plt.close()


# def select_least_overlapping_tuples(tuple_list, k):
#     if not tuple_list:
#         return []
#     # tuple_list: [(new_seq, new_mask, new_pkv, new_score), ...]
    
#     # 1. new_seq만 뽑아서 set 변환
#     seq_sets = [set(tup[0].squeeze().tolist()) for tup in tuple_list]
    
#     selected_indices = []
#     remaining_indices = set(range(len(seq_sets)))
    
#     # 첫 번째는 그냥 0번 고름
#     first_idx = 0
#     selected_indices.append(first_idx)
#     remaining_indices.remove(first_idx)
    
#     while len(selected_indices) < k and remaining_indices:
#         best_idx = None
#         best_overlap = float('inf')
        
#         for idx in remaining_indices:
#             overlap = sum(len(seq_sets[idx] & seq_sets[sel_idx]) for sel_idx in selected_indices)
            
#             if overlap < best_overlap:
#                 best_overlap = overlap
#                 best_idx = idx
        
#         selected_indices.append(best_idx)
#         remaining_indices.remove(best_idx)
    
#     # 최종 선택
#     selected_tuples = [tuple_list[i] for i in selected_indices]
#     return selected_tuples

def select_least_overlapping_tuples(tuple_list, k):
    if not tuple_list:
        return []
    
    seq_sets = [set(tup[0].squeeze().tolist()) for tup in tuple_list]
    selected_indices = []
    remaining_indices = set(range(len(seq_sets)))

    first_idx = 0
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < k and remaining_indices:
        best_idx = None
        best_overlap = float('inf')

        for idx in remaining_indices:
            # 'sum' 대신 'max'로
            overlap = max(len(seq_sets[idx] & seq_sets[sel_idx]) for sel_idx in selected_indices)
            
            if overlap < best_overlap:
                best_overlap = overlap
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    selected_tuples = [tuple_list[i] for i in selected_indices]
    return selected_tuples

def select_top_score_tuples(tuple_list, k):
    if not tuple_list:
        return []

    # 각 튜플의 score_dict 총합 계산 (score_dict는 인덱스 5에 위치)
    score_sums = [sum(tup[5].values()) for tup in tuple_list]

    # score 총합 기준으로 인덱스를 내림차순 정렬
    sorted_indices = sorted(range(len(tuple_list)), key=lambda i: score_sums[i], reverse=True)

    # 상위 k개 선택
    selected_indices = sorted_indices[:k]

    # 해당 튜플 반환
    selected_tuples = [tuple_list[i] for i in selected_indices]
    return selected_tuples

def select_high_score_least_overlapping_tuples(tuple_list, k):
    if not tuple_list:
        return []

    # Step 1: Prepare data
    seq_sets = [set(tup[0].squeeze().tolist()) for tup in tuple_list]
    score_sums = [sum(tup[5].values()) for tup in tuple_list]  # score_dict is at index 5

    selected_indices = []
    remaining_indices = set(range(len(tuple_list)))

    # Step 2: Pick the highest-scoring one first
    first_idx = max(range(len(score_sums)), key=lambda i: score_sums[i])
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < k and remaining_indices:
        best_idx = None
        best_score = -float('inf')

        for idx in remaining_indices:
            overlap = max(len(seq_sets[idx] & seq_sets[sel_idx]) for sel_idx in selected_indices)
            score = score_sums[idx]

            # Example heuristic: prioritize score, penalize overlap
            adjusted_score = score - overlap  # ← key line

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    selected_tuples = [tuple_list[i] for i in selected_indices]
    return selected_tuples

def visualize_tsne_top_k(token_step, next_token_name, top_indices, model, processor, save_path):
    top_indices = top_indices.squeeze(0)
    
    next_token_name = re.sub(r'[<>:/\\|?*"]', '|', str(next_token_name))  
    embedding_weight = model.language_model.model.embed_tokens.weight
    top_embeddings = embedding_weight[top_indices]  # shape: (top_k, hidden_size)
    top_k_token_names = [processor.tokenizer.decode([idx.item()]) for idx in top_indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    top_embeddings_2d = tsne.fit_transform(top_embeddings.detach().cpu().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(top_embeddings_2d[:, 0], top_embeddings_2d[:, 1])

    # ===== 무조건 [-300, 300] =====
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    # ==============================

    for i, token_name in enumerate(top_k_token_names):
        plt.annotate(token_name, (top_embeddings_2d[i, 0], top_embeddings_2d[i, 1]))

    plt.title('t-SNE Visualization of Top-30 Embeddings')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{token_step}_{next_token_name}_tsne_embeddings.png'))
    plt.close()