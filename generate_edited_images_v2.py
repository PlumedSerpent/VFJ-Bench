#!/usr/bin/env python3


import os
import json
import random
import re
import argparse
import requests
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import base64
from openai import OpenAI
from dashscope import MultiModalConversation
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib3.util import Timeout as Urllib3Timeout

_dashscope_pace_lock = threading.Lock()
_last_dashscope_monotonic: float = 0.0


def _is_dashscope_throttling(code: Optional[str], message: Optional[str]) -> bool:
    c = (code or "").lower()
    m = (message or "").lower()
    if "throttling" in c or "throttling" in m:
        return True
    if "ratequota" in c or "rate limit" in m or "too many requests" in m:
        return True
    return False


def _pace_dashscope_calls(min_interval_sec: float) -> None:
    """Limit the minimum interval for global DashScope calls under multi-threading to reduce RateQuota."""
    global _last_dashscope_monotonic
    if min_interval_sec <= 0:
        return
    with _dashscope_pace_lock:
        now = time.monotonic()
        wait = min_interval_sec - (now - _last_dashscope_monotonic)
        if wait > 0:
            time.sleep(wait)
        _last_dashscope_monotonic = time.monotonic()


def _sleep_rate_limit_backoff(attempt: int, base_wait_sec: float) -> None:
    cap = 120.0
    wait = min(cap, base_wait_sec * (2**attempt) + random.uniform(0, 3))
    print(f"  [Rate limit] 等待 {wait:.1f}s 后重试 (第 {attempt + 1} 次)...")
    time.sleep(wait)





# ==============================================================================
# 基础工具函数
# ==============================================================================

def load_subset_data(subset_path: str) -> List[Dict]:
    with open(subset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def dedupe_records_by_img(data: List[Dict]) -> Tuple[List[Dict], int]:
    """Keep only the first record when there are multiple records for the same img, to facilitate one-to-one statistics with the output directory and avoid re-running the same image in this round."""
    seen = set()
    out: List[Dict] = []
    for item in data:
        img = item.get("img")
        if not img or not isinstance(img, str):
            continue
        if img in seen:
            continue
        seen.add(img)
        out.append(item)
    return out, len(data) - len(out)


def is_sample_fully_processed(output_dir: str, img_filename: str, edit_intensities: List[int]) -> bool:
    """If `{{base}}_edit_{{step}}.jpg` and `{{base}}_del_{{step}}.jpg` exist for Step 1..max(intensities), the sample is considered fully processed."""
    if not edit_intensities:
        return False
    base_name = os.path.splitext(img_filename)[0]
    max_steps = max(edit_intensities)
    for step in range(1, max_steps + 1):
        edit_p = os.path.join(output_dir, f"{base_name}_edit_{step}.jpg")
        del_p = os.path.join(output_dir, f"{base_name}_del_{step}.jpg")
        if not (os.path.isfile(edit_p) and os.path.isfile(del_p)):
            return False
    return True


def select_random_incomplete_samples(
    data: List[Dict],
    output_dir: str,
    edit_intensities: List[int],
    num_samples: int,
) -> Tuple[List[Dict], int]:
    """
    Randomly draw up to num_samples incomplete samples without replacement (fully completed samples do not count towards num_samples).
    Assume `data` is deduplicated by img: statistics and quotas are counted by unique images.
    Returns (selected list, number of fully completed images in the candidate pool).
    """
    if num_samples <= 0 or not data:
        return [], sum(
            1
            for it in data
            if is_sample_fully_processed(output_dir, it["img"], edit_intensities)
        )
    ready_count = sum(
        1
        for it in data
        if is_sample_fully_processed(output_dir, it["img"], edit_intensities)
    )
    pool = list(data)
    random.shuffle(pool)
    selected: List[Dict] = []
    for item in pool:
        if len(selected) >= num_samples:
            break
        if is_sample_fully_processed(output_dir, item["img"], edit_intensities):
            continue
        selected.append(item)
    return selected, ready_count


def init_llm_client(api_key: Optional[str] = None) -> OpenAI:
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请提供 API Key (通过参数或 DASHSCOPE_API_KEY 环境变量)")
    
    # 使用阿里云 DashScope 兼容 OpenAI 接口
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client

def call_llm(client: OpenAI, prompt: str, model: str = "deepseek-v3.2", max_retries: int = 3) -> str:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                print(f"LLM API调用最终失败: {e}")
                return ""
    return ""

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')

def upload_image_to_url(image_path: str) -> str:
    if image_path.startswith('http://') or image_path.startswith('https://'):
        return image_path
    base64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1][1:].lower()
    if ext == 'jpg': ext = 'jpeg'
    return f"data:image/{ext};base64,{base64_image}"

def download_image(image_url: str, save_path: str):
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        raise Exception(f"Failed to download image: {e}")

# ==============================================================================
# 核心功能 1: 细节提取与约束检查 (Constraint Check)
# ==============================================================================

def check_progressive_constraint(client: OpenAI, candidate_detail: str, previous_details: List[str], model: str = "deepseek-v3.2") -> bool:
    """
    Check if the candidate detail violates the progressive editing constraint.
    Constraint: Subsequent edited details cannot be a "part" or "subset" of previously edited details.
    """
    if not previous_details:
        return True
        
    prev_details_str = "\n".join([f"- {d}" for d in previous_details])
    
    prompt = f"""
I have a list of visual details that have already been edited/removed in an image. I want to add a new detail to the edit list.
Constraint: The new detail MUST NOT be a physical part, a subset, or a visual component of any previously processed detail.
(Example of violation: If "a red car" was edited, "the car's tire" cannot be edited next because it's part of the car.)

Previously processed details:
{prev_details_str}

Candidate detail to process next:
"{candidate_detail}"

Question: Is the candidate detail a physical part or subset of any of the previously processed details?
Answer strictly with YES or NO.
"""
    try:
        response = call_llm(client, prompt, model=model).strip().lower()
        if "yes" in response:
            return False
        return True
    except Exception as e:
        print(f"  [Warning] Constraint check failed: {e}. Assuming safe.")
        return True

def extract_modifiable_details_with_llm(client: OpenAI, caption: str) -> List[Dict]:
    """Extract modifiable details"""
    prompt = f"""
You are an expert annotator. Analyze the image caption and extract specific, modifiable visual elements.

Input Caption: {caption}

Task: Extract 8-12 distinct details. For each, provide:
- `text`: The exact phrase from the caption.
- `importance`: 1-10 based on visual saliency (10=core subject, 1=background).

Output strictly JSON:
{{
  "details": [
    {{ "text": "phrase", "importance": 8 }}
  ]
}}
"""
    try:
        response = call_llm(client, prompt)
        json_match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            details = result.get("details", [])
            valid_details = []
            for d in details:
                text = d.get("text", "").strip()
                if text:
                    # 简单验证文本是否存在于 caption 中（忽略大小写和标点）
                    clean_text = re.sub(r'[^\w\s]', '', text).lower()
                    clean_caption = re.sub(r'[^\w\s]', '', caption).lower()
                    if clean_text in clean_caption:
                        valid_details.append(d)
            return valid_details
    except Exception as e:
        print(f"Detail extraction failed: {e}")
    return []

def filter_details_for_progression(client: OpenAI, details: List[Dict], max_needed: int) -> List[Dict]:
    """Filter the details list to ensure it meets progressive constraints"""
    if not details:
        return []
        
    # 按重要性排序（优先处理不重要的细节，逐渐逼近核心？或者反过来？）
    # 原脚本策略：优先修改不重要的细节 (low importance first)。
    # 这样可以在不破坏画面主体的情况下先改背景/小物体。
    sorted_details = sorted(details, key=lambda x: x.get("importance", 5))
    
    selected_details = []
    selected_texts = []
    
    print("  [约束检查] 正在筛选细节...")
    for detail in sorted_details:
        if len(selected_details) >= max_needed:
            break
            
        candidate_text = detail.get("text", "")
        if check_progressive_constraint(client, candidate_text, selected_texts):
            selected_details.append(detail)
            selected_texts.append(candidate_text)
        else:
            print(f"    - 跳过 '{candidate_text}' (与已选细节冲突)")
            
    return selected_details

# ==============================================================================
# 核心功能 2: VLM 验证 (Verification)
# ==============================================================================

def _vlm_assistant_text_from_multimodal_response(response: Any) -> str:
    """Extract assistant text from non-streaming MultiModalConversation response."""
    output = getattr(response, "output", None) if response is not None else None
    if output is None and isinstance(response, dict):
        output = response.get("output")
    choices = getattr(output, "choices", None) if output is not None else None
    if choices is None and isinstance(output, dict):
        choices = output.get("choices", [])
    if not choices:
        return ""
    msg = getattr(choices[0], "message", None)
    if msg is None and isinstance(choices[0], dict):
        msg = choices[0].get("message", {})
    content = getattr(msg, "content", None) if msg is not None else None
    if content is None and isinstance(msg, dict):
        content = msg.get("content", [])
    parts: List[str] = []
    for block in content or []:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def verify_edit_with_vlm(
    api_key: str,
    original_img_path: str,
    edited_img_path: str,
    instruction: str,
    model: str = "qwen3-vl-plus",
    thinking_budget: int = 81920,
    rate_limit_retries: int = 10,
    rate_limit_base_wait: float = 8.0,
    min_api_interval_sec: float = 0.0,
) -> Tuple[bool, str]:
    """Verify if the edit/deletion was successful using Bailian MultiModalConversation (qwen3-vl-plus, non-streaming with thinking)"""
    try:
        img1_url = upload_image_to_url(original_img_path)
        img2_url = upload_image_to_url(edited_img_path)

        prompt_text = f"""I have two images.
Image 1: Original image.
Image 2: Edited image after applying the instruction.

Instruction: "{instruction}"

Task: Verify if the instruction was successfully applied.
1. Check if the specific change is visible in Image 2.
2. Check if the rest of the image remains largely consistent.

Output strictly in JSON:
{{
  "pass": true/false,
  "reason": "short explanation"
}}
"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": img1_url},
                    {"image": img2_url},
                    {"text": prompt_text},
                ],
            }
        ]

        response = None
        for attempt in range(rate_limit_retries + 1):
            _pace_dashscope_calls(min_api_interval_sec)
            response = MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=messages,
                result_format="message",
                stream=False,
                enable_thinking=True,
                thinking_budget=thinking_budget,
            )
            status = getattr(response, "status_code", None)
            if status is None and isinstance(response, dict):
                status = response.get("status_code")
            err = getattr(response, "message", None) or (
                response.get("message") if isinstance(response, dict) else None
            )
            rcode = getattr(response, "code", None)
            if rcode is None and isinstance(response, dict):
                rcode = response.get("code")
            if status == 200:
                break
            if _is_dashscope_throttling(str(rcode) if rcode else None, err) and attempt < rate_limit_retries:
                _sleep_rate_limit_backoff(attempt, rate_limit_base_wait)
                continue
            return False, err or f"VLM HTTP status {status}"

        if response is None:
            return False, "VLM: empty response"

        status = getattr(response, "status_code", None)
        if status is None and isinstance(response, dict):
            status = response.get("status_code")
        if status != 200:
            err = getattr(response, "message", None) or (
                response.get("message") if isinstance(response, dict) else None
            )
            return False, err or f"VLM HTTP status {status}"

        response_text = _vlm_assistant_text_from_multimodal_response(response)
        json_match = re.search(r"\{[\s\S]*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result.get("pass", False), result.get("reason", "No reason provided")

        if "true" in response_text.lower():
            return True, response_text
        return False, response_text

    except Exception as e:
        print(f"  [VLM Verify Error] {e}")
        return False, str(e)

# ==============================================================================
# 生成逻辑
# ==============================================================================

def create_edit_prompt(client: OpenAI, original_caption: str, detail_text: str) -> str:
    """Generate edit prompt"""
    prompt = f"""
Original: {original_caption}
Task: Change '{detail_text}' to something visually distinct.
Output ONLY the instruction: "Target: {{detail}} -> Action: Change to {{new_desc}}"
"""
    response = call_llm(client, prompt).strip()
    # 清理
    instruction = re.sub(r'^["\']|["\']$', '', response)
    if "Target:" in instruction:
        return f"Generate a reasonable image. IMPORTANT: You must strictly maintain all other parts of the image unchanged except for executing: {instruction}"
    return f"Generate a reasonable image. IMPORTANT: You must strictly maintain all other parts of the image unchanged except for modifying: {detail_text}"

def create_deletion_prompt(client: OpenAI, original_caption: str, detail_text: str) -> str:
    """Generate deletion prompt"""
    return f"Generate a reasonable image. IMPORTANT: You must strictly maintain all other parts of the image unchanged except for executing the following deletion instructions:\nTarget: \"{detail_text}\" -> Action: Remove"

def _call_dashscope_generation_once(api_key: str, image_url: str, prompt: str) -> str:
    """Single HTTP call to the image generation API (without rate limit retries)."""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "qwen-image-2.0-pro",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": " ",
            "prompt_extend": True,
            "watermark": False
        }
    }

    response = requests.post(
        url, headers=headers, json=payload,
        timeout=Urllib3Timeout(connect=30, read=300, total=600)
    )
    resp_json = response.json()

    if "code" in resp_json and resp_json.get("code") != "":
        code = resp_json.get("code", "Unknown")
        message = resp_json.get("message", "No message")
        raise Exception(f"API Error [{code}]: {message}")

    response.raise_for_status()

    output = resp_json.get("output", {})
    if "choices" in output and len(output["choices"]) > 0:
        choice = output["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise Exception(f"Generation incomplete, finish_reason: {choice.get('finish_reason')}")
        content_list = choice.get("message", {}).get("content", [])
        if content_list and "image" in content_list[0]:
            return content_list[0]["image"]

    raise Exception(f"Unexpected API response: {resp_json}")


def call_dashscope_api(
    api_key: str,
    image_url: str,
    prompt: str,
    rate_limit_retries: int = 10,
    rate_limit_base_wait: float = 8.0,
    min_api_interval_sec: float = 0.0,
) -> str:
    """Call Aliyun DashScope API (Qwen-Image-Edit); backoff and retry on Throttling.RateQuota.

    Reference: https://help.aliyun.com/zh/model-studio/qwen-image-edit-api
    """
    last_exc: Optional[Exception] = None
    for attempt in range(rate_limit_retries + 1):
        try:
            _pace_dashscope_calls(min_api_interval_sec)
            return _call_dashscope_generation_once(api_key, image_url, prompt)
        except Exception as e:
            last_exc = e
            msg = str(e)
            if _is_dashscope_throttling(None, msg):
                if attempt < rate_limit_retries:
                    _sleep_rate_limit_backoff(attempt, rate_limit_base_wait)
                    continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("call_dashscope_api: no attempt made")

def process_single_sample(args_tuple) -> Dict:
    """Process the complete workflow for a single sample"""
    item, images_dir, output_dir, api_key, edit_intensities, max_retries = args_tuple
    
    
    try:
        client = init_llm_client(api_key)
        img_filename = item["img"]
        img_path = os.path.join(images_dir, img_filename)
        ref_caption = item.get("ref", "")
        
        if not os.path.exists(img_path):
            return {"success": False, "error": f"Image not found: {img_path}"}
        
        if not ref_caption:
            return {"success": False, "error": "No reference caption"}

        if is_sample_fully_processed(output_dir, img_filename, edit_intensities):
            print(f"  Skip {img_filename} — 输出目录已有全部强度的 edit_* / del_*，视为已处理。")
            return {
                "img": img_filename,
                "success": True,
                "skipped_already_complete": True,
                "versions": [],
            }

        print(f"Processing {img_filename}...")
        
        # 1. 提取所有细节（不进行预筛选，保留全部作为候选池）
        # 增加提取数量，以便有更多备选
        raw_details = extract_modifiable_details_with_llm(client, ref_caption)
        if not raw_details:
             return {"success": False, "error": "No details extracted"}

        # 按重要性排序（优先处理不重要的细节）
        raw_details.sort(key=lambda x: x.get("importance", 5))
        
        result = {
            "img": img_filename,
            "success": True,
            "versions": []
        }
        
        base_name = os.path.splitext(img_filename)[0]
        
        # 维护当前状态
        curr_edit_img_path = img_path
        curr_edit_img_url = upload_image_to_url(img_path)
        
        curr_del_img_path = img_path
        curr_del_img_url = upload_image_to_url(img_path)
        
        # 已使用的 details (用于约束检查)
        used_details_texts = []
        
        # 标记哪些 details 已经被使用过（索引），避免重复
        used_indices = set()
        
        max_steps = max(edit_intensities)
        
        # 逐步执行 (Step-by-step)
        for step in range(1, max_steps + 1):
            print(f"  [Step {step}] Finding suitable detail...")
            
            step_success = False
            selected_detail = None
            
            # 遍历候选 details
            for idx, detail in enumerate(raw_details):
                if idx in used_indices:
                    continue
                    
                candidate_text = detail['text']
                
                # 约束检查
                if not check_progressive_constraint(client, candidate_text, used_details_texts):
                    # print(f"    - Skip '{candidate_text}' (constraint violation)")
                    continue
                
                print(f"    Trying candidate: '{candidate_text}'")
                
                # 尝试生成
                version_entry = {
                    "intensity": step,
                    "modified_detail": candidate_text,
                    "edit_success": False,
                    "del_success": False
                }
                
                # --- Edit Generation ---
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        edit_prompt = create_edit_prompt(client, ref_caption, candidate_text)
                        # print(f"      [Edit] Attempt {retry_count+1}")
                        gen_url = call_dashscope_api(
                            api_key, curr_edit_img_url, edit_prompt
                        )
                        
                        temp_path = os.path.join(output_dir, f"temp_edit_{base_name}_{step}_{retry_count}.jpg")
                        download_image(gen_url, temp_path)
                        
                        passed, reason = verify_edit_with_vlm(api_key, curr_edit_img_path, temp_path, edit_prompt)
                        if passed:
                            print(f"      [Edit Pass] {reason}")
                            final_path = os.path.join(output_dir, f"{base_name}_edit_{step}.jpg")
                            os.rename(temp_path, final_path)
                            # 暂存结果，等 Del 也成功再更新状态
                            version_entry["edit_output"] = final_path
                            version_entry["edit_prompt"] = edit_prompt
                            version_entry["edit_success"] = True
                            break
                        else:
                            print(f"      [Edit Fail] {reason}")
                            os.remove(temp_path)
                            retry_count += 1
                    except Exception as e:
                        print(f"      [Edit Error] {e}")
                        retry_count += 1
                
                # 如果 Edit 失败，就不需要尝试 Del 了，直接换下一个 detail
                if not version_entry["edit_success"]:
                    print(f"    Detail '{candidate_text}' failed edit verification. Trying next candidate...")
                    continue

                # --- Deletion Generation ---
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        del_prompt = create_deletion_prompt(client, ref_caption, candidate_text)
                        # print(f"      [Del] Attempt {retry_count+1}")
                        gen_url = call_dashscope_api(
                            api_key, curr_del_img_url, del_prompt
                        )
                        
                        temp_path = os.path.join(output_dir, f"temp_del_{base_name}_{step}_{retry_count}.jpg")
                        download_image(gen_url, temp_path)
                        
                        passed, reason = verify_edit_with_vlm(api_key, curr_del_img_path, temp_path, del_prompt)
                        if passed:
                            print(f"      [Del Pass] {reason}")
                            final_path = os.path.join(output_dir, f"{base_name}_del_{step}.jpg")
                            os.rename(temp_path, final_path)
                            version_entry["del_output"] = final_path
                            version_entry["del_prompt"] = del_prompt
                            version_entry["del_success"] = True
                            break
                        else:
                            print(f"      [Del Fail] {reason}")
                            os.remove(temp_path)
                            retry_count += 1
                    except Exception as e:
                        print(f"      [Del Error] {e}")
                        retry_count += 1
                
                # 检查是否 Edit 和 Del 都成功
                if version_entry["edit_success"] and version_entry["del_success"]:
                    # 成功锁定该 detail
                    step_success = True
                    selected_detail = detail
                    used_indices.add(idx)
                    used_details_texts.append(candidate_text)
                    
                    # 更新当前图片状态
                    curr_edit_img_path = version_entry["edit_output"]
                    curr_edit_img_url = upload_image_to_url(curr_edit_img_path)
                    
                    curr_del_img_path = version_entry["del_output"]
                    curr_del_img_url = upload_image_to_url(curr_del_img_path)
                    
                    # 记录结果
                    if step in edit_intensities:
                        result["versions"].append(version_entry)
                    
                    print(f"  [Step {step}] Success with detail: '{candidate_text}'")
                    break # 跳出 detail 循环，进入下一个 step
                else:
                    print(f"    Detail '{candidate_text}' failed (Edit={version_entry['edit_success']}, Del={version_entry['del_success']}). Trying next...")
            
            # 如果遍历完所有 details 都没有成功
            if not step_success:
                print(f"  [Error] Step {step} failed. No suitable details found after trying all candidates. Skipping sample.")
                result["success"] = False
                result["error"] = f"Failed at step {step}: no valid details"
                break # 停止 step 循环，放弃该样本
        
        return result

    except Exception as e:
        return {"success": False, "error": str(e), "img": item["img"]}

def main():
    parser = argparse.ArgumentParser(description="Generate edited images with VLM verification")
    parser.add_argument("--subset_path", type=str, default="subset.json")
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--output_dir", type=str, default="edited_images")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--edit_intensities", type=str, default="1,2,3,4")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3, help="Verification failure retries")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: API Key required.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    data = []
    if os.path.exists(args.subset_path):
        try:
            full_data = load_subset_data(args.subset_path)
            # 筛选出有图片的
            for item in full_data:
                if os.path.exists(os.path.join(args.images_dir, item["img"])):
                    data.append(item)
        except Exception as e:
            print(f"Error loading subset: {e}")
            
    if not data:
        print("Fallback to scanning directory...")
        for f in os.listdir(args.images_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({"img": f, "ref": f"A photo of {f}"}) # Placeholder caption

    raw_row_count = len(data)
    data, dedupe_dropped = dedupe_records_by_img(data)
    if dedupe_dropped:
        print(
            f"候选 json 中含重复 img：原始 {raw_row_count} 行，去重后唯一图片 {len(data)} "
            f"（合并丢弃 {dedupe_dropped} 行，避免计数与抽样按行翻倍）。"
        )

    intensities = [int(x) for x in args.edit_intensities.split(',')]
    intensities.sort()

    selected_samples, already_complete_count = select_random_incomplete_samples(
        data, args.output_dir, intensities, args.num_samples
    )
    # num_samples 按唯一图片计数；已满强度输出的图片不占名额。
    incomplete_pool = len(data) - already_complete_count
    print(
        f"唯一图片中已满强度(edit/del 1..{max(intensities)} 成对)：{already_complete_count}；"
        f"待处理池：{incomplete_pool}；"
        f"本轮抽样：{len(selected_samples)} / --num_samples {args.num_samples}。"
    )
    if len(selected_samples) < args.num_samples:
        print(
            f"[注意] 未完成样本不足，仅处理 {len(selected_samples)} 条（少于 --num_samples {args.num_samples}）。"
        )
    
    # 多线程处理
    tasks = []
    for item in selected_samples:
        tasks.append((item, args.images_dir, args.output_dir, api_key, intensities, args.max_retries))
        
    results = []
    
    if args.num_threads > 1:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            future_to_item = {executor.submit(process_single_sample, task): task[0]["img"] for task in tasks}
            for future in tqdm(as_completed(future_to_item), total=len(tasks)):
                res = future.result()
                results.append(res)
    else:
        for task in tqdm(tasks):
            res = process_single_sample(task)
            results.append(res)
            
    # 保存结果
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. Results saved to {args.output_dir}/results.json")

if __name__ == "__main__":
    main()
