
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
    """多线程下限制全局 DashScope 调用最小间隔，减轻 RateQuota。"""
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

def parse_wavespeed_api_keys(arg_value: Optional[str] = None) -> List[str]:
    """从命令行或环境变量解析 WaveSpeed API keys（支持逗号分隔多个 key 做轮换）。"""
    raw = (arg_value or "").strip() or (os.getenv("WAVESPEED_API_KEYS") or "").strip()
    if raw:
        return [k.strip() for k in raw.split(",") if k.strip()]
    single = (os.getenv("WAVESPEED_API_KEY") or "").strip()
    return [single] if single else []

def _is_wavespeed_rate_limit(http_status: Optional[int], message: str) -> bool:
    m = (message or "").lower()
    if http_status == 429:
        return True
    if "429" in m and ("rate" in m or "limit" in m or "too many" in m):
        return True
    if "rate limit" in m or "too many requests" in m:
        return True
    return False

_WAVESPEED_SUBMIT_URL = "https://api.wavespeed.ai/api/v3/wavespeed-ai/qwen-image/edit-plus"

def _call_wavespeed_edit_plus(api_key: str, image_url: str, prompt: str) -> str:
    """调用 WaveSpeed Qwen-Image-Edit-Plus；提交后轮询直至完成。遇 429 抛出便于轮换 key。"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "enable_base64_output": False,
        "enable_sync_mode": False,
        "images": [image_url],
        "output_format": "jpeg",
        "prompt": prompt,
        "seed": -1,
    }
    r = requests.post(
        _WAVESPEED_SUBMIT_URL,
        headers=headers,
        json=payload,
        timeout=Urllib3Timeout(connect=30, read=120, total=180),
    )
    try:
        resp_json = r.json()
    except Exception:
        resp_json = {}

    code = resp_json.get("code")
    msg = str(resp_json.get("message") or r.text or "")
    if r.status_code == 429 or code == 429 or _is_wavespeed_rate_limit(r.status_code, msg):
        raise Exception(f"WaveSpeed rate limited [HTTP {r.status_code}] [{code}]: {msg}")
    if code != 200:
        raise Exception(f"WaveSpeed submit error [HTTP {r.status_code}] [{code}]: {msg}")

    data = resp_json.get("data") or {}
    if data.get("status") == "completed" and data.get("outputs"):
        print("  [WaveSpeed] 图生完成（同步返回）")
        return data["outputs"][0]

    task_id = data.get("id")
    poll_url = (data.get("urls") or {}).get("get") if isinstance(data.get("urls"), dict) else None
    if not poll_url and task_id:
        poll_url = f"https://api.wavespeed.ai/api/v3/predictions/{task_id}"
    if not poll_url:
        raise Exception(f"WaveSpeed: missing task id / poll URL: {resp_json}")

    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        pr = requests.get(
            poll_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=Urllib3Timeout(connect=30, read=60, total=90),
        )
        try:
            pj = pr.json()
        except Exception:
            pj = {}
        if pr.status_code == 429 or pj.get("code") == 429:
            raise Exception(f"WaveSpeed poll rate limited [HTTP {pr.status_code}]")
        pdata = pj.get("data") or {}
        status = pdata.get("status")
        if status == "completed":
            outs = pdata.get("outputs") or []
            if outs:
                print("  [WaveSpeed] 图生完成（轮询）")
                return outs[0]
            raise Exception(f"WaveSpeed completed but no outputs: {pj}")
        if status == "failed":
            raise Exception(f"WaveSpeed task failed: {pdata.get('error', pj)}")
        time.sleep(2.0)

    raise Exception("WaveSpeed: polling timeout")

def load_subset_data(subset_path: str) -> List[Dict]:
    with open(subset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def dedupe_records_by_img(data: List[Dict]) -> Tuple[List[Dict], int]:
    """同一 img 多条记录时仅保留第一条，便于与输出目录一对一统计、避免本轮重复跑同图。"""
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
    """若 Step 1..max(intensities) 下 `{{base}}_edit_{{step}}.jpg` 与 `{{base}}_del_{{step}}.jpg` 均存在，视为该样本已全部跑完。"""
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

def check_progressive_constraint(client: OpenAI, candidate_detail: str, previous_details: List[str], model: str = "deepseek-v3.2") -> bool:
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
                    clean_text = re.sub(r'[^\w\s]', '', text).lower()
                    clean_caption = re.sub(r'[^\w\s]', '', caption).lower()
                    if clean_text in clean_caption:
                        valid_details.append(d)
            return valid_details
    except Exception as e:
        print(f"Detail extraction failed: {e}")
    return []

def filter_details_for_progression(client: OpenAI, details: List[Dict], max_needed: int) -> List[Dict]:
    """筛选细节列表，确保满足渐进式约束"""
    if not details:
        return []
        
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

def _vlm_assistant_text_from_multimodal_response(response: Any) -> str:
    """从 MultiModalConversation 非流式返回中拼出助手正文。"""
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
    """使用百炼 MultiModalConversation（qwen3-vl-plus，非流带思考）验证编辑/删除是否成功"""
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
    response = call_llm(client, prompt).strip()
    instruction = re.sub(r'^["\']|["\']$', '', response)
    if "Target:" in instruction:
        return f"Generate a reasonable image. IMPORTANT: You must strictly maintain all other parts of the image unchanged except for executing: {instruction}"
    return f"Generate a reasonable image. IMPORTANT: You must strictly maintain all other parts of the image unchanged except for modifying: {detail_text}"

def create_deletion_prompt(client: OpenAI, original_caption: str, detail_text: str) -> str:
    """生成删除 prompt"""
    return f"Generate a reasonable image. IMPORTANT: You must strictly maintain all other parts of the image unchanged except for executing the following deletion instructions:\nTarget: \"{detail_text}\" -> Action: Remove"

def _call_dashscope_generation_once(api_key: str, image_url: str, prompt: str) -> str:
    """单次 HTTP 调用图生接口（不含限流重试）。"""
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
    wavespeed_api_keys: Optional[List[str]] = None,
) -> str:
    """调用阿里云 DashScope API (Qwen-Image-Edit)；遇 Throttling.RateQuota 时退避重试。

    若配置了 WaveSpeed keys（环境变量 WAVESPEED_API_KEY / WAVESPEED_API_KEYS），在 DashScope
    触发限流时会依次尝试 WaveSpeed Qwen-Image-Edit-Plus；WaveSpeed 返回 429 时轮换下一个 key。

    参考文档: https://help.aliyun.com/zh/model-studio/qwen-image-edit-api
