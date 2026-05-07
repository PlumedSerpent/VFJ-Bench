import argparse
import base64
import json
import time
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from openai import OpenAI

from prompts import get_prompt

API_KEY_OPENROUTER = "YOUR_OPENROUTER_API_KEY"
API_KEY_ALIYUN = "YOUR_ALIYUN_API_KEY"

DEFAULT_BASE_DIR = Path("captions/")
DEFAULT_EVAL_DIR = Path("eval/461samples_baseexp/")
IMAGE_DIR = Path("/home/wang.j/Captions/images")

DEFAULT_FILES = [
    "origin_qwen_vs_origin_gpt4o.json"

]

def build_client(api_key: str, url: str, proxy: str = None) -> OpenAI:
    http_client = httpx.Client(proxy=proxy) if proxy else None
    return OpenAI(api_key=api_key, base_url=url, http_client=http_client)

def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_judge_user_prompt(caption1: str, caption2: str) -> str:
    return f"""Caption 1:
{caption1}

Caption 2:
{caption2}

Determine which is better and answer with the given format. Only mark a tie if there is no discernible difference in quality, informativeness, and precision after careful evaluation."""

def call_vlm_judge(
    client: OpenAI,
    model_name: str,
    image_path: Path,
    caption1: str,
    caption2: str,
    system_prompt: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    try:
        print(image_path)
        image_b64 = encode_image_to_base64(image_path)
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}"}

    user_prompt = get_judge_user_prompt(caption1, caption2)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_completion_tokens=2048,
            )
        except Exception:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    extra_body={"reasoning": {"enabled": True}},
                )
            except Exception:
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                except Exception:
                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0.3,
                            max_completion_tokens=2048,
                        )
                    except Exception as e:
                        print(traceback.format_exc())
                        print(f"[WARN] attempt {attempt} failed: {e}")
                        time.sleep(2)
                        continue

        response_message = completion.choices[0].message
        content = response_message.content.strip() if response_message.content else ""

        reason_match = re.search(r"Reason:\s*(.+?)\s*Judgment:", content, re.DOTALL | re.IGNORECASE)
        judgment_match = re.search(r"Judgment:\s*(.+)", content, re.DOTALL | re.IGNORECASE)

        reason = reason_match.group(1).strip() if reason_match else content
        judgment_text = judgment_match.group(1).strip() if judgment_match else content

        choice = "Unknown"
        if "Caption 1 is better" in judgment_text:
            choice = "Caption 1 is better"
        elif "Caption 2 is better" in judgment_text:
            choice = "Caption 2 is better"
        elif "Tie" in judgment_text:
            choice = "Tie"
        if choice == "Unknown":
            print(f"[WARN] could not parse judgment: {judgment_text}")

        return {"reasoning": reason, "choice": choice, "raw_output": content}

    return {"error": "Max retries exceeded"}

def process_file(
    input_file: Path,
    judge_model: str,
    judge_client: OpenAI,
    judge_name: str,
    system_prompt: str,
    output_tag: str,
    eval_dir: Path,
    sleep_time: float = 1.0,
    max_workers: int = 1,
    max_samples: Optional[int] = None,
):
    print(f"Processing {input_file} with judge {judge_name} (prompt={output_tag or 'base'})...")

    tag_infix = f"{output_tag}" if output_tag else ""
    output_filename = f"eval_{judge_name}_{tag_infix}on_{input_file.name}"
    output_path = eval_dir / output_filename

    if output_path.exists():
        print(f"Found existing eval file, resuming from {output_path}")
        with output_path.open("r") as f:
            data = json.load(f)
    else:
        with input_file.open("r") as f:
            data = json.load(f)

    results = [item.copy() for item in data]

    def save_checkpoint(processed_count: int):
        with output_path.open("w") as f:
            json.dump(results, f, indent=4)
        print(f"Checkpoint saved to {output_path} (new evaluations: {processed_count})")

    if max_samples is not None:
        data = data[:max_samples]
        results = results[:len(data)]

    to_evaluate_indices: List[int] = []
    for idx, item in enumerate(data):
        jr = item.get("judge_result")
        if jr and isinstance(jr, dict) and jr.get("choice") and jr.get("choice") != "Unknown":
            continue
        to_evaluate_indices.append(idx)

    total_to_evaluate = len(to_evaluate_indices)
    print(f"Total items: {len(data)}, to evaluate now: {total_to_evaluate}")

    if total_to_evaluate == 0:
        with output_path.open("w") as f:
            json.dump(results, f, indent=4)
        print(f"No new items to evaluate. Results saved to {output_path}")
        return

    new_eval_count = 0

    def _run_one(idx: int):
        item = data[idx]
        image_path = IMAGE_DIR / item.get("img")
        return idx, item, call_vlm_judge(
            judge_client, judge_model, image_path,
            item["caption1"], item["caption2"], system_prompt,
        )

    if max_workers == 1:
        for idx in to_evaluate_indices:
            _, item, judge_result = _run_one(idx)
            result_entry = item.copy()
            result_entry["judge_model"] = judge_name
            result_entry["judge_result"] = judge_result
            results[idx] = result_entry

            new_eval_count += 1
            print(f"Evaluated {item.get('img')} - Choice: {judge_result.get('choice')}")

            if new_eval_count % 10 == 0:
                save_checkpoint(new_eval_count)

            time.sleep(sleep_time)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    call_vlm_judge,
                    judge_client, judge_model, IMAGE_DIR / data[idx].get("img"),
                    data[idx]["caption1"], data[idx]["caption2"], system_prompt,
                ): idx
                for idx in to_evaluate_indices
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                item = data[idx]
                judge_result = future.result()

                result_entry = item.copy()
                result_entry["judge_model"] = judge_name
                result_entry["judge_result"] = judge_result
                results[idx] = result_entry

                new_eval_count += 1
                print(f"Evaluated {item.get('img')} - Choice: {judge_result.get('choice')}")

                if new_eval_count % 10 == 0:
                    save_checkpoint(new_eval_count)

    with output_path.open("w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_path}")

def check_proxy_ip(proxy: str):
    if not proxy:
        return
    try:
        with httpx.Client(proxy=proxy) as client:
            resp = client.get("https://api.ipify.org?format=json", timeout=10)
            print(f"Current Proxy IP: {resp.json()['ip']}")
    except Exception as e:
        print(f"Warning: Failed to check proxy IP: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate captions with VLMs as judges.")
    parser.add_argument("--judge", type=str, default="Gemini-3-pro", help="Judge display name")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview-c", help="Model identifier")
    parser.add_argument("--api_key", type=str, default="", help="Override API key")
    parser.add_argument("--url", type=str, default="https://openrouter.ai/api/v1", help="API base URL")
    parser.add_argument("--proxy", type=str, default="", help="Proxy URL")
    parser.add_argument(
        "--prompt", type=str, default="base",
        choices=["base", "grounding", "negative"],
        help="System prompt variant to use (default: base)",
    )
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR), help="Input captions directory")
    parser.add_argument("--eval_dir", type=str, default=str(DEFAULT_EVAL_DIR), help="Output eval directory")
    parser.add_argument("--sleep_time", type=float, default=1.0)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    system_prompt, output_tag = get_prompt(args.prompt)

    check_proxy_ip(args.proxy)

    client = build_client(
        api_key=args.api_key or API_KEY_OPENROUTER,
        url=args.url,
        proxy=args.proxy or None,
    )

    for filename in DEFAULT_FILES:
        input_path = base_dir / filename
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue

        process_file(
            input_path,
            args.model,
            client,
            args.judge,
            system_prompt=system_prompt,
            output_tag=output_tag,
            eval_dir=eval_dir,
            sleep_time=args.sleep_time,
            max_workers=args.max_workers,
            max_samples=args.max_samples,
        )

if __name__ == "__main__":
    main()
