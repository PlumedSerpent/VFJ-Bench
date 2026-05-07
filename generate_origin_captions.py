
import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI

MODEL_QWEN2 ="qwen2.5-vl-72b-instruct"# "openai/gpt-4o-2024-11-20"
MODEL_QWEN2 = "gpt-4o-2024-11-20"

CAPTION_PROMPT = "Generate a detailed caption for this image."

SYSTEM_PROMPT = (
    "You are an expert multilingual vision-language assistant. "
    "Given an image and a user prompt, produce a single, coherent, richly detailed caption "
    "that is precise, faithful to the visual content, and free of hallucination. "
    "Do not list multiple options or include meta commentary."
)

def build_client_aliyun() -> OpenAI:
    """Build OpenAI client (using Aliyun for Qwen2)"""
    api_key = "YOUR_API_KEY_HERE"
    return OpenAI(api_key=api_key, base_url="https://api.bltcy.ai/v1")

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64"""
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def normalize_response_content(content: Any) -> str:
    """Normalize the content returned by the model"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "\n".join(chunks).strip()
    return str(content).strip()

def call_vlm(
    client: OpenAI,
    model_name: str,
    image_b64: str,
    prompt: str,
    max_retries: int = 3,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> Optional[str]:
    """Call VLM API to generate caption"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            raw_content = completion.choices[0].message.content
            caption = normalize_response_content(raw_content)
            if caption:
                return caption
            raise ValueError("模型返回内容为空")
        except Exception as exc:
            wait = min(2 ** (attempt - 1), 10)
            print(f"[WARN] {model_name} 调用失败（第 {attempt} 次）: {exc}")
            if attempt == max_retries:
                return None
            time.sleep(wait)
    return None

def process_single_image(
    image_path: Path,
    client: OpenAI,
    model_name: str,
) -> Dict[str, Any]:
    """Process a single image and generate a caption using the specified model"""
    image_b64 = encode_image_to_base64(image_path)
    image_name = image_path.name
    
    result = {
        "img": image_name,
        "source": model_name,
        "caption": None,
    }
    
    caption = call_vlm(
        client=client,
        model_name=model_name,
        image_b64=image_b64,
        prompt=CAPTION_PROMPT,
    )
    result["caption"] = caption
    
    return result

def find_jpg_files(images_dir: Path) -> List[Path]:
    """Find all jpg files in the directory"""
    jpg_files = list(images_dir.glob("*.jpg"))
    jpg_files.extend(images_dir.glob("*.JPG"))
    return sorted(jpg_files)

def get_valid_base_names(edited_dir: Path) -> Set[str]:
    valid_names = set()
    if not edited_dir.exists():
        print(f"[WARN] edited_images 目录不存在: {edited_dir}")
        return valid_names
        
    for file_path in edited_dir.glob("*.jpg"):
        filename = file_path.name
        if "_edit" in filename:
            base_name = filename.split("_edit")[0]
            if base_name:
                valid_names.add(base_name)
    
    return valid_names

def main():
    parser = argparse.ArgumentParser(
        description="为 images 文件夹下的原始图片生成 caption（使用 Qwen2）"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="origin_images",
        help="图像目录路径",
    )
    parser.add_argument(
        "--edited-images-dir",
        type=str,
        default="edited_images",
        help="编辑后图像的目录路径（用于过滤原始图片）",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="captions/origin_captions_qwen2.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制处理的图像数量（默认处理全部）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数量（默认：1，单线程处理）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="生成温度",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="最大输出 token 数",
    )
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    edited_images_dir = Path(args.edited_images_dir)
    output_file = Path(args.output_file)
    output_dir = output_file.parent
    
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到图像目录：{images_dir}")
    
    print(f"正在扫描 {edited_images_dir} 以提取有效的文件名...")
    valid_base_names = get_valid_base_names(edited_images_dir)
    print(f"找到 {len(valid_base_names)} 个有效的 base name（从 edited_images 中提取）")
    
    jpg_files = find_jpg_files(images_dir)
    if not jpg_files:
        print(f"在 {images_dir} 中未找到任何 jpg 文件")
        return
    
    print(f"在 {images_dir} 中找到 {len(jpg_files)} 个 jpg 文件")
    
    filtered_files = []
    for f in jpg_files:
        if f.stem in valid_base_names:
            filtered_files.append(f)
            
    if not filtered_files:
        print(f"没有找到匹配的原始图片（文件名需在 edited_images 中存在对应的 _edit 版本）")
        return
        
    print(f"过滤后剩余 {len(filtered_files)} 个待处理文件")
    jpg_files = filtered_files
    
    if args.max_samples is not None and args.max_samples > 0:
        jpg_files = jpg_files[:args.max_samples]
    
    client_qwen = build_client_aliyun()
    model_name = MODEL_QWEN2
    
    print(f"\n{'='*60}")
    print(f"开始处理图片（使用 {model_name}）...")
    print(f"{'='*60}")
    
    existing_results = {}
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    for item in existing_data:
                        if "img" in item and item["caption"] is not None:
                            existing_results[item["img"]] = item
            print(f"加载了 {len(existing_results)} 个已有结果")
        except Exception as e:
            print(f"[WARN] 加载已有结果失败: {e}")
    
    results: List[Dict[str, Any]] = []
    processed_count = 0
    skipped_count = 0
    
    tasks = []
    for img_path in jpg_files:
        image_name = img_path.name
        if image_name not in existing_results:
            tasks.append(img_path)
        else:
            results.append(existing_results[image_name])
            skipped_count += 1
            
    print(f"待处理: {len(tasks)}，跳过: {skipped_count}")

    if not tasks:
        print("所有文件已处理完成。")
        return

    def save_results(current_results):
        all_results_to_save = list(existing_results.values()) + current_results
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_map = {item["img"]: item for item in results}
        final_list = list(final_map.values())
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(final_list, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已保存 {len(final_list)} 条结果到 {output_file}")

    if args.workers == 1:
        print(f"使用单线程处理...")
        for image_path in tqdm(tasks, desc="生成 Caption"):
            try:
                result = process_single_image(
                    image_path=image_path,
                    client=client_qwen,
                    model_name=model_name,
                )
                results.append(result)
                processed_count += 1
                
                if processed_count % 5 == 0:
                    save_results(results[skipped_count:])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with output_file.open("w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"[INFO] 已保存 {len(results)} 条结果")
                
            except Exception as e:
                print(f"[ERROR] 处理 {image_path.name} 失败: {e}")
                continue
    else:
        print(f"使用 {args.workers} 个线程处理...")
        
        def process_with_client(image_path: Path) -> tuple[str, Dict[str, Any]]:
            """Wrapper function for multi-threading processing"""
            image_name = image_path.name
            try:
                result = process_single_image(
                    image_path=image_path,
                    client=client_qwen,
                    model_name=model_name,
                )
                return image_name, result
            except Exception as e:
                print(f"[ERROR] 处理 {image_name} 失败: {e}")
                return image_name, None
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_with_client, img_path): img_path 
                      for img_path in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="生成 Caption"):
                image_name, result = future.result()
                if result is None:
                    continue
                
                processed_count += 1
                results.append(result)
                
                if processed_count % 5 == 0:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with output_file.open("w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"[INFO] 已保存 {len(results)} 条结果")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"  总文件数: {len(jpg_files)}")
    print(f"  新处理: {processed_count}")
    print(f"  跳过（已存在）: {skipped_count}")
    print(f"  总结果数: {len(results)}")
    print(f"  结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
