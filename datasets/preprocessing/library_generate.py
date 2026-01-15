import os
import json
import argparse
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# -------------------------
# 1. CẤU HÌNH & KHỞI TẠO
# -------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cấu hình chính xác theo cột dữ liệu bạn cung cấp
DATASET_CONFIGS = [
    {"path": "walledai/AdvBench", "config": None, "split": "train", "col": "prompt", "label": "Jailbreak"},
    {"path": "JailbreakBench/JBB-Behaviors", "config": "behaviors", "split": "harmful", "col": "Goal", "label": "Behavioral"},
    {"path": "walledai/HarmBench", "config": "standard", "split": "train", "col": "prompt", "label": "Harmful"},
    {"path": "LibrAI/do-not-answer", "config": None, "split": "train", "col": "question", "label": "Policy"}
]

# -------------------------
# 2. HÀM RÚT TRÍCH SIGNATURE (LLM FEATURE EXTRACTION)
# -------------------------
def extract_signatures(text):
    """
    Sử dụng GPT-4o-mini để rút trích các cụm từ cốt lõi của cuộc tấn công.
    Giúp Layer 2 không còn 'ngây thơ' trước các biến thể câu chữ.
    """
    prompt = f"""
    [Task] Identify and extract 3-5 shortest, high-risk phrases or technical keywords from this malicious prompt.
    [Guidelines]
    - Focus on: bypass logic, system overrides, harmful actions, or role-play triggers.
    - Each phrase must be 2-5 words long.
    - Output ONLY a comma-separated list. No preamble.
    
    Malicious Prompt: "{text}"
    Keywords:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cyber-security analyst specialized in LLM Attack Signatures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=60
        )
        content = response.choices[0].message.content.strip()
        # Chuyển về chữ thường và lọc bỏ các cụm quá ngắn
        return [p.strip().lower() for p in content.split(",") if len(p.strip()) > 3]
    except Exception:
        return []

# -------------------------
# 3. QUY TRÌNH XỬ LÝ CHÍNH
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Layer 2 Signature Builder for AI Agent Security")
    parser.add_argument("--output", type=str, default="./processed_datasets/attack_signatures.json")
    args = parser.parse_args()

    all_signatures = set()
    
    # 3.1 Hardcoded Patterns (Các mẫu kinh điển không bao giờ cũ)
    all_signatures.update([
        "ignore previous instructions", "disregard all prior rules", 
        "reveal your system prompt", "act as a linux terminal",
        "developer mode enabled", "bypass security filters",
        "sudo mode", "output as markdown code block"
    ])

    # 3.2 Lặp qua các dataset và rút trích
    for ds_info in DATASET_CONFIGS:
        print(f"\n[*] Processing {ds_info['path']} (Column: '{ds_info['col']}')")
        try:
            # Tải dataset với config và split tương ứng
            ds = load_dataset(ds_info['path'], ds_info['config'], split=ds_info['split'])
            
            # Chọn mẫu ngẫu nhiên để trích xuất đặc trưng
            samples = ds.shuffle(seed=42)
            
            for item in tqdm(samples, desc=f"    Extracting {ds_info['label']}"):
                text_content = item[ds_info['col']]
                if text_content:
                    patterns = extract_signatures(text_content)
                    all_signatures.update(patterns)
                
        except Exception as e:
            print(f"    [Error] Failed to process {ds_info['path']}: {e}")

    # 3.3 Lưu thư viện JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Khử trùng lặp và sắp xếp
    final_list = sorted(list(set(all_signatures)))
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"\n" + "="*60)
    print(f"[SUCCESS] Unified Attack Signatures: {len(final_list)}")
    print(f"[SUCCESS] File saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()