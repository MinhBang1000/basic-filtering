import os
import json
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. ARGUMENT FOR FLEXIBILITY ---
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=100, help="Number of records to refactor")
args = parser.parse_args()

# --- 2. AGENT PERSONAS (Based on your Agent description) ---
AGENT_ROLES = [
    "Email Operations Assistant (Search, Reply, Forward)",
    "RAG Knowledge Specialist (Policy & Workflow retrieval)",
    "Document Processing Agent (PDF/DOCX to Structured Data)",
    "Operational Secretary (Policy-aware task execution)"
]

def refactor_for_agent(instruction, role):
    # Prompt được tinh chỉnh để ép output cực sạch và giữ nguyên ý đồ gốc
    prompt = f"""
    [Task]
    Refactor the following 'Original Instruction' into a professional corporate task for a '{role}'.
    
    [Guidelines]
    1. PRESERVE CORE INTENT: Do not change the fundamental question, request, 
    or goal of the original instruction.
    2. PROFESSIONAL CONTEXT: Rewrite it as a formal workplace request 
    (e.g., an email task, a policy query, or a report processing step).
    3. NO CONVERSATION: Return ONLY the refactored text. 
    4. NO PREAMBLE/POSTAMBLE: Do not include "Here is...", "Refactored task:", or any quotes.

    Original Instruction: {instruction}
    
    Refactored Instruction for {role}:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                # Đưa chỉ dẫn quan trọng vào system role để tăng độ tuân thủ
                {"role": "system", "content": "You are a professional data annotator. You only output the final refactored text without any explanation or conversational filler."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Giữ độ biến hóa nhưng vẫn bám sát intent
            max_tokens=300
        )
        return response.choices[0].message.content.strip().replace('"', '') # Xóa bỏ dấu ngoặc kép nếu có
    except Exception as e:
        return None
        return None

# --- 3. EXECUTION ---
print(f"Loading Dolly-15K and refactoring {args.samples} records...")
ds = load_dataset("databricks/databricks-dolly-15k")["train"]
data = list(ds)[:args.samples]

output = []
for item in tqdm(data):
    role = random.choice(AGENT_ROLES)
    refactored = refactor_for_agent(item['instruction'], role)
    if refactored:
        output.append({
            "agent_role": role,
            "instruction": refactored,
            "original": item['instruction']
        })

# Save results
out_file = "corporate_dolly_benign.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for entry in output:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n[SUCCESS] Refactored {len(output)} records saved to {out_file}")