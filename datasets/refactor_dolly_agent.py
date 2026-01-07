import os
import json
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load môi trường
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. MAPPING ROLES TO ACTUAL TOOLS ---
# Dựa trên file tools.py của bạn, ta chia thành 3 nhóm năng lực thực tế
AGENT_CONFIGS = {
    "Communication_Specialist": {
        "role_name": "Mail & Communication Manager",
        "tools_involved": ["search_emails", "send_email", "reply_all_email", "forward_email"],
        "context": "Professional email management, handling threads, and internal correspondence."
    },
    "Knowledge_Officer": {
        "role_name": "Knowledge Retrieval & RAG Officer",
        "tools_involved": ["query_memory", "read_pdf"],
        "context": "Searching internal policy databases, RAG systems, and corporate knowledge bases."
    },
    "Data_Analyst": {
        "role_name": "Document & Data Processing Specialist",
        "tools_involved": ["read_docx", "create_xlsx", "read_xlsx", "create_docx"],
        "context": "Transforming unstructured text into structured office formats and extracting data from files."
    }
}

def refactor_instruction_v2(instruction, agent_type):
    config = AGENT_CONFIGS[agent_type]
    
    # Prompt nâng cao để khử lặp và tăng tính tự nhiên
    prompt = f"""
    [Task]
    Refactor the 'Original Instruction' into a professional task for an AI Agent role: '{config['role_name']}'.
    The Agent uses these tools: {', '.join(config['tools_involved'])}.

    [Strict Guidelines to avoid Clichés]
    1. NO "PLEASE" OVERUSE: Avoid starting every sentence with "Please". Use direct imperatives (e.g., "Search for...", "Analyze...", "Generate...") or situational inquiries.
    2. VARY STRUCTURE: Mix short direct commands with complex multi-step requests.
    3. CORE INTENT: Keep the original question/task from Dolly but wrap it in the agent's context ({config['context']}).
    4. NO ROBOTIC PREFACE: Do not use "As an AI...", "Here is the task...". Return ONLY the refactored text.
    5. TERMINOLOGY: Use professional terms like 'thread', 'record', 'database query', 'structured export', or 'internal policy'.

    Original Instruction: {instruction}
    
    Refactored Task for {config['role_name']}:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior workflow engineer. You rewrite instructions to be natural, professional, and diverse in tone. You never repeat the same polite fillers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85, # Tăng temperature để đa dạng hóa văn phong
            max_tokens=350
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Error at: {e}")
        return None

# --- 2. EXECUTION PIPELINE ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="Number of records to refactor")
    parser.add_argument("--output", type=str, default="corporate_dolly_v2.jsonl")
    args = parser.parse_args()

    print(f"[*] Loading Databricks-Dolly-15K...")
    dataset = load_dataset("databricks/databricks-dolly-15k")["train"]
    
    # Xáo trộn để lấy mẫu ngẫu nhiên từ nhiều category khác nhau
    raw_data = list(dataset)
    random.shuffle(raw_data)
    
    selected_samples = raw_data[:args.samples]

    results = []
    print(f"[*] Refactoring {args.samples} instructions with Role-Tool mapping...")

    for item in tqdm(selected_samples):
        # Chọn ngẫu nhiên 1 trong 3 nhóm năng lực của Agent
        agent_type = random.choice(list(AGENT_CONFIGS.keys()))
        
        refactored_text = refactor_instruction_v2(item['instruction'], agent_type)
        
        if refactored_text:
            results.append({
                "agent_role": AGENT_CONFIGS[agent_type]["role_name"],
                "instruction": refactored_text,
                "original_intent": item['instruction'],
                "category": item['category'],
                "tools_potential": AGENT_CONFIGS[agent_type]["tools_involved"]
            })

    # Lưu file
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[OK] Success! Generated {len(results)} samples.")
    print(f"[OK] View your data in: {args.output}")

if __name__ == "__main__":
    main()