import os
from datasets import load_dataset, concatenate_datasets

# 1. Load processed datasets
ds_refactored_dolly = load_dataset("json", data_files="./processed_datasets/corporate_dolly_v2.jsonl")["train"]
ds_refactored_deepset = load_dataset("json", data_files="./processed_datasets/deepset_prompt_injections_refactored.jsonl")["train"]

# 2. Train test split - Sử dụng phương thức chính chủ của Hugging Face
# Nó trả về một DatasetDict chứa key 'train' và 'test'
split_dolly = ds_refactored_dolly.train_test_split(test_size=0.2, seed=42)
train_dolly = split_dolly["train"]
test_dolly = split_dolly["test"]

# 3. Định nghĩa các hàm convert để align cột
def convert_dataset_dolly(example):
    return {
        "instruction": example["instruction"],
        "label": 0,
        "category": 0,  # Benign
        "note": "benign"
    }

def convert_dataset_deepset(example):
    return {
        "instruction": example["text"],
        "label": example["original_label"], 
        "category": 1,  # Suspicious
        "note": "suspicious"
    }

# 4. Map dữ liệu
ds_refactored_deepset = ds_refactored_deepset.map(convert_dataset_deepset)
train_dolly = train_dolly.map(convert_dataset_dolly)
test_dolly = test_dolly.map(convert_dataset_dolly)

# 5. Xóa các cột thừa để đảm bảo 2 dataset khớp cấu trúc trước khi gộp
# Điều này cực kỳ quan trọng để tránh lỗi 'Features mismatch'
target_columns = ["instruction", "label", "category", "note"]
train_dataset = train_dolly.remove_columns([c for c in train_dolly.column_names if c not in target_columns])
test_dolly = test_dolly.remove_columns([c for c in test_dolly.column_names if c not in target_columns])
ds_refactored_deepset = ds_refactored_deepset.remove_columns([c for c in ds_refactored_deepset.column_names if c not in target_columns])

# 6. Combine datasets sử dụng concatenate_datasets
# Gộp phần test của Dolly với toàn bộ Deepset để làm tập Benchmark
test_dataset = concatenate_datasets([test_dolly, ds_refactored_deepset])

# 7. Save to jsonl
os.makedirs("./processed_datasets", exist_ok=True)
train_dataset.to_json("./processed_datasets/train_corporate_dolly_v2.jsonl")
test_dataset.to_json("./processed_datasets/test_corporate_dolly_v2.jsonl")

print(f"[*] Done! Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")