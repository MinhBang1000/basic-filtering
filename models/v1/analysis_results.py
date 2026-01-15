import pandas as pd

def analyze_errors(csv_path):
    # Đọc file kết quả
    df = pd.read_csv(csv_path)
    
    # 1. Lọc False Negatives (FN): Tấn công thật nhưng hệ thống cho qua (Nguy hiểm nhất)
    # Điều kiện: gt_system == 1 VÀ pred_system == 0
    fn_samples = df[(df['gt_system'] == 1) & (df['pred_system'] == 0)]
    
    # 2. Lọc False Positives (FP): Câu lệnh sạch nhưng hệ thống chặn nhầm
    # Điều kiện: gt_system == 0 VÀ pred_system == 1
    fp_samples = df[(df['gt_system'] == 0) & (df['pred_system'] == 1)]
    
    # Xuất thông báo tóm tắt
    print(f"[*] Phân tích file: {csv_path}")
    print(f"[-] Số lượng False Negatives (Lọt lưới): {len(fn_samples)}")
    print(f"[-] Số lượng False Positives (Chặn nhầm): {len(fp_samples)}")
    
    # Lưu ra file riêng để bạn dễ đọc nội dung
    fn_samples.to_csv("analysis_fn_samples.csv", index=False)
    fp_samples.to_csv("analysis_fp_samples.csv", index=False)
    
    return fn_samples, fp_samples

# Chạy hàm phân tích
fn, fp = analyze_errors("integrated_system_benchmark_results.csv")