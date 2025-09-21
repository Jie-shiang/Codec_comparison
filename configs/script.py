import json
import os

def transform_json_structure(input_file, output_file):
    """
    讀取JSON檔案並轉換其結構
    
    Args:
        input_file (str): 輸入JSON檔案路徑
        output_file (str): 輸出JSON檔案路徑
    """
    
    # 讀取原始JSON檔案
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_file}")
        return
    except json.JSONDecodeError:
        print(f"錯誤：{input_file} 不是有效的JSON格式")
        return
    
    # 處理每個測試集
    for test_set_name, test_set_data in data.items():
        # 跳過 model_info
        if test_set_name == "model_info":
            continue
            
        # 處理測試集內的樣本
        sample_counter = 1
        error_sample_counter = 1
        new_test_set_data = {}
        
        for original_key, sample_data in test_set_data.items():
            # 跳過 Total
            if original_key == "Total":
                new_test_set_data[original_key] = sample_data
                continue
            
            # 判斷是否為錯誤樣本
            if original_key.startswith("Error_Sample"):
                new_key = f"Error_Sample_{error_sample_counter}"
                error_sample_counter += 1
            else:
                new_key = f"Sample_{sample_counter}"
                sample_counter += 1
            
            # 建立新的樣本結構
            new_sample_data = {"File_name": original_key}
            
            # 複製原有的資料，或者如果原本就是N/A則使用Sample_x作為檔名
            if (sample_data.get("Transcription") == "N/A" or 
                sample_data.get("dWER") == "N/A"):
                new_sample_data["File_name"] = new_key
            
            # 複製其他欄位
            for field in ["Transcription", "dWER", "UTMOS", "PESQ", "STOI"]:
                new_sample_data[field] = sample_data.get(field, "N/A")
            
            new_test_set_data[new_key] = new_sample_data
        
        # 更新原始資料
        data[test_set_name] = new_test_set_data
    
    # 寫入轉換後的JSON檔案
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"轉換完成！結果已儲存到 {output_file}")
    except Exception as e:
        print(f"儲存檔案時發生錯誤：{e}")

def main():
    """主函數"""
    # 要處理的檔案清單
    files_to_process = [
        "/home/jieshiang/Desktop/GitHub/Codec_comparison/configs/BigCodec_80Hz_config.json",
        "/home/jieshiang/Desktop/GitHub/Codec_comparison/configs/FocalCodec_12.5Hz_config.json",
        "/home/jieshiang/Desktop/GitHub/Codec_comparison/configs/FocalCodec_25Hz_config.json",
        "/home/jieshiang/Desktop/GitHub/Codec_comparison/configs/FocalCodec_50Hz_config.json"
    ]
    
    # 處理每個檔案
    for input_file in files_to_process:
        # 檢查輸入檔案是否存在
        if not os.path.exists(input_file):
            print(f"警告：找不到檔案 {input_file}，跳過處理")
            continue
        
        # 生成輸出檔案名稱（在原檔案名後加上_converted）
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.json"
        
        print(f"正在處理：{input_file}")
        
        # 執行轉換
        transform_json_structure(input_file, output_file)

if __name__ == "__main__":
    main()