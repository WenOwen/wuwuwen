import pandas as pd
import os
import glob

def clean_stock_data():
    """
    删除datas_em文件夹中所有CSV文件的指定列
    删除的列：所属行业,概念板块,地区,总股本,流通股,每股收益,每股净资产
    """
    # 要删除的列名
    columns_to_remove = ['所属行业', '概念板块', '地区', '总股本', '流通股', '每股收益', '每股净资产']
    
    # 获取datas_em文件夹中的所有CSV文件
    csv_files = glob.glob('datas_em/*.csv')
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    processed_count = 0
    error_count = 0
    
    for file_path in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查哪些列存在
            existing_columns = [col for col in columns_to_remove if col in df.columns]
            
            if existing_columns:
                # 删除指定的列
                df = df.drop(columns=existing_columns)
                
                # 保存修改后的文件
                df.to_csv(file_path, index=False)
                
                print(f"✓ {os.path.basename(file_path)}: 删除了 {len(existing_columns)} 列")
                processed_count += 1
            else:
                print(f"- {os.path.basename(file_path)}: 没有找到需要删除的列")
                
        except Exception as e:
            print(f"✗ {os.path.basename(file_path)}: 处理失败 - {str(e)}")
            error_count += 1
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")

if __name__ == "__main__":
    clean_stock_data() 