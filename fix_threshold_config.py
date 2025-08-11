#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复阈值配置功能
"""

import re
from pathlib import Path

def fix_training_script():
    """修改训练脚本使用可配置阈值"""
    script_path = Path('./lightgbm_stock_train.py')
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换硬编码的0.5阈值
    original_code = '''                # 转换为类别预测（概率 > 0.5 为看多）
                y_train_pred = (y_train_pred_proba > 0.5).astype(int)
                y_val_pred = (y_val_pred_proba > 0.5).astype(int)
                y_test_pred = (y_test_pred_proba > 0.5).astype(int)'''
    
    replacement_code = '''                # 获取分类阈值配置
                eval_config = self.config.get('evaluation', {})
                threshold = eval_config.get('classification_threshold', 0.5)
                self.logger.info(f"   🎯 使用分类阈值: {threshold}")
                
                # 转换为类别预测（概率 > threshold 为看多）
                y_train_pred = (y_train_pred_proba > threshold).astype(int)
                y_val_pred = (y_val_pred_proba > threshold).astype(int)
                y_test_pred = (y_test_pred_proba > threshold).astype(int)'''
    
    new_content = content.replace(original_code, replacement_code)
    
    # 保存修改
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ lightgbm_stock_train.py - 已修改为使用可配置阈值")

def fix_visualization_script():
    """修改可视化脚本使用可配置阈值"""
    script_path = Path('./visualization_extension.py')
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 在__init__方法中添加threshold属性
    if 'self.threshold = 0.5' not in content:
        init_pattern = r'(def __init__\(self, results_dir\):\s*\n\s*self\.results_dir = Path\(results_dir\)\s*\n\s*self\.results_dir\.mkdir\(parents=True, exist_ok=True\)\s*\n\s*self\.training_history = \{\'train\': \[\], \'val\': \[\]\})'
        init_replacement = r'\1\n        self.threshold = 0.5  # 默认阈值，可由外部设置'
        content = re.sub(init_pattern, init_replacement, content)
    
    # 2. 替换硬编码的0.5阈值
    replacements = [
        # 分类阈值线
        ("ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='分类阈值')",
         "ax3.axvline(x=self.threshold, color='black', linestyle='--', alpha=0.8, label=f'分类阈值({self.threshold})')"),
        
        # 默认阈值线  
        ("ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='默认阈值')",
         "ax3.axvline(x=self.threshold, color='black', linestyle='--', alpha=0.7, label=f'分类阈值({self.threshold})')"),
        
        # 置信度计算
        ("confidence = np.abs(y_pred_proba - 0.5) * 2",
         "confidence = np.abs(y_pred_proba - self.threshold) * 2"),
        
        # 二分类预测
        ("y_train_pred_binary = (y_train_pred_proba > 0.5).astype(int)",
         "y_train_pred_binary = (y_train_pred_proba > self.threshold).astype(int)"),
        ("y_val_pred_binary = (y_val_pred_proba > 0.5).astype(int)",
         "y_val_pred_binary = (y_val_pred_proba > self.threshold).astype(int)"),
        ("y_test_pred_binary = (y_test_pred_proba > 0.5).astype(int)",
         "y_test_pred_binary = (y_test_pred_proba > self.threshold).astype(int)")
    ]
    
    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True
    
    # 保存修改
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if modified:
        print("✅ visualization_extension.py - 已修改为使用可配置阈值")
    else:
        print("⚠️ visualization_extension.py - 部分修改可能已存在")

def add_threshold_passing():
    """在训练器中添加阈值传递给可视化器"""
    script_path = Path('./lightgbm_stock_train.py')
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找可视化器初始化的位置并添加阈值传递
    if 'self.visualizer.threshold = threshold' not in content:
        old_init = 'self.visualizer = LightGBMVisualizer(self.results_save_dir)'
        new_init = '''self.visualizer = LightGBMVisualizer(self.results_save_dir)
            # 传递阈值配置给可视化器
            eval_config = self.config.get('evaluation', {})
            threshold = eval_config.get('classification_threshold', 0.5)
            self.visualizer.threshold = threshold'''
        
        if old_init in content:
            content = content.replace(old_init, new_init)
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ 已添加阈值传递给可视化器")
        else:
            print("⚠️ 未找到可视化器初始化代码")
    else:
        print("✅ 阈值传递功能已存在")

def main():
    print("🔧 开始修复可配置阈值功能...")
    
    # 1. 修改训练脚本
    fix_training_script()
    
    # 2. 修改可视化脚本
    fix_visualization_script()
    
    # 3. 添加阈值传递
    add_threshold_passing()
    
    print("\n🎉 可配置阈值功能修复完成！")
    print("\n📋 现在您可以在配置文件中设置阈值：")
    print("   evaluation:")
    print("     classification_threshold: 0.6  # 设置为0.6")
    print("\n💡 阈值说明：")
    print("   - 0.3: 激进策略，更容易预测看多")
    print("   - 0.5: 均衡策略，默认值") 
    print("   - 0.7: 保守策略，只在高概率时预测看多")

if __name__ == "__main__":
    main()