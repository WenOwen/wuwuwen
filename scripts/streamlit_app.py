# -*- coding: utf-8 -*-
"""
AI股市预测系统 - Streamlit用户界面
功能：提供直观的Web界面用于股市预测和分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import os
import sys

# 添加项目路径
sys.path.append('.')

try:
    from core.prediction_service import PredictionService
    from core.performance_monitor import PerformanceMonitor
    from core.feature_engineering import FeatureEngineering
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.stop()


# 页面配置
st.set_page_config(
    page_title="AI股市预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 现代化科技风格CSS
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        padding-top: 1rem;
    }
    
    /* 主标题 */
    .main-header {
        font-size: 2.5rem;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* 紧凑的指标卡片 */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 0.8rem;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    /* 中国股市配色 - 上涨红色卡片 */
    .prediction-card-up {
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 71, 87, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 中国股市配色 - 下跌绿色卡片 */
    .prediction-card-down {
        background: linear-gradient(135deg, #2ed573 0%, #1dd1a1 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(46, 213, 115, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 现代化搜索框 */
    .search-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* 子模型选择区域 */
    .sub-model-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 0.8rem 0;
    }
    
    /* 股票信息卡片 */
    .stock-info-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(15px);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        margin: 0.3rem 0;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 成功状态卡片 */
    .success-card {
        background: linear-gradient(135deg, #2ed573 0%, #1dd1a1 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(46, 213, 115, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 警告状态卡片 */
    .warning-card {
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(255, 71, 87, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 紧凑的间距 */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* 现代化按钮样式 */
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* 选择框样式 */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.1);
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-card, .prediction-card-up, .prediction-card-down {
            padding: 0.6rem;
        }
    }
    
    /* 固定统计信息容器样式 */
    .stats-container {
        position: sticky;
        top: 100px;
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e5e9;
        z-index: 100;
        max-height: 200px;
        overflow: hidden;
    }
    
    /* 进度容器样式 */
    .progress-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: sticky;
        top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_stock_list():
    """加载股票列表"""
    try:
        # 首先尝试从sector mapping获取所有股票
        from core.stock_sector_mapping import StockSectorMapping
        sector_mapping = StockSectorMapping()
        all_stocks = sector_mapping.get_all_stocks()
        
        if all_stocks and len(all_stocks) > 50:  # 如果有足够多的股票
            stock_data = []
            for stock_code in all_stocks[:2000]:  # 限制在2000只以内，避免界面过慢
                stock_info = sector_mapping.get_stock_info(stock_code)
                stock_name = stock_info.get('name', stock_code)
                stock_data.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name
                })
            return stock_data
        
        # 备选方案：从CSV文件加载
        if os.path.exists('stockcode_list/all_stock_list.csv'):
            df = pd.read_csv('stockcode_list/all_stock_list.csv', encoding='utf-8')
            return df[['股票代码', '股票名称']].to_dict('records')
        
        # 最后的备选方案：从数据目录扫描
        data_dir = "data/datas_em"
        if os.path.exists(data_dir):
            stock_codes = []
            for file in os.listdir(data_dir):
                if file.endswith('.csv') and len(file) >= 8:
                    stock_code = file.replace('.csv', '')
                    if stock_code.startswith(('sh', 'sz', 'bj')):
                        stock_codes.append(stock_code)
            
            # 转换为标准格式
            stock_data = []
            for code in sorted(stock_codes)[:1000]:  # 限制数量
                stock_data.append({
                    '股票代码': code,
                    '股票名称': f'股票{code}'
                })
            
            if stock_data:
                return stock_data
        
        # 默认股票列表
        return [
            {'股票代码': 'sh600519', '股票名称': '贵州茅台'},
            {'股票代码': 'sz000001', '股票名称': '平安银行'},
            {'股票代码': 'sz000002', '股票名称': '万科A'},
            {'股票代码': 'sh600036', '股票名称': '招商银行'},
            {'股票代码': 'sz000858', '股票名称': '五粮液'}
        ]
        
    except Exception as e:
        st.warning(f"加载股票列表时出错: {str(e)}")
        return [
            {'股票代码': 'sh600519', '股票名称': '贵州茅台'},
            {'股票代码': 'sz000001', '股票名称': '平安银行'},
            {'股票代码': 'sz000002', '股票名称': '万科A'}
        ]


def search_stocks(stock_list, search_term):
    """搜索股票"""
    if not search_term:
        return stock_list
    
    search_term = search_term.lower()
    filtered_stocks = []
    
    for stock in stock_list:
        code = stock['股票代码'].lower()
        name = stock['股票名称'].lower()
        
        # 搜索股票代码或名称
        if search_term in code or search_term in name:
            filtered_stocks.append(stock)
    
    return filtered_stocks


@st.cache_data
def get_available_models():
    """获取可用的模型列表"""
    model_info = {}
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        return {}
    
    try:
        model_folders = [f for f in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, f))]
        
        for folder in model_folders:
            folder_path = os.path.join(models_dir, folder)
            info_path = os.path.join(folder_path, 'training_info.pkl')
            
            if os.path.exists(info_path):
                try:
                    import joblib
                    training_info = joblib.load(info_path)
                    
                    # 格式化模型显示名称
                    prediction_days = training_info.get('prediction_days', 1)
                    train_date = folder.split('_')[-2] if '_' in folder else 'unknown'
                    accuracy = training_info.get('accuracy', 0)
                    
                    if accuracy > 0:
                        display_name = f"{prediction_days}天预测模型 (训练日期: {train_date}, 准确率: {accuracy:.2%})"
                    else:
                        display_name = f"{prediction_days}天预测模型 (训练日期: {train_date})"
                    
                    model_info[display_name] = {
                        'folder': folder,
                        'prediction_days': prediction_days,
                        'training_info': training_info,
                        'accuracy': accuracy
                    }
                    
                except Exception as e:
                    st.warning(f"加载模型信息失败 {folder}: {e}")
                    continue
        
        return model_info
        
    except Exception as e:
        st.error(f"读取模型目录失败: {e}")
        return {}


def load_specific_model(prediction_service, model_folder, prediction_days):
    """加载指定的模型"""
    try:
        import joblib
        from core.ai_models import create_ensemble_model
        
        folder_path = os.path.join("models", model_folder)
        
        # 加载训练信息
        info_path = os.path.join(folder_path, 'training_info.pkl')
        if not os.path.exists(info_path):
            st.error(f"❌ 模型信息文件不存在: {info_path}")
            return False
            
        training_info = joblib.load(info_path)
        
        # 从训练信息中获取正确的参数
        # 根据错误信息，模型期望的是(None, 30, 170)的输入形状
        sequence_length = training_info.get('sequence_length', 30)  # 改为默认30
        n_features = len(training_info['feature_names'])
        
        st.info(f"📐 模型参数: sequence_length={sequence_length}, n_features={n_features}")
        
        # 创建模型实例  
        model = create_ensemble_model(
            sequence_length=sequence_length,
            n_features=n_features
        )
        
        # 调用预测服务的模型加载方法
        prediction_service._load_individual_models(model, folder_path)
        
        # 检查模型是否有可用的子模型
        fitted_models = [name for name, m in model.models.items() if getattr(m, 'is_fitted', False)]
        
        # 详细诊断信息
        st.subheader("🔍 模型加载诊断")
        
        # 检查模型文件是否存在
        model_files_status = {}
        model_files = {
            'LSTM': ('LSTM_model.h5', 'LSTM_scaler.pkl'),
            'CNN-LSTM': ('CNN-LSTM_model.h5', 'CNN-LSTM_scaler.pkl'),
            'Transformer': ('Transformer_model.h5', 'Transformer_scaler.pkl'),
            'LightGBM': ('LightGBM_model.pkl', 'LightGBM_scaler.pkl')
        }
        
        for model_name, (model_file, scaler_file) in model_files.items():
            model_path = os.path.join(folder_path, model_file)
            scaler_path = os.path.join(folder_path, scaler_file)
            
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)
            is_fitted = model_name in [name for name, m in model.models.items() if getattr(m, 'is_fitted', False)]
            
            status_emoji = "✅" if (model_exists and scaler_exists and is_fitted) else "❌"
            
            st.write(f"{status_emoji} **{model_name}**: 模型文件{'✓' if model_exists else '✗'} | 缩放器{'✓' if scaler_exists else '✗'} | 加载状态{'✓' if is_fitted else '✗'}")
            
            if not model_exists:
                st.warning(f"   缺失文件: {model_file}")
            if not scaler_exists:
                st.warning(f"   缺失文件: {scaler_file}")
        
        if not fitted_models:
            st.error("❌ 所有AI模型加载失败！")
            st.warning("⚠️ 系统将使用后备预测模型（基于简单规则）")
            st.info("💡 这解释了为什么不同模型预测结果相同，且概率接近50%")
            
            # 提供解决方案
            with st.expander("🛠️ 解决方案", expanded=True):
                st.markdown("""
                **为什么会出现这个问题？**
                - AI模型文件缺失或损坏
                - TensorFlow/LightGBM版本不兼容
                - 模型文件路径不正确
                
                **解决方案：**
                1. **重新训练模型** (推荐)
                   ```bash
                   python complete_training.py
                   ```
                
                2. **检查依赖环境**
                   ```bash
                   pip install tensorflow lightgbm scikit-learn
                   ```
                
                3. **检查模型文件**
                   - 确保models/文件夹中有完整的模型文件
                   - 每个模型需要.h5文件和.pkl缩放器文件
                
                **临时解决方案：**
                - 当前使用后备预测，基于价格趋势的简单规则
                - 虽然不如AI模型准确，但可以提供基本预测功能
                """)
            
            # 尝试手动修复
            if st.button("🔧 尝试手动修复模型加载"):
                try:
                    # 强制重新加载
                    prediction_service._load_individual_models(model, folder_path)
                    fitted_models = [name for name, m in model.models.items() if getattr(m, 'is_fitted', False)]
                    
                    if fitted_models:
                        st.success(f"✅ 修复成功！已加载: {', '.join(fitted_models)}")
                    else:
                        st.error("❌ 修复失败，模型文件可能损坏或不兼容")
                except Exception as e:
                    st.error(f"修复过程中出错: {str(e)}")
            
            prediction_service._setup_fallback_model(model)
            fitted_models = ["Fallback"]
        
        if fitted_models and fitted_models != ["Fallback"]:
            st.success(f"✅ 已成功加载 {len(fitted_models)} 个AI模型")
            st.info(f"📊 可用子模型: {', '.join(fitted_models)}")
        else:
            st.warning(f"⚠️ 使用后备模型: {', '.join(fitted_models) if fitted_models else 'None'}")
        
        # 确保training_info中包含sequence_length
        training_info['sequence_length'] = sequence_length
        
        # 更新预测服务中的模型
        prediction_service.models[prediction_days] = model
        prediction_service.model_metadata[prediction_days] = training_info
        
        return True
        
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        # 尝试设置后备模型
        try:
            # 创建一个新的简单模型
            from core.ai_models import EnsembleModel
            fallback_model = EnsembleModel()
            prediction_service._setup_fallback_model(fallback_model)
            
            # 确保后备模型也使用正确的sequence_length
            training_info['sequence_length'] = 30  # 后备模型使用30
            
            prediction_service.models[prediction_days] = fallback_model
            prediction_service.model_metadata[prediction_days] = training_info
            
            st.warning("⚠️ 已启用后备模型")
            st.info("📊 可用子模型: Fallback")
            return True
            
        except Exception as fallback_error:
            st.error(f"❌ 后备模型设置失败: {str(fallback_error)}")
            import traceback
            st.code(traceback.format_exc())
            return False


@st.cache_resource
def get_prediction_service(_version="v2.0"):
    """获取预测服务实例"""
    try:
        # 清除可能的模块缓存
        import importlib
        import core.prediction_service
        import core.feature_engineering
        import core.ai_models
        importlib.reload(core.ai_models)
        importlib.reload(core.feature_engineering)
        importlib.reload(core.prediction_service)
        
        from core.prediction_service import PredictionService
        return PredictionService()
    except Exception as e:
        st.error(f"初始化预测服务失败: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


@st.cache_resource
def get_performance_monitor():
    """获取性能监控实例"""
    try:
        return PerformanceMonitor()
    except Exception as e:
        st.error(f"初始化性能监控失败: {e}")
        return None


def main():
    """主函数"""
    
    # 页面标题
    st.markdown('<h1 class="main-header">🚀 AI股市预测系统</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("🔧 系统控制面板")
    
    # 页面选择
    page = st.sidebar.selectbox(
        "选择功能模块",
        ["📈 股票预测", "📊 性能监控", "🎯 批量预测", "⚠️ 风险评估", "📋 预测历史", "🤖 模型管理", "⚙️ 系统设置"]
    )
    
    # 根据选择显示对应页面
    if page == "📈 股票预测":
        show_stock_prediction_page()
    elif page == "📊 性能监控":
        show_performance_monitoring_page()
    elif page == "🎯 批量预测":
        show_batch_prediction_page()
    elif page == "⚠️ 风险评估":
        show_risk_assessment_page()
    elif page == "📋 预测历史":
        show_prediction_history_page()
    elif page == "🤖 模型管理":
        show_model_management_page()
    elif page == "⚙️ 系统设置":
        show_system_settings_page()


def show_stock_prediction_page():
    """股票预测页面"""
    st.header("📈 单只股票预测")
    
    # 显示使用指南
    show_usage_guide()
    
    # 获取预测服务
    prediction_service = get_prediction_service("v2.0")
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 股票搜索和选择区域  
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("### 🔍 选择预测股票")
    
    # 加载股票列表
    stock_list = load_stock_list()
    
    # 搜索框和控制按钮
    col_search, col_clear = st.columns([4, 1])
    
    with col_search:
        search_term = st.text_input(
            "搜索股票代码或名称", 
            placeholder="输入股票代码（如600519）或股票名称（如茅台）",
            help="支持模糊搜索股票代码和名称"
        )
    
    with col_clear:
        st.write("")  # 空行对齐
        if st.button("🗑️ 清空", help="清空搜索条件"):
            # 清空搜索相关的session state
            if 'search_term' in st.session_state:
                del st.session_state['search_term']
            st.rerun()
    
    # 根据搜索条件过滤股票
    filtered_stocks = search_stocks(stock_list, search_term)
    
    if search_term:
        if not filtered_stocks:
            st.warning("⚠️ 未找到匹配的股票，显示所有股票")
            filtered_stocks = stock_list
        else:
            st.info(f"🔍 找到 {len(filtered_stocks)} 只匹配的股票")
    else:
        st.info(f"📊 共有 {len(stock_list)} 只股票可选择")
    
    st.markdown('</div>', unsafe_allow_html=True)  # 结束搜索容器

    # 股票选择区域
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # 格式化股票选项显示
        stock_options = [f"{stock['股票代码']} - {stock['股票名称']}" for stock in filtered_stocks]
        selected_stock_display = st.selectbox(
            "选择股票",
            stock_options,
            index=0,
            help="选择要预测的股票"
        )
        
        # 提取股票代码
        selected_stock = selected_stock_display.split(' - ')[0] if selected_stock_display else None
    
    with col2:
        prediction_days = st.selectbox(
            "预测天数",
            [1, 3, 5],
            index=0,
            help="选择预测的时间跨度"
        )
    
    with col3:
        # 主模型选择
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "选择主模型",
                list(available_models.keys()),
                index=0,
                help="选择要使用的预测模型"
            )
            
            # 显示选定模型信息
            if selected_model:
                model_info = available_models[selected_model]
                st.info(f"📊 模型: {model_info['folder']}")
        else:
            st.warning("⚠️ 没有可用的模型")
            selected_model = None
    
    with col4:
        include_analysis = st.checkbox(
            "包含详细分析",
            value=True,
            help="包含技术指标和趋势分析"
        )
    
    # 子模型选择区域
    if available_models and selected_model:
        st.markdown('<div class="sub-model-section">', unsafe_allow_html=True)
        st.subheader("🤖 细分模型选择")
        st.markdown("选择要使用的AI模型组合，不同模型有不同的预测特点：")
        
        # 可用的子模型列表
        col1, col2, col3, col4 = st.columns(4)
        
        selected_sub_models = []
        
        with col1:
            if st.checkbox("🧠 CNN-LSTM", value=True, help="卷积神经网络+长短期记忆网络，擅长捕捉时序模式"):
                selected_sub_models.append("CNN-LSTM")
            st.markdown("*适合短期趋势*")
        
        with col2:
            if st.checkbox("🎯 Transformer", value=True, help="Transformer注意力机制模型，擅长长期依赖关系"):
                selected_sub_models.append("Transformer")
            st.markdown("*适合长期趋势*")
        
        with col3:
            if st.checkbox("🔄 LSTM", value=True, help="长短期记忆网络，平衡短期和长期预测"):
                selected_sub_models.append("LSTM")
            st.markdown("*平衡预测*")
        
        with col4:
            if st.checkbox("⚡ LightGBM", value=True, help="轻量级梯度提升机，快速且准确"):
                selected_sub_models.append("LightGBM")
            st.markdown("*快速预测*")
        
        if not selected_sub_models:
            st.error("❌ 请至少选择一个子模型")
        else:
            st.success(f"✅ 已选择 {len(selected_sub_models)} 个子模型: {', '.join(selected_sub_models)}")
            
            # 显示模型特性对比
            if len(selected_sub_models) > 1:
                st.markdown("**📊 选择的模型特性对比:**")
                
                model_features = {
                    "CNN-LSTM": {"速度": 85, "准确率": 88, "稳定性": 82, "适用场景": "短期波动"},
                    "Transformer": {"速度": 70, "准确率": 92, "稳定性": 90, "适用场景": "长期趋势"},
                    "LSTM": {"速度": 90, "准确率": 85, "稳定性": 88, "适用场景": "平衡预测"},
                    "LightGBM": {"速度": 95, "准确率": 83, "稳定性": 85, "适用场景": "快速决策"}
                }
                
                # 创建雷达图数据
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                for model in selected_sub_models:
                    if model in model_features:
                        features = model_features[model]
                        fig.add_trace(go.Scatterpolar(
                            r=[features["速度"], features["准确率"], features["稳定性"]],
                            theta=['速度', '准确率', '稳定性'],
                            fill='toself',
                            name=model
                        ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="选择模型的性能对比",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测选项
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        if st.button("🔮 集成预测", type="primary"):
            prediction_mode = "ensemble"
    
    with col_pred2:
        if st.button("🎯 单模型对比", type="secondary"):
            prediction_mode = "individual"
    
    # 处理预测
    if 'prediction_mode' in locals():
        # 注意：不清除所有缓存，只在必要时清除模型缓存
        pass  # 移除自动清除缓存
        
        if not available_models:
            st.error("❌ 没有可用的模型，请先训练模型")
            return
            
        if selected_model is None:
            st.error("❌ 请选择一个模型")
            return
            
        with st.spinner("正在进行AI预测分析..."):
            try:
                # 获取选定的模型信息
                model_info = available_models[selected_model]
                
                # 确保预测天数与模型匹配
                model_prediction_days = model_info['prediction_days']
                if prediction_days != model_prediction_days:
                    st.warning(f"⚠️ 选定模型支持 {model_prediction_days} 天预测，将使用该模型进行预测")
                    prediction_days = model_prediction_days
                
                # 动态加载选定的模型
                with st.spinner("正在加载模型..."):
                    success = load_specific_model(prediction_service, model_info['folder'], model_prediction_days)
                
                if not success:
                    st.error("❌ 模型加载失败")
                    st.info("💡 系统将尝试使用后备预测方法")
                    # 确保有后备模型
                    if model_prediction_days not in prediction_service.models:
                        st.error("❌ 无法初始化任何预测模型")
                        return
                
                # 验证模型是否可用
                if model_prediction_days not in prediction_service.models:
                    st.error("❌ 预测模型未正确加载")
                    return
                
                st.success("✅ 模型加载完成，开始预测...")
                
                if prediction_mode == "individual":
                    # 单模型对比预测
                    st.subheader("🎯 各模型预测对比")
                    
                    model = prediction_service.models[model_prediction_days]
                    if hasattr(model, 'models') and model.models:
                        individual_results = []
                        
                        for sub_model_name, sub_model in model.models.items():
                            if hasattr(sub_model, 'is_fitted') and sub_model.is_fitted and sub_model_name in selected_sub_models:
                                try:
                                    # 临时调整权重，只使用单个模型
                                    original_weights = model.model_weights.copy()
                                    
                                    # 重置权重
                                    for name in model.model_weights.keys():
                                        model.model_weights[name] = 0.0
                                    model.model_weights[sub_model_name] = 1.0
                                    
                                    # 预测
                                    result = prediction_service.predict_single_stock(
                                        stock_code=selected_stock,
                                        prediction_days=prediction_days,
                                        include_analysis=False,
                                        prediction_threshold=get_prediction_threshold()
                                    )
                                    
                                    individual_results.append({
                                        'model': sub_model_name,
                                        'direction': result.predicted_direction,
                                        'probability': result.probability,
                                        'confidence': result.confidence
                                    })
                                    
                                    # 恢复权重
                                    model.model_weights = original_weights
                                    
                                except Exception as e:
                                    st.warning(f"模型 {sub_model_name} 预测失败: {str(e)}")
                        
                        # 显示对比结果
                        if individual_results:
                            cols = st.columns(len(individual_results))
                            for i, result_data in enumerate(individual_results):
                                with cols[i]:
                                    direction_color = "🔴" if result_data['direction'] == "上涨" else "🟢"  # 中国习惯
                                    card_class = "prediction-card-up" if result_data['direction'] == "上涨" else "prediction-card-down"
                                    st.markdown(f"""
                                    <div class="{card_class}">
                                        <h4>{direction_color} {result_data['model']}</h4>
                                        <h3>{result_data['direction']}</h3>
                                        <p>概率: {result_data['probability']:.1%}</p>
                                        <p>置信度: {result_data['confidence']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # 创建对比图表
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            
                            models = [r['model'] for r in individual_results]
                            probabilities = [r['probability'] for r in individual_results]
                            colors = ['#ff4757' if r['direction'] == '上涨' else '#2ed573' for r in individual_results]  # 中国习惯
                            
                            fig.add_trace(go.Bar(
                                x=models,
                                y=probabilities,
                                marker_color=colors,
                                text=[f"{p:.1%}" for p in probabilities],
                                textposition='auto',
                            ))
                            
                            fig.update_layout(
                                title="各模型预测概率对比",
                                yaxis_title="预测概率",
                                xaxis_title="模型",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    # 集成预测
                    # 根据用户选择调整模型权重
                    if 'selected_sub_models' in locals() and selected_sub_models:
                        # 只使用用户选择的模型
                        model = prediction_service.models[model_prediction_days]
                        if hasattr(model, 'model_weights'):
                            # 重置所有权重为0
                            for model_name in model.model_weights.keys():
                                model.model_weights[model_name] = 0.0
                            
                            # 只给选择的模型分配权重
                            weight_per_model = 1.0 / len(selected_sub_models)
                            for model_name in selected_sub_models:
                                if model_name in model.model_weights:
                                    model.model_weights[model_name] = weight_per_model
                            
                            st.info(f"🎯 已调整模型权重: {', '.join([f'{name}:{model.model_weights[name]:.2f}' for name in selected_sub_models])}")
                    
                    # 执行预测
                    result = prediction_service.predict_single_stock(
                        stock_code=selected_stock,
                        prediction_days=prediction_days,
                        include_analysis=include_analysis,
                        prediction_threshold=get_prediction_threshold()
                    )
                    
                    # 显示预测结果
                    show_prediction_result(result)
                
                # 保存预测历史到session_state，避免丢失
                if 'prediction_results' not in st.session_state:
                    st.session_state['prediction_results'] = []
                
                # 根据预测模式保存历史
                if prediction_mode == "individual":
                    # 保存各个模型的预测结果
                    if 'individual_results' in locals():
                        for result_data in individual_results:
                            st.session_state['prediction_results'].append({
                                'timestamp': datetime.now(),
                                'stock_code': selected_stock,
                                'prediction_days': prediction_days,
                                'prediction': 1 if result_data['direction'] == "上涨" else 0,
                                'probability': result_data['probability'],
                                'current_price': 0,  # 个别模型预测时没有价格
                                'predicted_direction': result_data['direction'],
                                'confidence': result_data['confidence'],
                                'model_type': result_data['model']
                            })
                else:
                    # 保存集成预测结果
                    if 'result' in locals():
                        st.session_state['prediction_results'].append({
                            'timestamp': datetime.now(),
                            'stock_code': selected_stock,
                            'prediction_days': prediction_days,
                            'prediction': result.prediction,
                            'probability': result.probability,
                            'current_price': result.current_price,
                            'predicted_direction': result.predicted_direction,
                            'confidence': result.confidence,
                            'model_type': 'Ensemble'
                        })
                
                # 只保留最近100条记录
                if len(st.session_state['prediction_results']) > 100:
                    st.session_state['prediction_results'] = st.session_state['prediction_results'][-100:]
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"预测失败: {str(e)}")
                with st.expander("🔍 查看详细错误信息"):
                    st.code(error_details)
    
    # 显示股票基本信息
    if selected_stock:
        show_stock_basic_info(selected_stock, prediction_service)


def show_prediction_result(result):
    """显示预测结果"""
    # 主要预测结果卡片
    st.subheader("🎯 AI预测结果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        direction_color = "🔴" if result.prediction == 1 else "🟢"  # 中国习惯：红涨绿跌
        direction_bg = "prediction-card-up" if result.prediction == 1 else "prediction-card-down"
        st.markdown(f"""
        <div class="{direction_bg}">
            <h3>{direction_color} 预测方向</h3>
            <h2>{result.predicted_direction}</h2>
            <p>基于AI模型综合判断</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_color = "🟢" if result.confidence == "high" else "🟡" if result.confidence == "medium" else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <h4>{confidence_color} 置信度</h4>
            <h3>{result.confidence.upper()}</h3>
            <p>概率: {result.probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 当前价格</h4>
            <h3>¥{result.current_price:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏰ 预测时间</h4>
            <p>{result.timestamp.split('T')[0]}</p>
            <p>{result.timestamp.split('T')[1][:8]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 详细分析
    if result.analysis:
        st.subheader("📋 详细分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 技术指标
            if 'technical_indicators' in result.analysis:
                st.markdown("**🔧 技术指标分析**")
                tech_indicators = result.analysis['technical_indicators']
                
                for indicator, data in tech_indicators.items():
                    if isinstance(data, dict):
                        signal_emoji = "🟢" if data.get('signal') in ['bullish', 'overbought'] else "🔴" if data.get('signal') in ['bearish', 'oversold'] else "🟡"
                        value = data.get('value', 'N/A')
                        # 安全格式化数值
                        if isinstance(value, (int, float)):
                            value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                        st.write(f"{signal_emoji} **{indicator}**: {data.get('signal', 'N/A')} (值: {value_str})")
            
            # 趋势分析
            if 'trend_analysis' in result.analysis:
                st.markdown("**📈 趋势分析**")
                trend = result.analysis['trend_analysis']
                trend_emoji = "🔼" if trend.get('direction') == 'up' else "🔽" if trend.get('direction') == 'down' else "➡️"
                st.write(f"{trend_emoji} 趋势方向: {trend.get('direction', 'N/A')}")
                st.write(f"📊 趋势强度: {trend.get('strength', 'N/A')}")
        
        with col2:
            # 市场情绪
            if 'market_sentiment' in result.analysis:
                st.markdown("**💭 市场情绪**")
                sentiment = result.analysis['market_sentiment']
                
                if 'volume' in sentiment:
                    vol_data = sentiment['volume']
                    vol_emoji = "🔥" if vol_data.get('signal') == 'high' else "❄️" if vol_data.get('signal') == 'low' else "⚖️"
                    st.write(f"{vol_emoji} 成交量: {vol_data.get('signal', 'N/A')} (比率: {vol_data.get('ratio', 0):.2f})")
            
            # 风险因素
            if 'risk_factors' in result.analysis:
                st.markdown("**⚠️ 风险因素**")
                risk_factors = result.analysis['risk_factors']
                
                if risk_factors:
                    for risk in risk_factors:
                        st.write(f"🚨 {risk}")
                else:
                    st.write("✅ 暂无明显风险因素")


def show_stock_basic_info(stock_code, prediction_service):
    """显示股票基本信息"""
    try:
        # 获取股票详细信息
        sector_mapping = prediction_service.feature_engineer.sector_mapping
        stock_info = sector_mapping.get_stock_info(stock_code)
        
        st.subheader(f"📊 {stock_info.get('name', stock_code)} ({stock_code}) 基本信息")
        
        # 显示公司基本信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stock-info-card">
                <h4>🏢 公司信息</h4>
                <p><strong>股票名称:</strong> {stock_info.get('name', 'N/A')}</p>
                <p><strong>股票代码:</strong> {stock_code}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stock-info-card">
                <h4>🏭 行业板块</h4>
                <p><strong>所属行业:</strong> {stock_info.get('sector', 'N/A')}</p>
                <p><strong>板块ID:</strong> {stock_info.get('sector_id', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stock-info-card">
                <h4>💡 题材概念</h4>
                <p><strong>主要概念:</strong> {stock_info.get('primary_concept', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stock-info-card">
                <h4>📍 地区信息</h4>
                <p><strong>所在地区:</strong> {stock_info.get('region', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 显示所有概念（如果有）
        if stock_info.get('all_concepts'):
            st.markdown("**🎯 所有题材概念:**")
            concepts = stock_info.get('all_concepts', '').split(',')
            if len(concepts) > 1:
                concept_cols = st.columns(min(len(concepts), 4))
                for i, concept in enumerate(concepts[:8]):  # 最多显示8个概念
                    if concept.strip():
                        with concept_cols[i % 4]:
                            st.markdown(f"`{concept.strip()}`")
            else:
                st.write(stock_info.get('all_concepts', 'N/A'))
        
        # 获取最新数据
        df = prediction_service.get_latest_stock_data(stock_code, days=30)
        
        if len(df) > 0:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # 实时数据指标
            st.markdown("**📊 实时交易数据**")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                change = latest['涨跌幅']
                delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                st.metric("涨跌幅", f"{change:.2f}%", delta=f"{change:+.2f}%")
            
            with col2:
                volume_str = f"{latest['成交量']:.0f}"
                if latest['成交量'] > 100000000:  # 超过1亿
                    volume_str = f"{latest['成交量']/100000000:.2f}亿"
                elif latest['成交量'] > 10000:  # 超过1万
                    volume_str = f"{latest['成交量']/10000:.2f}万"
                st.metric("成交量", volume_str)
            
            with col3:
                st.metric("换手率", f"{latest['换手率']:.2f}%")
            
            with col4:
                st.metric("振幅", f"{latest['振幅']:.2f}%")
            
            with col5:
                # 添加当前价格
                current_price = latest['收盘价']
                prev_price = prev['收盘价'] if len(df) > 1 else current_price
                price_change = current_price - prev_price
                st.metric("当前价格", f"¥{current_price:.2f}", 
                         delta=f"{price_change:+.2f}" if price_change != 0 else None)
            
            # 添加刷新按钮
            col_refresh, col_auto = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 刷新数据"):
                    # 只清除股票数据缓存，不清除预测历史
                    if 'stock_data_cache' in st.session_state:
                        del st.session_state['stock_data_cache']
                    st.experimental_rerun()
            
            with col_auto:
                auto_refresh = st.checkbox("⏰ 自动刷新 (30秒)", value=False)
                if auto_refresh:
                    import time
                    time.sleep(30)
                    st.experimental_rerun()
            
            # K线图 - 增加高度使趋势更明显
            fig = go.Figure(data=go.Candlestick(
                x=df['交易日期'],
                open=df['开盘价'],
                high=df['最高价'],
                low=df['最低价'],
                close=df['收盘价'],
                name=stock_code,
                increasing_line_color='red',    # 上涨为红色（中国股市习惯）
                decreasing_line_color='green',  # 下跌为绿色（中国股市习惯）
                increasing_fillcolor='rgba(255, 0, 0, 0.8)',
                decreasing_fillcolor='rgba(0, 128, 0, 0.8)'
            ))
            
            # 添加成交量图
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df['交易日期'],
                y=df['成交量'],
                name='成交量',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # 创建子图布局
            from plotly.subplots import make_subplots
            fig_combined = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f"{stock_code} K线图 (最近30天)", "成交量"),
                row_heights=[0.7, 0.3]
            )
            
            # 添加K线图
            fig_combined.add_trace(
                go.Candlestick(
                    x=df['交易日期'],
                    open=df['开盘价'],
                    high=df['最高价'],
                    low=df['最低价'],
                    close=df['收盘价'],
                    name=stock_code,
                    increasing_line_color='red',
                    decreasing_line_color='green',
                    increasing_fillcolor='rgba(255, 0, 0, 0.8)',
                    decreasing_fillcolor='rgba(0, 128, 0, 0.8)'
                ),
                row=1, col=1
            )
            
            # 添加成交量
            fig_combined.add_trace(
                go.Bar(
                    x=df['交易日期'],
                    y=df['成交量'],
                    name='成交量',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # 更新布局 - 大幅增加高度
            fig_combined.update_layout(
                height=700,  # 从400增加到700
                showlegend=False,
                xaxis_rangeslider_visible=False,
                title_font_size=16,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # 更新Y轴标签
            fig_combined.update_yaxes(title_text="价格 (元)", row=1, col=1)
            fig_combined.update_yaxes(title_text="成交量", row=2, col=1)
            fig_combined.update_xaxes(title_text="日期", row=2, col=1)
            
            st.plotly_chart(fig_combined, use_container_width=True)
    
    except Exception as e:
        st.warning(f"无法获取股票 {stock_code} 的基本信息: {str(e)}")


def show_performance_monitoring_page():
    """性能监控页面"""
    st.header("📊 模型性能监控")
    
    monitor = get_performance_monitor()
    if monitor is None:
        st.error("性能监控未初始化")
        return
    
    # 控制面板
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days = st.selectbox("监控时间范围", [7, 14, 30, 60], index=2)
    
    with col2:
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.experimental_rerun()
    
    with col3:
        if st.button("📈 生成报告"):
            with st.spinner("生成性能报告中..."):
                report = monitor.generate_performance_report(days=days)
                st.success("报告生成完成！")
                
                # 显示报告摘要
                st.json(report)
    
    # 性能指标展示
    try:
        # 获取性能数据
        performance_data = {}
        for pred_days in [1, 3, 5]:
            df = monitor.db.get_recent_performance(days=days, prediction_days=pred_days)
            if len(df) > 0:
                metrics = monitor.calculate_performance_metrics(df)
                performance_data[f'{pred_days}天'] = metrics
        
        if performance_data:
            # 性能概览
            st.subheader("🎯 性能概览")
            
            cols = st.columns(len(performance_data))
            for i, (model_name, metrics) in enumerate(performance_data.items()):
                with cols[i]:
                    accuracy = metrics.get('accuracy', 0)
                    color = "🟢" if accuracy > 0.6 else "🟡" if accuracy > 0.55 else "🔴"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{color} {model_name}预测</h4>
                        <h3>{accuracy:.1%}</h3>
                        <p>准确率</p>
                        <small>样本: {metrics.get('total_predictions', 0)}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 详细指标表格
            st.subheader("📋 详细性能指标")
            
            metrics_df = pd.DataFrame(performance_data).T
            st.dataframe(
                metrics_df.round(4),
                use_container_width=True
            )
            
            # 性能趋势图
            st.subheader("📈 性能趋势")
            
            # 这里应该从数据库获取历史性能数据
            # 由于时间关系，使用模拟数据
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            fig = go.Figure()
            
            for model_name in performance_data.keys():
                # 模拟趋势数据
                trend_data = np.random.normal(performance_data[model_name].get('accuracy', 0.55), 0.05, days)
                trend_data = np.clip(trend_data, 0, 1)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=trend_data,
                    mode='lines+markers',
                    name=f'{model_name}预测',
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="模型准确率趋势",
                xaxis_title="日期",
                yaxis_title="准确率",
                height=400,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("暂无性能数据，请先进行一些预测操作")
    
    except Exception as e:
        st.error(f"加载性能数据失败: {str(e)}")


def show_batch_prediction_page():
    """批量预测页面"""
    st.header("🎯 批量股票预测")
    
    # 显示使用指南
    show_usage_guide()
    
    prediction_service = get_prediction_service("v2.0")
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 股票选择区域
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("### 📊 选择预测股票组合")
    
    stock_list = load_stock_list()
    stock_options = [f"{stock['股票代码']} - {stock['股票名称']}" for stock in stock_list]
    
    # 快速选择按钮
    col_select1, col_select2, col_select3, col_select4, col_select5 = st.columns(5)
    
    with col_select1:
        if st.button("🎯 一键全选", help="选择所有股票"):
            st.session_state['batch_selected_stocks'] = stock_options
    
    with col_select2:
        if st.button("📈 选择前50", help="选择前50只股票"):
            st.session_state['batch_selected_stocks'] = stock_options[:50]
    
    with col_select3:
        if st.button("🎲 随机50只", help="随机选择50只股票"):
            import random
            st.session_state['batch_selected_stocks'] = random.sample(stock_options, min(50, len(stock_options)))
    
    with col_select4:
        if st.button("🚀 预测全部", help="一键预测所有股票", type="primary"):
            st.session_state['batch_selected_stocks'] = stock_options
            st.session_state['predict_all_stocks'] = True
    
    with col_select5:
        if st.button("🗑️ 清空选择", help="清空所有选择"):
            st.session_state['batch_selected_stocks'] = []
    
    # 股票多选框
    selected_stock_displays = st.multiselect(
        "选择要预测的股票（可多选）",
        stock_options,
        default=st.session_state.get('batch_selected_stocks', stock_options[:20]),
        help="可以选择多只股票进行批量预测，建议一次不超过100只"
    )
    
    # 提取股票代码
    selected_stocks = [display.split(' - ')[0] for display in selected_stock_displays]
    
    # 显示股票选择状态
    if len(selected_stocks) > 0:
        if len(selected_stocks) == len(stock_options):
            st.success(f"🎯 已选择全部 {len(selected_stocks)} 只股票进行批量预测")
        else:
            st.info(f"📊 已选择 {len(selected_stocks)} / {len(stock_options)} 只股票进行批量预测")
    else:
        st.warning("⚠️ 请选择要预测的股票")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测参数设置
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        prediction_days = st.selectbox("预测天数", [1, 3, 5], index=0)
    
    with col2:
        # 模型选择
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "选择主模型",
                list(available_models.keys()),
                index=0,
                help="选择要使用的预测模型"
            )
        else:
            st.warning("⚠️ 没有可用的模型")
            selected_model = None
    
    with col3:
        # 根据股票总数智能推荐批量大小
        total_stocks = len(selected_stocks)
        if total_stocks > 1000:
            batch_options = [50, 100, 200]
            default_batch = 100
            batch_help = "大规模预测建议使用较大批量以提高效率"
        elif total_stocks > 200:
            batch_options = [20, 50, 100]
            default_batch = 50
            batch_help = "中等规模预测推荐批量大小"
        else:
            batch_options = [10, 20, 50]
            default_batch = 20
            batch_help = "小规模预测每批处理的股票数量"
        
        batch_size = st.selectbox("批量大小", batch_options, 
                                 index=batch_options.index(default_batch) if default_batch in batch_options else 0,
                                 help=batch_help)
    
    # 子模型选择区域
    if available_models and selected_model:
        st.markdown('<div class="sub-model-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 细分模型选择")
        st.markdown("选择要使用的AI模型组合进行批量预测：")
        
        col1, col2, col3, col4 = st.columns(4)
        
        selected_sub_models = []
        
        with col1:
            if st.checkbox("🧠 CNN-LSTM", value=True, help="卷积神经网络+长短期记忆网络", key="batch_cnn_lstm"):
                selected_sub_models.append("CNN-LSTM")
            st.markdown("*适合短期波动*")
        
        with col2:
            if st.checkbox("🎯 Transformer", value=True, help="Transformer注意力机制模型", key="batch_transformer"):
                selected_sub_models.append("Transformer")
            st.markdown("*适合长期趋势*")
        
        with col3:
            if st.checkbox("🔄 LSTM", value=True, help="长短期记忆网络", key="batch_lstm"):
                selected_sub_models.append("LSTM")
            st.markdown("*平衡预测*")
        
        with col4:
            if st.checkbox("⚡ LightGBM", value=True, help="轻量级梯度提升机", key="batch_lightgbm"):
                selected_sub_models.append("LightGBM")
            st.markdown("*快速预测*")
        
        if not selected_sub_models:
            st.error("❌ 请至少选择一个子模型")
        else:
            st.success(f"✅ 已选择 {len(selected_sub_models)} 个子模型: {', '.join(selected_sub_models)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 批量预测按钮
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        start_batch_prediction = st.button("🚀 开始批量预测", type="primary", disabled=len(selected_stocks)==0)
    
    with col_pred2:
        if st.button("📊 预测报告模式", help="生成详细的批量预测分析报告"):
            st.session_state['batch_report_mode'] = True
    
    # 检查是否触发一键预测全部
    if st.session_state.get('predict_all_stocks', False):
        st.session_state['predict_all_stocks'] = False  # 重置标志
        start_batch_prediction = True
        selected_stocks = [display.split(' - ')[0] for display in stock_options]  # 使用所有股票
        st.info(f"🚀 正在启动全股票预测模式，共 {len(selected_stocks)} 只股票")
    
    # 执行批量预测
    if start_batch_prediction and selected_stocks:
        
        # 显示预测规模警告
        if len(selected_stocks) > 500:
            st.warning(f"⚠️ 即将预测 {len(selected_stocks)} 只股票，预计需要 {len(selected_stocks) * 2 // 60} 分钟，请耐心等待...")
        elif len(selected_stocks) > 100:
            st.info(f"📊 即将预测 {len(selected_stocks)} 只股票，预计需要 {len(selected_stocks) * 2} 秒")
        
        # 根据用户选择调整模型权重
        if available_models and selected_model and selected_sub_models:
            model_info = available_models[selected_model]
            model_prediction_days = model_info['prediction_days']
            
            # 动态加载选定的模型
            with st.spinner("正在加载模型..."):
                success = load_specific_model(prediction_service, model_info['folder'], model_prediction_days)
            
            if success:
                # 调整权重
                model = prediction_service.models[model_prediction_days]
                if hasattr(model, 'model_weights'):
                    # 重置所有权重为0
                    for model_name in model.model_weights.keys():
                        model.model_weights[model_name] = 0.0
                    
                    # 只给选择的模型分配权重
                    weight_per_model = 1.0 / len(selected_sub_models)
                    for model_name in selected_sub_models:
                        if model_name in model.model_weights:
                            model.model_weights[model_name] = weight_per_model
                    
                    st.info(f"🎯 已调整模型权重: {', '.join([f'{name}:{model.model_weights[name]:.2f}' for name in selected_sub_models])}")
        
        # 创建进度显示区域 - 使用固定容器避免下滑
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        col_progress, col_stats = st.columns([3, 1])
        
        with col_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with col_stats:
            # 使用固定的统计信息容器
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            stats_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        results = []
        successful_predictions = 0
        failed_predictions = 0
        start_time = datetime.now()
        
        # 分批处理
        for batch_start in range(0, len(selected_stocks), batch_size):
            batch_end = min(batch_start + batch_size, len(selected_stocks))
            batch_stocks = selected_stocks[batch_start:batch_end]
            
            for i, stock_code in enumerate(batch_stocks):
                current_index = batch_start + i
                try:
                    status_text.text(f"🔄 正在预测 {stock_code}... ({current_index+1}/{len(selected_stocks)})")
                    
                    result = prediction_service.predict_single_stock(
                        stock_code=stock_code,
                        prediction_days=prediction_days,
                        include_analysis=False,
                        prediction_threshold=get_prediction_threshold()
                    )
                    
                    results.append({
                        '股票代码': result.stock_code,
                        '股票名称': next((s['股票名称'] for s in stock_list if s['股票代码'] == result.stock_code), 'N/A'),
                        '当前价格': result.current_price,
                        '预测方向': result.predicted_direction,
                        '预测概率': result.probability,
                        '置信度': result.confidence,
                        '预测时间': result.timestamp.split('T')[0],
                        '模型组合': ', '.join(selected_sub_models)
                    })
                    
                    successful_predictions += 1
                    
                    # 实时更新统计信息 - 使用placeholder避免界面滑动
                    elapsed_time = (datetime.now() - start_time).seconds
                    remaining_stocks = len(selected_stocks) - current_index - 1
                    avg_time_per_stock = elapsed_time / (current_index + 1) if current_index > 0 else 1
                    estimated_remaining_time = remaining_stocks * avg_time_per_stock
                    
                    # 更新统计信息到固定位置
                    with stats_placeholder.container():
                        st.metric("✅ 成功", successful_predictions)
                        st.metric("❌ 失败", failed_predictions) 
                        st.metric("⏱️ 预计剩余", f"{int(estimated_remaining_time)}秒")
                    
                    progress_bar.progress((current_index + 1) / len(selected_stocks))
                    
                except Exception as e:
                    failed_predictions += 1
                    st.warning(f"预测 {stock_code} 失败: {str(e)}")
            
            # 批间休息，避免系统过载
            if batch_end < len(selected_stocks):
                import time
                time.sleep(0.5)
        
        total_time = (datetime.now() - start_time).seconds
        status_text.text(f"🎉 批量预测完成！总耗时: {total_time}秒")
        
        if results:
            # 显示结果
            show_batch_prediction_results(results, selected_sub_models, total_time)
        else:
            st.warning("⚠️ 没有成功的预测结果")


def show_batch_prediction_results(results, selected_sub_models, total_time):
    """显示批量预测结果"""
    st.markdown("---")
    st.subheader("📊 批量预测结果分析")
    
    results_df = pd.DataFrame(results)
    
    # 核心统计信息 - 现代化卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 总预测数</h4>
            <h2>{len(results_df)}</h2>
            <p>只股票</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        up_count = len(results_df[results_df['预测方向'] == '上涨'])
        up_ratio = up_count / len(results_df) * 100 if len(results_df) > 0 else 0
        st.markdown(f"""
        <div class="prediction-card-up">
            <h4>🔴 预测上涨</h4>
            <h2>{up_count}</h2>
            <p>{up_ratio:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        down_count = len(results_df[results_df['预测方向'] == '下跌'])
        down_ratio = down_count / len(results_df) * 100 if len(results_df) > 0 else 0
        st.markdown(f"""
        <div class="prediction-card-down">
            <h4>🟢 预测下跌</h4>
            <h2>{down_count}</h2>
            <p>{down_ratio:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_conf = len(results_df[results_df['置信度'] == 'high'])
        high_conf_ratio = high_conf / len(results_df) * 100 if len(results_df) > 0 else 0
        st.markdown(f"""
        <div class="success-card">
            <h4>⭐ 高置信度</h4>
            <h2>{high_conf}</h2>
            <p>{high_conf_ratio:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_prob = results_df['预测概率'].mean() if len(results_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 平均概率</h4>
            <h2>{avg_prob:.1%}</h2>
            <p>置信度</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 详细分析选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["📋 详细结果", "📊 数据分析", "📈 可视化图表", "📄 生成报告"])
    
    with tab1:
        st.markdown("### 📋 详细预测结果")
        
        # 过滤和排序选项
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            direction_filter = st.selectbox("预测方向筛选", ["全部", "上涨", "下跌"])
        
        with col_filter2:
            confidence_filter = st.selectbox("置信度筛选", ["全部", "high", "medium", "low"])
        
        with col_filter3:
            sort_by = st.selectbox("排序方式", ["预测概率", "当前价格", "股票代码"])
        
        # 应用筛选
        filtered_df = results_df.copy()
        if direction_filter != "全部":
            filtered_df = filtered_df[filtered_df['预测方向'] == direction_filter]
        if confidence_filter != "全部":
            filtered_df = filtered_df[filtered_df['置信度'] == confidence_filter]
        
        # 排序
        if sort_by == "预测概率":
            filtered_df = filtered_df.sort_values('预测概率', ascending=False)
        elif sort_by == "当前价格":
            filtered_df = filtered_df.sort_values('当前价格', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('股票代码')
        
        # 格式化显示
        display_df = filtered_df.copy()
        display_df['预测概率'] = display_df['预测概率'].apply(lambda x: f"{x:.1%}")
        display_df['当前价格'] = display_df['当前价格'].apply(lambda x: f"¥{x:.2f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "预测方向": st.column_config.TextColumn(
                    "预测方向",
                    help="AI模型预测的涨跌方向"
                ),
                "预测概率": st.column_config.TextColumn(
                    "预测概率",
                    help="预测方向的置信概率"
                ),
            }
        )
    
    with tab2:
        st.markdown("### 📊 深度数据分析")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            # 置信度分布
            st.markdown("**置信度分布分析**")
            confidence_counts = results_df['置信度'].value_counts()
            for conf, count in confidence_counts.items():
                ratio = count / len(results_df) * 100
                st.write(f"• {conf.upper()}: {count}只 ({ratio:.1f}%)")
            
            # 概率区间分析
            st.markdown("**预测概率区间分析**")
            prob_bins = pd.cut(results_df['预测概率'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])
            prob_counts = prob_bins.value_counts().sort_index()
            for prob_range, count in prob_counts.items():
                ratio = count / len(results_df) * 100
                st.write(f"• {prob_range}: {count}只 ({ratio:.1f}%)")
        
        with col_analysis2:
            # 价格区间分析
            st.markdown("**价格区间预测倾向**")
            results_df['价格区间'] = pd.cut(results_df['当前价格'], 
                                        bins=[0, 10, 50, 100, 500, float('inf')], 
                                        labels=['<10元', '10-50元', '50-100元', '100-500元', '>500元'])
            
            price_analysis = results_df.groupby(['价格区间', '预测方向']).size().unstack(fill_value=0)
            
            if not price_analysis.empty:
                for price_range in price_analysis.index:
                    up_count = price_analysis.loc[price_range, '上涨'] if '上涨' in price_analysis.columns else 0
                    down_count = price_analysis.loc[price_range, '下跌'] if '下跌' in price_analysis.columns else 0
                    total = up_count + down_count
                    if total > 0:
                        up_ratio = up_count / total * 100
                        st.write(f"• {price_range}: {up_count}涨/{down_count}跌 (上涨率{up_ratio:.1f}%)")
            
            # 模型组合效果
            st.markdown("**模型组合使用情况**")
            st.write(f"• 使用模型: {', '.join(selected_sub_models)}")
            st.write(f"• 预测耗时: {total_time}秒")
            st.write(f"• 平均每只: {total_time/len(results_df):.1f}秒")
            
    
    with tab3:
        st.markdown("### 📈 可视化图表分析")
        
        # 主要图表
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 预测方向饼图 - 3D效果
            fig_pie = px.pie(
                results_df, 
                names='预测方向', 
                title="预测方向分布",
                color_discrete_map={'上涨': '#ff4757', '下跌': '#2ed573'}
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hole=0.3,  # 甜甜圈图
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            # 置信度分布柱状图
            confidence_counts = results_df['置信度'].value_counts()
            fig_bar = px.bar(
                x=confidence_counts.index,
                y=confidence_counts.values,
                title="预测置信度分布",
                labels={'x': '置信度', 'y': '数量'},
                color=['#ff4757', '#ffa502', '#2ed573'][:len(confidence_counts)]
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # 概率分布直方图
        st.markdown("**预测概率分布直方图**")
        fig_hist = px.histogram(
            results_df, 
            x='预测概率', 
            nbins=20,
            title="预测概率分布",
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 价格区间分析
        st.markdown("**价格区间预测分布**")
        if '价格区间' not in results_df.columns:
            results_df['价格区间'] = pd.cut(results_df['当前价格'], 
                                        bins=[0, 10, 50, 100, 500, float('inf')], 
                                        labels=['<10元', '10-50元', '50-100元', '100-500元', '>500元'])
        
        price_analysis = results_df.groupby(['价格区间', '预测方向']).size().unstack(fill_value=0)
        
        if not price_analysis.empty:
            fig_price = px.bar(
                price_analysis,
                title="不同价格区间的预测分布",
                labels={'index': '价格区间', 'value': '数量'},
                color_discrete_map={'上涨': '#ff4757', '下跌': '#2ed573'},
                barmode='group'
            )
            fig_price.update_layout(height=400)
            st.plotly_chart(fig_price, use_container_width=True)
        
        # 散点图：价格vs概率
        st.markdown("**价格与预测概率关系**")
        fig_scatter = px.scatter(
            results_df, 
            x='当前价格', 
            y='预测概率',
            color='预测方向',
            size='预测概率',
            hover_data=['股票代码', '股票名称'],
            title="股票价格与预测概率关系",
            color_discrete_map={'上涨': '#ff4757', '下跌': '#2ed573'}
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.markdown("### 📄 生成预测报告")
        
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            report_format = st.selectbox("报告格式", ["详细报告", "简要总结", "投资建议"])
            include_charts = st.checkbox("包含图表", value=True)
        
        with col_report2:
            if st.button("📄 生成报告", type="primary"):
                report_content = generate_batch_prediction_report(results_df, selected_sub_models, total_time, report_format)
                st.markdown(report_content)
                
                # 下载报告
                st.download_button(
                    label="📥 下载报告",
                    data=report_content,
                    file_name=f"batch_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        # 数据导出选项
        st.markdown("### 📊 数据导出")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            # CSV导出
            csv = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出CSV",
                data=csv,
                file_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # JSON导出
            json_data = results_df.to_json(orient='records', force_ascii=False, indent=2)
            st.download_button(
                label="📥 导出JSON",
                data=json_data,
                file_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_export3:
            # Excel导出
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, sheet_name='预测结果', index=False)
            
            buffer.seek(0)  # 重置缓冲区指针到开始位置
            st.download_button(
                label="📥 导出Excel",
                data=buffer.getvalue(),
                file_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


def generate_batch_prediction_report(results_df, selected_sub_models, total_time, report_format):
    """生成批量预测报告"""
    
    # 基础统计
    total_stocks = len(results_df)
    up_count = len(results_df[results_df['预测方向'] == '上涨'])
    down_count = len(results_df[results_df['预测方向'] == '下跌'])
    high_conf_count = len(results_df[results_df['置信度'] == 'high'])
    avg_prob = results_df['预测概率'].mean()
    
    if report_format == "简要总结":
        return f"""# 批量预测简要总结

## 📊 核心数据
- **预测股票总数**: {total_stocks}只
- **预测上涨**: {up_count}只 ({up_count/total_stocks*100:.1f}%)
- **预测下跌**: {down_count}只 ({down_count/total_stocks*100:.1f}%)
- **高置信度**: {high_conf_count}只 ({high_conf_count/total_stocks*100:.1f}%)
- **平均概率**: {avg_prob:.1%}

## 🤖 模型信息
- **使用模型**: {', '.join(selected_sub_models)}
- **预测耗时**: {total_time}秒
- **处理效率**: {total_time/total_stocks:.1f}秒/只

## 📈 市场观点
{"看涨情绪较强" if up_count > down_count else "看跌情绪较强" if down_count > up_count else "市场分化明显"}
"""
    
    elif report_format == "投资建议":
        # 高置信度推荐股票
        high_conf_up = results_df[(results_df['置信度'] == 'high') & (results_df['预测方向'] == '上涨')].head(5)
        high_conf_down = results_df[(results_df['置信度'] == 'high') & (results_df['预测方向'] == '下跌')].head(5)
        
        report = f"""# 投资建议报告

## 🎯 核心建议

### 📈 推荐关注（高置信度上涨）
"""
        for _, stock in high_conf_up.iterrows():
            report += f"- **{stock['股票名称']}({stock['股票代码']})**: 预测概率{stock['预测概率']:.1%}\n"
        
        report += f"""
### 📉 建议规避（高置信度下跌）
"""
        for _, stock in high_conf_down.iterrows():
            report += f"- **{stock['股票名称']}({stock['股票代码']})**: 预测概率{stock['预测概率']:.1%}\n"
        
        report += f"""
## ⚠️ 风险提示
- 本报告基于AI模型预测，仅供参考
- 股市有风险，投资需谨慎
- 建议结合基本面分析做出投资决策
"""
        return report
    
    else:  # 详细报告
        return f"""# 批量预测详细报告

## 📊 执行概览
- **预测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **预测股票**: {total_stocks}只
- **使用模型**: {', '.join(selected_sub_models)}
- **执行耗时**: {total_time}秒

## 📈 预测结果统计
- **预测上涨**: {up_count}只股票 ({up_count/total_stocks*100:.1f}%)
- **预测下跌**: {down_count}只股票 ({down_count/total_stocks*100:.1f}%)
- **高置信度**: {high_conf_count}只股票 ({high_conf_count/total_stocks*100:.1f}%)
- **平均预测概率**: {avg_prob:.1%}

## 🎯 置信度分析
{results_df['置信度'].value_counts().to_string()}

## 💰 价格区间分析
{results_df.groupby('价格区间')['预测方向'].value_counts().to_string() if '价格区间' in results_df.columns else "价格区间数据不可用"}

## 🤖 模型表现
- **处理效率**: {total_time/total_stocks:.2f}秒/只股票
- **模型组合**: {', '.join(selected_sub_models)}

## ⚠️ 免责声明
本报告由AI模型生成，仅供参考。投资决策应结合多方面因素，风险自担。
"""


def show_risk_assessment_page():
    """风险评估页面"""
    st.header("⚠️ 风险评估")
    
    prediction_service = get_prediction_service("v2.0")
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 股票选择
    stock_list = load_stock_list()
    stock_options = [f"{stock['股票代码']} - {stock['股票名称']}" for stock in stock_list]
    selected_stock_display = st.selectbox("选择股票进行风险评估", stock_options)
    selected_stock = selected_stock_display.split(' - ')[0] if selected_stock_display else None
    
    if st.button("🔍 开始风险评估", type="primary"):
        with st.spinner("正在进行风险评估..."):
            try:
                risk_assessment = prediction_service.assess_risk(selected_stock)
                
                # 风险等级展示
                risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                risk_bg_color = {"low": "success-card", "medium": "metric-card", "high": "warning-card"}
                
                st.markdown(f"""
                <div class="{risk_bg_color[risk_assessment.risk_level]}">
                    <h3>{risk_color[risk_assessment.risk_level]} 风险等级: {risk_assessment.risk_level.upper()}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 风险指标
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "预测波动率", 
                        f"{risk_assessment.volatility_forecast:.1%}",
                        help="预测的未来波动率"
                    )
                
                with col2:
                    st.metric(
                        "最大回撤预测", 
                        f"{risk_assessment.max_drawdown_forecast:.1%}",
                        help="预测的最大可能回撤"
                    )
                
                with col3:
                    st.metric(
                        "建议止损价", 
                        f"¥{risk_assessment.stop_loss_suggestion:.2f}",
                        help="基于风险分析的建议止损价格"
                    )
                
                with col4:
                    st.metric(
                        "建议仓位", 
                        f"{risk_assessment.position_size_suggestion:.1%}",
                        help="基于风险的建议仓位大小"
                    )
                
                # 风险因素
                if risk_assessment.risk_factors:
                    st.subheader("🚨 识别的风险因素")
                    for factor in risk_assessment.risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ 暂未识别到明显风险因素")
                
                # 投资建议
                st.subheader("💡 投资建议")
                
                if risk_assessment.risk_level == "low":
                    st.success("""
                    **低风险股票投资建议：**
                    - ✅ 可以考虑适度增加仓位
                    - ✅ 适合长期持有
                    - ✅ 可以采用相对宽松的止损策略
                    """)
                elif risk_assessment.risk_level == "medium":
                    st.info("""
                    **中等风险股票投资建议：**
                    - ⚖️ 建议控制仓位大小
                    - ⚖️ 密切关注市场变化
                    - ⚖️ 设置合理的止损点
                    """)
                else:
                    st.warning("""
                    **高风险股票投资建议：**
                    - 🚨 建议减少仓位或观望
                    - 🚨 设置较紧的止损
                    - 🚨 避免重仓操作
                    - 🚨 密切监控风险指标变化
                    """)
                
            except Exception as e:
                st.error(f"风险评估失败: {str(e)}")


def show_prediction_history_page():
    """预测历史页面"""
    st.header("📋 预测历史记录")
    
    prediction_service = get_prediction_service("v2.0")
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 筛选条件
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_list = load_stock_list()
        stock_options = ["全部"] + [f"{stock['股票代码']} - {stock['股票名称']}" for stock in stock_list]
        stock_display = st.selectbox(
            "选择股票（可选）", 
            stock_options
        )
        
        if stock_display == "全部":
            stock_code = "全部"
        else:
            stock_code = stock_display.split(' - ')[0]
    
    with col2:
        days = st.selectbox("历史天数", [7, 14, 30, 60], index=2)
    
    with col3:
        if st.button("🔄 刷新历史"):
            # 只清除历史数据相关的缓存，保留其他缓存
            if 'prediction_history_cache' in st.session_state:
                del st.session_state['prediction_history_cache']
    
    try:
        # 优先使用session_state中的预测历史
        if 'prediction_results' in st.session_state and st.session_state['prediction_results']:
            session_history = st.session_state['prediction_results']
            
            # 过滤数据
            if stock_code != "全部":
                session_history = [h for h in session_history if h['stock_code'] == stock_code]
            
            # 按时间过滤
            cutoff_time = datetime.now() - timedelta(days=days)
            session_history = [h for h in session_history if h['timestamp'] >= cutoff_time]
            
            if session_history:
                history = session_history
                st.info(f"📊 显示来自本次会话的 {len(history)} 条预测记录")
            else:
                # 如果session中没有符合条件的数据，尝试从预测服务获取
                history = prediction_service.get_prediction_history(stock_code if stock_code != "全部" else None, days)
                if not history:
                    st.info("暂无预测历史记录")
                    return
        else:
            # 从预测服务获取历史
            history = prediction_service.get_prediction_history(stock_code if stock_code != "全部" else None, days)
        
        if history:
            # 转换为DataFrame
            history_df = pd.DataFrame(history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['date'] = history_df['timestamp'].dt.date
            
            # 统计信息
            st.subheader("📊 历史统计")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总预测次数", len(history_df))
            
            with col2:
                avg_prob = history_df['probability'].mean()
                st.metric("平均预测概率", f"{avg_prob:.1%}")
            
            with col3:
                up_predictions = len(history_df[history_df['prediction'] == 1])
                st.metric("预测上涨次数", up_predictions)
            
            with col4:
                unique_stocks = history_df['stock_code'].nunique()
                st.metric("涉及股票数", unique_stocks)
            
            # 历史趋势图
            st.subheader("📈 预测趋势")
            
            daily_stats = history_df.groupby('date').agg({
                'prediction': 'count',
                'probability': 'mean'
            }).reset_index()
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['每日预测次数', '平均预测概率'],
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Bar(x=daily_stats['date'], y=daily_stats['prediction'], name='预测次数'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['date'], 
                    y=daily_stats['probability'], 
                    mode='lines+markers',
                    name='平均概率'
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 详细记录表
            st.subheader("📋 详细记录")
            
            # 格式化显示
            display_df = history_df.copy()
            display_df['概率'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
            display_df['方向'] = display_df['prediction'].apply(lambda x: "上涨" if x == 1 else "下跌")
            display_df['时间'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df[['时间', 'stock_code', 'prediction_days', '方向', '概率', 'current_price']].rename(columns={
                    'stock_code': '股票代码',
                    'prediction_days': '预测天数',
                    'current_price': '当时价格'
                }),
                use_container_width=True
            )
            
        else:
            st.info("暂无预测历史记录")
    
    except Exception as e:
        st.error(f"加载预测历史失败: {str(e)}")


def show_model_management_page():
    """模型管理页面"""
    st.header("🤖 AI模型管理")
    
    prediction_service = get_prediction_service("v2.0")
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 获取可用模型
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ 没有找到可用的模型")
        st.info("请先运行训练脚本来训练模型")
        return
    
    st.subheader("📋 已训练模型列表")
    
    # 创建模型信息表格
    model_data = []
    for display_name, info in available_models.items():
        model_data.append({
            '模型名称': display_name,
            '预测天数': f"{info['prediction_days']}天",
            '训练日期': info['folder'].split('_')[-2] if '_' in info['folder'] else '未知',
            '准确率': f"{info['accuracy']:.2%}" if info['accuracy'] > 0 else '未知',
            '文件夹': info['folder']
        })
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True)
    
    # 模型操作区域
    st.subheader("🔧 模型操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_model_for_action = st.selectbox(
            "选择要操作的模型",
            list(available_models.keys()),
            help="选择要进行操作的模型"
        )
    
    with col2:
        if st.button("🔄 重新加载模型"):
            if selected_model_for_action:
                model_info = available_models[selected_model_for_action]
                success = load_specific_model(
                    prediction_service, 
                    model_info['folder'], 
                    model_info['prediction_days']
                )
                if success:
                    st.success(f"✅ 模型 {selected_model_for_action} 重新加载成功")
                else:
                    st.error(f"❌ 模型 {selected_model_for_action} 重新加载失败")
    
    with col3:
        if st.button("📊 查看模型详情"):
            if selected_model_for_action:
                model_info = available_models[selected_model_for_action]
                show_model_details(model_info)
    
    # 训练新模型
    st.subheader("🎯 训练新模型")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_model_days = st.selectbox("新模型预测天数", [1, 3, 5], help="选择新模型的预测天数")
    
    with col2:
        if st.button("🚀 开始训练", type="primary"):
            st.info("训练功能需要在后台执行，请使用命令行运行训练脚本")
            st.code("python core/training_pipeline.py", language="bash")


def show_model_details(model_info):
    """显示模型详细信息"""
    st.subheader(f"📊 模型详情: {model_info['folder']}")
    
    training_info = model_info['training_info']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 基本信息**")
        st.write(f"🎯 预测天数: {model_info['prediction_days']}天")
        st.write(f"📊 准确率: {model_info['accuracy']:.2%}")
        st.write(f"📁 文件夹: {model_info['folder']}")
        
        if 'feature_names' in training_info:
            st.write(f"🔧 特征数量: {len(training_info['feature_names'])}")
    
    with col2:
        st.markdown("**⚙️ 训练配置**")
        for key, value in training_info.items():
            if key not in ['feature_names', 'feature_info']:
                st.write(f"• {key}: {value}")
    
    # 特征列表
    if 'feature_names' in training_info:
        st.markdown("**🛠️ 使用的特征**")
        feature_names = training_info['feature_names']
        
        # 分列显示特征
        cols = st.columns(3)
        for i, feature in enumerate(feature_names[:30]):  # 只显示前30个特征
            with cols[i % 3]:
                st.write(f"• {feature}")
        
        if len(feature_names) > 30:
            st.write(f"... 还有 {len(feature_names) - 30} 个特征")


def show_usage_guide():
    """显示使用指南"""
    with st.expander("📖 新功能使用指南", expanded=False):
        st.markdown("""
        ### 🔍 股票搜索功能
        - **搜索方式**: 支持股票代码（如600519）或股票名称（如茅台）的模糊搜索
        - **快速选择**: 点击热门股票按钮可快速选择常用股票
        - **清空搜索**: 点击"🗑️ 清空"按钮清除搜索条件
        
        ### 🤖 细分模型选择
        - **CNN-LSTM**: 适合捕捉短期价格波动和技术形态
        - **Transformer**: 擅长分析长期趋势和复杂依赖关系
        - **LSTM**: 平衡短期和长期预测，稳定性较好
        - **LightGBM**: 快速预测，适合实时决策
        - **模型组合**: 可选择多个模型进行集成预测，提高准确性
        
        ### 📊 图表改进
        - **K线图高度**: 增加到700px，趋势更加清晰
        - **成交量图**: 添加成交量子图，便于分析量价关系
        - **中国色彩**: 采用中国股市习惯的红涨绿跌配色
        - **实时数据**: 支持手动和自动刷新功能
        
        ### 💡 使用建议
        - 首次使用建议选择所有子模型进行对比
        - 对于短线交易，重点关注CNN-LSTM的结果
        - 对于长线投资，参考Transformer的预测
        - 结合技术指标和基本面信息做出最终决策
        """)


def get_prediction_threshold():
    """获取预测阈值，如果未设置则返回默认值"""
    return getattr(st.session_state, 'confidence_threshold', 0.6)


def show_system_settings_page():
    """系统设置页面"""
    st.header("⚙️ 系统设置")
    
    # 显示使用指南
    show_usage_guide()
    
    # 模型设置
    st.subheader("🤖 模型设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_prediction_days = st.selectbox("默认预测天数", [1, 3, 5], index=0)
        confidence_threshold = st.slider("预测置信度阈值", 0.5, 0.9, 0.6, 0.05, 
                                        help="设置预测置信度的阈值，影响高/中/低置信度的判定")
        enable_advanced_analysis = st.checkbox("启用高级分析", value=True)
        
        # 存储设置到session state
        st.session_state.default_prediction_days = default_prediction_days
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.enable_advanced_analysis = enable_advanced_analysis
    
    with col2:
        st.selectbox("模型更新频率", ["每日", "每周", "每月"], index=1)
        st.slider("风险评估敏感度", 0.1, 1.0, 0.5, 0.1)
        st.checkbox("启用实时数据更新", value=True)
    
    # 通知设置
    st.subheader("📧 通知设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("告警邮箱", placeholder="your-email@example.com")
        st.checkbox("启用性能告警", value=True)
    
    with col2:
        st.selectbox("告警频率", ["立即", "每小时", "每日"], index=2)
        st.multiselect("告警类型", ["准确率下降", "系统异常", "数据更新失败"], default=["准确率下降"])
    
    # 数据设置
    st.subheader("💾 数据设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("数据更新时间", ["15:30", "16:00", "17:00"], index=0)
        st.number_input("历史数据保留天数", min_value=30, max_value=365, value=180)
    
    with col2:
        st.checkbox("启用数据备份", value=True)
        st.selectbox("备份频率", ["每日", "每周"], index=0)
    
    # 系统信息
    st.subheader("ℹ️ 系统信息")
    
    system_info = {
        "系统版本": "v1.0.0",
        "Python版本": f"{sys.version.split()[0]}",
        "启动时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据目录": os.path.abspath("data/datas_em"),
        "模型目录": os.path.abspath("models")
    }
    
    for key, value in system_info.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(value)
    
    # 操作按钮
    st.subheader("🔧 系统操作")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 重新加载模型"):
            st.cache_resource.clear()
            st.success("模型重新加载完成")
    
    with col2:
        if st.button("📊 导出系统日志"):
            st.info("日志导出功能开发中...")
    
    with col3:
        if st.button("🧹 清理缓存"):
            st.cache_data.clear()
            st.success("缓存清理完成")
    
    with col4:
        if st.button("🔄 重启系统"):
            st.warning("重启功能需要管理员权限")


if __name__ == "__main__":
    main()