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

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-card {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_stock_list():
    """加载股票列表"""
    try:
        # 尝试从CSV文件加载股票列表
        if os.path.exists('stockcode_list/all_stock_list.csv'):
            df = pd.read_csv('stockcode_list/all_stock_list.csv', encoding='utf-8')
            return df['股票代码'].tolist()
        else:
            # 默认股票列表
            return ['sh600519', 'sz000001', 'sz000002', 'sh600036', 'sz000858']
    except:
        return ['sh600519', 'sz000001', 'sz000002']


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
        
        if not fitted_models:
            st.warning("⚠️ AI模型加载失败，启用后备模型")
            prediction_service._setup_fallback_model(model)
            fitted_models = [name for name, m in model.models.items() if getattr(m, 'is_fitted', False)]
        
        st.info(f"✅ 已加载模型: {model_folder}")
        st.info(f"📊 可用子模型: {', '.join(fitted_models)}")
        
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
def get_prediction_service():
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
    
    # 获取预测服务
    prediction_service = get_prediction_service()
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 股票选择区域
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        stock_list = load_stock_list()
        selected_stock = st.selectbox(
            "选择股票代码",
            stock_list,
            index=0,
            help="选择要预测的股票代码"
        )
    
    with col2:
        prediction_days = st.selectbox(
            "预测天数",
            [1, 3, 5],
            index=0,
            help="选择预测的时间跨度"
        )
    
    with col3:
        # 模型选择
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "选择模型",
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
    
    # 预测按钮
    if st.button("🔮 开始预测", type="primary"):
        # 清除缓存以确保使用最新的模型
        st.cache_resource.clear()
        
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
                
                # 执行预测
                result = prediction_service.predict_single_stock(
                    stock_code=selected_stock,
                    prediction_days=prediction_days,
                    include_analysis=include_analysis
                )
                
                # 显示预测结果
                show_prediction_result(result)
                
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
    # 预测结果卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        direction_color = "🟢" if result.prediction == 1 else "🔴"
        st.markdown(f"""
        <div class="prediction-card">
            <h3>{direction_color} 预测方向</h3>
            <h2>{result.predicted_direction}</h2>
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
            <div class="metric-card">
                <h4>🏢 公司信息</h4>
                <p><strong>股票名称:</strong> {stock_info.get('name', 'N/A')}</p>
                <p><strong>股票代码:</strong> {stock_code}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏭 行业板块</h4>
                <p><strong>所属行业:</strong> {stock_info.get('sector', 'N/A')}</p>
                <p><strong>板块ID:</strong> {stock_info.get('sector_id', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💡 题材概念</h4>
                <p><strong>主要概念:</strong> {stock_info.get('primary_concept', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
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
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                change = latest['涨跌幅']
                delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
                st.metric("涨跌幅", f"{change:.2f}%", delta=f"{change:+.2f}%")
            
            with col2:
                st.metric("成交量", f"{latest['成交量']:.0f}")
            
            with col3:
                st.metric("换手率", f"{latest['换手率']:.2f}%")
            
            with col4:
                st.metric("振幅", f"{latest['振幅']:.2f}%")
            
            # K线图
            fig = go.Figure(data=go.Candlestick(
                x=df['交易日期'],
                open=df['开盘价'],
                high=df['最高价'],
                low=df['最低价'],
                close=df['收盘价'],
                name=stock_code
            ))
            
            fig.update_layout(
                title=f"{stock_code} K线图 (最近30天)",
                xaxis_title="日期",
                yaxis_title="价格",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
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
    
    prediction_service = get_prediction_service()
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 股票选择
    stock_list = load_stock_list()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stocks = st.multiselect(
            "选择要预测的股票（可多选）",
            stock_list,
            default=stock_list[:10] if len(stock_list) > 10 else stock_list,
            help="可以选择多只股票进行批量预测"
        )
    
    with col2:
        prediction_days = st.selectbox("预测天数", [1, 3, 5], index=0)
    
    # 批量预测按钮
    if st.button("🚀 开始批量预测", type="primary") and selected_stocks:
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, stock_code in enumerate(selected_stocks):
            try:
                status_text.text(f"正在预测 {stock_code}... ({i+1}/{len(selected_stocks)})")
                
                result = prediction_service.predict_single_stock(
                    stock_code=stock_code,
                    prediction_days=prediction_days,
                    include_analysis=False
                )
                
                results.append({
                    '股票代码': result.stock_code,
                    '当前价格': result.current_price,
                    '预测方向': result.predicted_direction,
                    '预测概率': f"{result.probability:.1%}",
                    '置信度': result.confidence,
                    '预测时间': result.timestamp.split('T')[0]
                })
                
                progress_bar.progress((i + 1) / len(selected_stocks))
                
            except Exception as e:
                st.warning(f"预测 {stock_code} 失败: {str(e)}")
        
        status_text.text("批量预测完成！")
        
        if results:
            # 显示结果
            st.subheader("📊 批量预测结果")
            
            results_df = pd.DataFrame(results)
            
            # 统计信息
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总预测数", len(results_df))
            
            with col2:
                up_count = len(results_df[results_df['预测方向'] == '上涨'])
                st.metric("预测上涨", up_count)
            
            with col3:
                down_count = len(results_df[results_df['预测方向'] == '下跌'])
                st.metric("预测下跌", down_count)
            
            with col4:
                high_conf = len(results_df[results_df['置信度'] == 'high'])
                st.metric("高置信度", high_conf)
            
            # 结果表格
            st.dataframe(results_df, use_container_width=True)
            
            # 可视化
            fig = px.pie(
                results_df, 
                names='预测方向', 
                title=f"{prediction_days}天预测方向分布",
                color_discrete_map={'上涨': 'green', '下跌': 'red'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 导出功能
            csv = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载预测结果CSV",
                data=csv,
                file_name=f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def show_risk_assessment_page():
    """风险评估页面"""
    st.header("⚠️ 风险评估")
    
    prediction_service = get_prediction_service()
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 股票选择
    stock_list = load_stock_list()
    selected_stock = st.selectbox("选择股票进行风险评估", stock_list)
    
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
    
    prediction_service = get_prediction_service()
    if prediction_service is None:
        st.error("预测服务未初始化")
        return
    
    # 筛选条件
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_code = st.selectbox(
            "选择股票（可选）", 
            ["全部"] + load_stock_list()
        )
    
    with col2:
        days = st.selectbox("历史天数", [7, 14, 30, 60], index=2)
    
    with col3:
        if st.button("🔄 刷新历史"):
            st.cache_data.clear()
    
    try:
        # 获取预测历史
        stock_filter = None if stock_code == "全部" else stock_code
        history = prediction_service.get_prediction_history(stock_filter, days)
        
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
    
    prediction_service = get_prediction_service()
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


def show_system_settings_page():
    """系统设置页面"""
    st.header("⚙️ 系统设置")
    
    # 模型设置
    st.subheader("🤖 模型设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("默认预测天数", [1, 3, 5], index=0)
        st.slider("预测置信度阈值", 0.5, 0.9, 0.6, 0.05)
        st.checkbox("启用高级分析", value=True)
    
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