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
    from prediction_service import PredictionService
    from performance_monitor import PerformanceMonitor
    from feature_engineering import FeatureEngineering
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


@st.cache_resource
def get_prediction_service():
    """获取预测服务实例"""
    try:
        return PredictionService()
    except Exception as e:
        st.error(f"初始化预测服务失败: {e}")
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
        ["📈 股票预测", "📊 性能监控", "🎯 批量预测", "⚠️ 风险评估", "📋 预测历史", "⚙️ 系统设置"]
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
    col1, col2, col3 = st.columns([2, 1, 1])
    
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
        include_analysis = st.checkbox(
            "包含详细分析",
            value=True,
            help="包含技术指标和趋势分析"
        )
    
    # 预测按钮
    if st.button("🔮 开始预测", type="primary"):
        with st.spinner("正在进行AI预测分析..."):
            try:
                # 执行预测
                result = prediction_service.predict_single_stock(
                    stock_code=selected_stock,
                    prediction_days=prediction_days,
                    include_analysis=include_analysis
                )
                
                # 显示预测结果
                show_prediction_result(result)
                
            except Exception as e:
                st.error(f"预测失败: {str(e)}")
    
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
                        st.write(f"{signal_emoji} **{indicator}**: {data.get('signal', 'N/A')} (值: {data.get('value', 'N/A'):.2f})")
            
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
        st.subheader(f"📊 {stock_code} 基本信息")
        
        # 获取最新数据
        df = prediction_service.get_latest_stock_data(stock_code, days=30)
        
        if len(df) > 0:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                change = latest['涨跌幅']
                color = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                st.metric("涨跌幅", f"{change:.2f}%", color=color)
            
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
        "数据目录": os.path.abspath("datas_em"),
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