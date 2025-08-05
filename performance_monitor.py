# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 性能监控模块
功能：实时监控模型性能、自动优化、报警系统
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sqlite3
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceDatabase:
    """
    性能数据库管理类
    存储和管理模型性能历史数据
    """
    
    def __init__(self, db_path: str = "performance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 预测记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                prediction_days INTEGER NOT NULL,
                predicted_direction INTEGER NOT NULL,
                predicted_probability REAL NOT NULL,
                actual_direction INTEGER,
                actual_return REAL,
                correct INTEGER,
                model_version TEXT,
                confidence_level TEXT
            )
        ''')
        
        # 模型性能统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                prediction_days INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                auc_score REAL,
                total_predictions INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                avg_confidence REAL,
                model_version TEXT
            )
        ''')
        
        # 风险指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                prediction_days INTEGER NOT NULL,
                max_drawdown REAL,
                volatility REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                avg_return REAL,
                model_version TEXT
            )
        ''')
        
        # 系统健康指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                api_response_time REAL,
                active_models INTEGER,
                cache_hit_rate REAL,
                error_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("性能数据库初始化完成")
    
    def save_prediction(self, stock_code: str, prediction_days: int,
                       predicted_direction: int, predicted_probability: float,
                       model_version: str = "v1.0", confidence_level: str = "medium"):
        """保存预测记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, stock_code, prediction_days, predicted_direction, 
             predicted_probability, model_version, confidence_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            stock_code,
            prediction_days,
            predicted_direction,
            predicted_probability,
            model_version,
            confidence_level
        ))
        
        conn.commit()
        conn.close()
    
    def update_prediction_result(self, stock_code: str, prediction_days: int,
                               actual_direction: int, actual_return: float,
                               timestamp_from: datetime):
        """更新预测结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_direction = ?, actual_return = ?, 
                correct = (predicted_direction = ?)
            WHERE stock_code = ? AND prediction_days = ? 
            AND timestamp >= ? AND actual_direction IS NULL
        ''', (
            actual_direction,
            actual_return,
            actual_direction,
            stock_code,
            prediction_days,
            timestamp_from.isoformat()
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"更新预测结果: {stock_code}, {prediction_days}天")
    
    def get_recent_performance(self, days: int = 30, 
                             prediction_days: int = None) -> pd.DataFrame:
        """获取最近的性能数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM predictions 
            WHERE timestamp >= ? AND actual_direction IS NOT NULL
        '''
        params = [
            (datetime.now() - timedelta(days=days)).isoformat()
        ]
        
        if prediction_days is not None:
            query += " AND prediction_days = ?"
            params.append(prediction_days)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def save_performance_metrics(self, date: str, prediction_days: int,
                               metrics: Dict, model_version: str = "v1.0"):
        """保存性能指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_performance 
            (date, prediction_days, accuracy, precision_score, recall, f1_score,
             auc_score, total_predictions, correct_predictions, avg_confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            prediction_days,
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc', 0),
            metrics.get('total_predictions', 0),
            metrics.get('correct_predictions', 0),
            metrics.get('avg_confidence', 0),
            model_version
        ))
        
        conn.commit()
        conn.close()


class PerformanceMonitor:
    """
    性能监控主类
    监控模型预测准确性、系统健康状况等
    """
    
    def __init__(self, db_path: str = "performance.db"):
        self.db = PerformanceDatabase(db_path)
        
        # 监控配置
        self.config = {
            'min_accuracy_threshold': 0.55,
            'min_precision_threshold': 0.50,
            'max_drawdown_threshold': 0.20,
            'min_sample_size': 50,
            'monitoring_window_days': 30,
            'alert_email': None,  # 设置告警邮箱
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_user': None,
            'email_password': None,
        }
        
        # 性能指标历史
        self.performance_history = []
        
        # 告警状态
        self.alert_status = {
            'accuracy_alert': False,
            'system_alert': False,
            'data_quality_alert': False
        }
    
    def calculate_performance_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """计算性能指标"""
        if len(predictions_df) == 0:
            return {}
        
        # 基础指标
        y_true = predictions_df['actual_direction'].values
        y_pred = predictions_df['predicted_direction'].values
        y_proba = predictions_df['predicted_probability'].values
        
        metrics = {
            'total_predictions': len(predictions_df),
            'correct_predictions': int(predictions_df['correct'].sum()),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'avg_confidence': float(y_proba.mean()),
        }
        
        # AUC计算（如果有足够的正负样本）
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['auc'] = 0.5
        
        # 金融特定指标
        if 'actual_return' in predictions_df.columns:
            returns = predictions_df['actual_return'].dropna()
            if len(returns) > 0:
                metrics.update(self._calculate_financial_metrics(predictions_df))
        
        return metrics
    
    def _calculate_financial_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """计算金融特定指标"""
        returns = predictions_df['actual_return'].dropna()
        correct_predictions = predictions_df[predictions_df['correct'] == 1]['actual_return'].dropna()
        
        financial_metrics = {
            'avg_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'win_rate': float(len(correct_predictions) / len(returns)) if len(returns) > 0 else 0,
            'avg_winning_return': float(correct_predictions.mean()) if len(correct_predictions) > 0 else 0,
        }
        
        # 夏普比率
        if financial_metrics['volatility'] > 0:
            financial_metrics['sharpe_ratio'] = financial_metrics['avg_return'] / financial_metrics['volatility']
        else:
            financial_metrics['sharpe_ratio'] = 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        financial_metrics['max_drawdown'] = float(abs(drawdown.min()))
        
        return financial_metrics
    
    def daily_performance_check(self):
        """每日性能检查"""
        logger.info("开始每日性能检查...")
        
        today = datetime.now().date().isoformat()
        
        for prediction_days in [1, 3, 5]:
            try:
                # 获取最近的预测数据
                df = self.db.get_recent_performance(
                    days=self.config['monitoring_window_days'],
                    prediction_days=prediction_days
                )
                
                if len(df) < self.config['min_sample_size']:
                    logger.warning(f"样本数量不足 ({len(df)} < {self.config['min_sample_size']})，跳过 {prediction_days} 天预测检查")
                    continue
                
                # 计算性能指标
                metrics = self.calculate_performance_metrics(df)
                
                # 保存到数据库
                self.db.save_performance_metrics(today, prediction_days, metrics)
                
                # 性能告警检查
                self._check_performance_alerts(prediction_days, metrics)
                
                logger.info(f"{prediction_days}天预测性能 - 准确率: {metrics['accuracy']:.3f}, "
                          f"精确率: {metrics['precision']:.3f}, F1: {metrics['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"检查 {prediction_days} 天预测性能失败: {str(e)}")
    
    def _check_performance_alerts(self, prediction_days: int, metrics: Dict):
        """检查性能告警"""
        alerts = []
        
        # 准确率告警
        if metrics['accuracy'] < self.config['min_accuracy_threshold']:
            alerts.append(f"{prediction_days}天预测准确率过低: {metrics['accuracy']:.3f}")
            self.alert_status['accuracy_alert'] = True
        
        # 精确率告警
        if metrics['precision'] < self.config['min_precision_threshold']:
            alerts.append(f"{prediction_days}天预测精确率过低: {metrics['precision']:.3f}")
        
        # 最大回撤告警
        if 'max_drawdown' in metrics and metrics['max_drawdown'] > self.config['max_drawdown_threshold']:
            alerts.append(f"{prediction_days}天预测最大回撤过高: {metrics['max_drawdown']:.3f}")
        
        # 发送告警
        if alerts:
            self._send_alerts(alerts)
    
    def _send_alerts(self, alerts: List[str]):
        """发送告警邮件"""
        if not self.config['alert_email'] or not self.config['email_user']:
            logger.warning("未配置告警邮箱，跳过邮件发送")
            return
        
        try:
            # 创建邮件
            msg = MimeMultipart()
            msg['From'] = self.config['email_user']
            msg['To'] = self.config['alert_email']
            msg['Subject'] = "AI股市预测系统告警"
            
            body = "检测到以下性能问题：\n\n" + "\n".join(f"• {alert}" for alert in alerts)
            body += f"\n\n告警时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email_user'], self.config['email_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("告警邮件发送成功")
            
        except Exception as e:
            logger.error(f"发送告警邮件失败: {str(e)}")
    
    def generate_performance_report(self, days: int = 30) -> Dict:
        """生成性能报告"""
        logger.info(f"生成最近 {days} 天的性能报告...")
        
        report = {
            'report_date': datetime.now().isoformat(),
            'monitoring_period': f"{days} days",
            'models_performance': {},
            'system_health': {},
            'recommendations': []
        }
        
        # 各预测天数的性能
        for prediction_days in [1, 3, 5]:
            df = self.db.get_recent_performance(days=days, prediction_days=prediction_days)
            
            if len(df) > 0:
                metrics = self.calculate_performance_metrics(df)
                report['models_performance'][f'{prediction_days}d'] = metrics
                
                # 性能趋势分析
                trend_analysis = self._analyze_performance_trend(df)
                report['models_performance'][f'{prediction_days}d']['trend'] = trend_analysis
        
        # 生成建议
        report['recommendations'] = self._generate_recommendations(report['models_performance'])
        
        return report
    
    def _analyze_performance_trend(self, df: pd.DataFrame) -> Dict:
        """分析性能趋势"""
        if len(df) < 14:  # 至少需要2周数据
            return {'trend': 'insufficient_data'}
        
        # 按日期分组计算准确率
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_accuracy = df.groupby('date')['correct'].mean()
        
        if len(daily_accuracy) < 7:
            return {'trend': 'insufficient_data'}
        
        # 计算趋势
        recent_accuracy = daily_accuracy.tail(7).mean()
        earlier_accuracy = daily_accuracy.head(7).mean()
        
        trend_direction = 'improving' if recent_accuracy > earlier_accuracy else 'declining'
        trend_magnitude = abs(recent_accuracy - earlier_accuracy)
        
        return {
            'trend': trend_direction,
            'magnitude': float(trend_magnitude),
            'recent_accuracy': float(recent_accuracy),
            'earlier_accuracy': float(earlier_accuracy)
        }
    
    def _generate_recommendations(self, models_performance: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        for model_name, metrics in models_performance.items():
            if 'accuracy' not in metrics:
                continue
            
            accuracy = metrics['accuracy']
            precision = metrics.get('precision', 0)
            
            # 准确率建议
            if accuracy < 0.55:
                recommendations.append(f"{model_name}模型准确率过低({accuracy:.3f})，建议重新训练或调整特征")
            
            # 精确率建议
            if precision < 0.50:
                recommendations.append(f"{model_name}模型精确率过低({precision:.3f})，建议调整决策阈值")
            
            # 趋势建议
            if 'trend' in metrics and metrics['trend'].get('trend') == 'declining':
                recommendations.append(f"{model_name}模型性能呈下降趋势，建议增加新的训练数据")
        
        # 通用建议
        if not recommendations:
            recommendations.append("所有模型表现良好，建议继续监控性能变化")
        
        return recommendations
    
    def create_performance_dashboard(self, days: int = 30) -> str:
        """创建性能监控仪表板"""
        logger.info("创建性能监控仪表板...")
        
        # 获取数据
        all_data = []
        for prediction_days in [1, 3, 5]:
            df = self.db.get_recent_performance(days=days, prediction_days=prediction_days)
            if len(df) > 0:
                df['model'] = f'{prediction_days}d'
                all_data.append(df)
        
        if not all_data:
            return "no_data.html"
        
        df_all = pd.concat(all_data, ignore_index=True)
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        df_all['date'] = df_all['timestamp'].dt.date
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['准确率趋势', '预测概率分布', '每日预测数量', '胜率统计'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 准确率趋势
        daily_accuracy = df_all.groupby(['date', 'model'])['correct'].mean().reset_index()
        for model in daily_accuracy['model'].unique():
            model_data = daily_accuracy[daily_accuracy['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['date'],
                    y=model_data['correct'],
                    mode='lines+markers',
                    name=f'{model} 准确率',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. 预测概率分布
        fig.add_trace(
            go.Histogram(
                x=df_all['predicted_probability'],
                nbinsx=20,
                name='概率分布',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. 每日预测数量
        daily_count = df_all.groupby(['date', 'model']).size().reset_index(name='count')
        for model in daily_count['model'].unique():
            model_data = daily_count[daily_count['model'] == model]
            fig.add_trace(
                go.Bar(
                    x=model_data['date'],
                    y=model_data['count'],
                    name=f'{model} 预测数量'
                ),
                row=2, col=1
            )
        
        # 4. 胜率统计
        win_rate = df_all.groupby('model')['correct'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=win_rate['model'],
                y=win_rate['correct'],
                name='胜率',
                text=[f'{x:.1%}' for x in win_rate['correct']],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='AI股市预测系统性能监控仪表板',
            height=800,
            showlegend=True
        )
        
        # 保存为HTML文件
        dashboard_path = f"performance_dashboard_{datetime.now().strftime('%Y%m%d')}.html"
        fig.write_html(dashboard_path)
        
        logger.info(f"性能仪表板已保存: {dashboard_path}")
        return dashboard_path
    
    def export_performance_data(self, days: int = 30, format: str = 'csv') -> str:
        """导出性能数据"""
        logger.info(f"导出最近 {days} 天的性能数据...")
        
        # 获取所有预测数据
        all_data = []
        for prediction_days in [1, 3, 5]:
            df = self.db.get_recent_performance(days=days, prediction_days=prediction_days)
            if len(df) > 0:
                all_data.append(df)
        
        if not all_data:
            logger.warning("没有性能数据可导出")
            return None
        
        df_export = pd.concat(all_data, ignore_index=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_data_{timestamp}.{format}"
        
        # 导出文件
        if format == 'csv':
            df_export.to_csv(filename, index=False, encoding='utf-8-sig')
        elif format == 'excel':
            df_export.to_excel(filename, index=False)
        elif format == 'json':
            df_export.to_json(filename, orient='records', force_ascii=False, indent=2)
        
        logger.info(f"性能数据已导出: {filename}")
        return filename


class AutoOptimizer:
    """
    自动优化器
    基于性能监控结果自动调整模型参数和策略
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history = []
        
        # 优化配置
        self.config = {
            'min_optimization_interval_days': 7,  # 最小优化间隔
            'performance_decline_threshold': 0.05,  # 性能下降阈值
            'min_sample_size_for_optimization': 100,
        }
    
    def should_optimize(self, prediction_days: int) -> bool:
        """判断是否需要优化"""
        # 获取最近性能数据
        df = self.monitor.db.get_recent_performance(days=30, prediction_days=prediction_days)
        
        if len(df) < self.config['min_sample_size_for_optimization']:
            return False
        
        # 检查性能趋势
        recent_performance = df.tail(50)['correct'].mean()
        earlier_performance = df.head(50)['correct'].mean()
        
        performance_decline = earlier_performance - recent_performance
        
        if performance_decline > self.config['performance_decline_threshold']:
            logger.info(f"{prediction_days}天预测性能下降 {performance_decline:.3f}，需要优化")
            return True
        
        return False
    
    def optimize_model_weights(self, prediction_days: int) -> Dict:
        """优化模型权重"""
        logger.info(f"开始优化 {prediction_days} 天预测模型权重...")
        
        # 获取各子模型的历史表现（这里简化处理）
        df = self.monitor.db.get_recent_performance(days=30, prediction_days=prediction_days)
        
        if len(df) < 50:
            logger.warning("数据不足，无法优化权重")
            return {}
        
        # 基于准确率调整权重（简化版本）
        base_accuracy = df['correct'].mean()
        
        # 新的权重分配策略
        if base_accuracy > 0.6:
            # 性能好时，保持当前权重分布
            new_weights = {'LSTM': 0.4, 'XGBoost': 0.3, 'Transformer': 0.2, 'CNN-LSTM': 0.1}
        elif base_accuracy > 0.55:
            # 性能中等时，增加XGBoost权重
            new_weights = {'LSTM': 0.35, 'XGBoost': 0.35, 'Transformer': 0.2, 'CNN-LSTM': 0.1}
        else:
            # 性能较差时，更依赖传统方法
            new_weights = {'LSTM': 0.3, 'XGBoost': 0.4, 'Transformer': 0.2, 'CNN-LSTM': 0.1}
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'prediction_days': prediction_days,
            'old_accuracy': base_accuracy,
            'new_weights': new_weights,
            'optimization_type': 'weight_adjustment'
        })
        
        logger.info(f"权重优化完成: {new_weights}")
        return new_weights
    
    def auto_optimization_check(self):
        """自动优化检查"""
        logger.info("开始自动优化检查...")
        
        optimizations = []
        
        for prediction_days in [1, 3, 5]:
            if self.should_optimize(prediction_days):
                # 执行优化
                new_weights = self.optimize_model_weights(prediction_days)
                if new_weights:
                    optimizations.append({
                        'prediction_days': prediction_days,
                        'weights': new_weights
                    })
        
        if optimizations:
            logger.info(f"执行了 {len(optimizations)} 个模型优化")
            # 这里应该更新实际的模型权重
            # 实际实现时需要与预测服务集成
        else:
            logger.info("所有模型性能良好，无需优化")
        
        return optimizations


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    # 模拟一些预测数据
    import random
    
    # 生成模拟数据
    for i in range(100):
        stock_code = random.choice(['sh600519', 'sz000001', 'sz000002'])
        prediction_days = random.choice([1, 3, 5])
        predicted_direction = random.choice([0, 1])
        predicted_probability = random.uniform(0.3, 0.9)
        
        # 保存预测
        monitor.db.save_prediction(
            stock_code=stock_code,
            prediction_days=prediction_days,
            predicted_direction=predicted_direction,
            predicted_probability=predicted_probability
        )
        
        # 模拟实际结果（延迟几天）
        if random.random() > 0.3:  # 70%的预测有结果
            actual_direction = random.choice([0, 1])
            actual_return = random.uniform(-0.1, 0.1)
            
            monitor.db.update_prediction_result(
                stock_code=stock_code,
                prediction_days=prediction_days,
                actual_direction=actual_direction,
                actual_return=actual_return,
                timestamp_from=datetime.now() - timedelta(days=5)
            )
    
    # 执行性能检查
    monitor.daily_performance_check()
    
    # 生成报告
    report = monitor.generate_performance_report(days=30)
    print("性能报告生成完成")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 创建仪表板
    dashboard_file = monitor.create_performance_dashboard(days=30)
    print(f"仪表板文件: {dashboard_file}")
    
    # 导出数据
    export_file = monitor.export_performance_data(days=30, format='csv')
    print(f"数据导出文件: {export_file}")
    
    # 自动优化
    optimizer = AutoOptimizer(monitor)
    optimizations = optimizer.auto_optimization_check()
    print(f"执行的优化: {optimizations}")
    
    print("性能监控系统测试完成！")