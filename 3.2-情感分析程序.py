import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
import json
from datetime import datetime
import logging

# 导入之前定义的情感分析处理器
from sentiment_analysis_model import CommentProcessor


class CarReviewAnalyzer:
    def __init__(self, data_path, output_dir=None):
        """
        汽车评论分析器

        参数:
        data_path: Excel文件路径
        output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'car_review_analysis')
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化日志
        self._init_logging()

        # 加载数据
        self.df = self.load_data()

        # 初始化情感分析处理器
        self.processor = self.init_sentiment_processor()

        self.logger.info(f"成功加载数据，共 {len(self.df)} 条记录")

    def _init_logging(self):
        """初始化日志系统"""
        log_file = os.path.join(self.output_dir,
                                f'car_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """加载Excel数据"""
        try:
            df = pd.read_excel(self.data_path, sheet_name='clean_data_xlsx')
            self.logger.info(f"数据加载成功，形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise

    def init_sentiment_processor(self):
        """初始化情感分析处理器"""
        # 文件路径配置 - 请根据您的实际路径调整
        professional_dict_path = "30.自定义词典/汽车专业词典_最终版.txt"
        sentiment_dict_paths = {
            'positive': "31.情感词典/知网Hownet情感词典-中文/正面情感词语（中文）.txt",
            'negative': "31.情感词典/知网Hownet情感词典-中文/负面情感词语（中文）.txt",
            'perception': "31.情感词典/知网Hownet情感词典-中文/主张词语（中文）.txt",
            'positive_eval': "31.情感词典/知网Hownet情感词典-中文/正面评价词语（中文）.txt",
            'negative_eval': "31.情感词典/知网Hownet情感词典-中文/负面评价词语（中文）.txt"
        }
        degree_words_path = "31.情感词典/知网Hownet情感词典-中文/程度级别词语（中文）.txt"

        try:
            processor = CommentProcessor(
                professional_dict_path=professional_dict_path,
                sentiment_dict_paths=sentiment_dict_paths,
                degree_words_path=degree_words_path,
                output_dir=os.path.join(self.output_dir, 'sentiment_analysis')
            )
            self.logger.info("情感分析处理器初始化成功")
            return processor
        except Exception as e:
            self.logger.error(f"情感分析处理器初始化失败: {e}")
            # 返回None，后续可以使用简单的情感分析方法
            return None

    def extract_text_columns(self):
        """提取需要分析的文本列"""
        text_columns = {
            '最满意': '最满意',
            '最不满意': '最不满意',
            '空间评论': '空间评论',
            '驾驶感受评论': '驾驶感受评论',
            '续航评论': '续航评论',
            '外观评论': '外观评论',
            '内饰评论': '内饰评论',
            '性价比评论': '性价比评论',
            '智能化评论': '智能化评论'
        }
        return text_columns

    def simple_sentiment_analysis(self, text):
        """
        简单的情感分析（基于关键词）
        作为情感分析处理器的备选方案
        """
        if pd.isna(text) or text == '':
            return 0

        positive_words = ['好', '满意', '不错', '优秀', '棒', '赞', '喜欢', '爱', '舒适', '宽敞', '漂亮', '时尚']
        negative_words = ['差', '不满意', '不好', '糟糕', '问题', '缺陷', '不足', '缺点', '讨厌', '拥挤', '难看']

        text = str(text)
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return -1
        else:
            return 0

    def analyze_single_column(self, column_name, comments):
        """
        分析单个文本列的情感
        """
        results = []

        for comment in comments:
            if pd.isna(comment) or comment == '':
                results.append(0)
                continue

            try:
                if self.processor:
                    # 使用专业的情感分析
                    score = self.processor.calculate_sentiment_score(str(comment))
                    results.append(score)
                else:
                    # 使用简单的情感分析
                    score = self.simple_sentiment_analysis(comment)
                    results.append(score)
            except Exception as e:
                self.logger.warning(f"情感分析失败: {e}, 使用简单分析")
                score = self.simple_sentiment_analysis(comment)
                results.append(score)

        return results

    def analyze_all_columns(self):
        """
        分析所有文本列的情感
        """
        text_columns = self.extract_text_columns()
        analysis_results = {}

        for col_chinese, col_english in text_columns.items():
            if col_chinese in self.df.columns:
                self.logger.info(f"正在分析列: {col_chinese}")
                comments = self.df[col_chinese].fillna('')
                scores = self.analyze_single_column(col_chinese, comments)
                analysis_results[col_english] = scores
            else:
                self.logger.warning(f"列不存在: {col_chinese}")

        return analysis_results

    def calculate_statistics(self, analysis_results):
        """
        计算情感统计信息
        """
        stats = {}

        for column, scores in analysis_results.items():
            scores_array = np.array(scores)

            # 基本统计
            positive_count = np.sum(scores_array > 0)
            negative_count = np.sum(scores_array < 0)
            neutral_count = np.sum(scores_array == 0)
            total_count = len(scores_array)

            # 比例
            positive_ratio = positive_count / total_count if total_count > 0 else 0
            negative_ratio = negative_count / total_count if total_count > 0 else 0
            neutral_ratio = neutral_count / total_count if total_count > 0 else 0

            # 平均情感得分
            avg_score = np.mean(scores_array)

            stats[column] = {
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'neutral_count': int(neutral_count),
                'total_count': int(total_count),
                'positive_ratio': round(positive_ratio, 4),
                'negative_ratio': round(negative_ratio, 4),
                'neutral_ratio': round(neutral_ratio, 4),
                'average_score': round(avg_score, 4)
            }

        return stats

    def analyze_by_car_model(self, analysis_results):
        """
        按车型分析情感
        """
        car_models = self.df['车型名称'].unique()
        model_stats = {}

        for model in car_models:
            model_mask = self.df['车型名称'] == model
            model_data = {}

            for column, scores in analysis_results.items():
                model_scores = np.array(scores)[model_mask]

                if len(model_scores) > 0:
                    positive_count = np.sum(model_scores > 0)
                    negative_count = np.sum(model_scores < 0)
                    neutral_count = np.sum(model_scores == 0)
                    total_count = len(model_scores)
                    avg_score = np.mean(model_scores)

                    model_data[column] = {
                        'positive_count': int(positive_count),
                        'negative_count': int(negative_count),
                        'neutral_count': int(neutral_count),
                        'total_count': int(total_count),
                        'average_score': round(avg_score, 4)
                    }

            model_stats[model] = model_data

        return model_stats

    def save_results(self, analysis_results, overall_stats, model_stats):
        """保存分析结果"""

        # 1. 保存详细的情感得分
        detailed_df = self.df.copy()
        for column, scores in analysis_results.items():
            detailed_df[f'{column}_sentiment_score'] = scores
            detailed_df[f'{column}_sentiment'] = ['正面' if s > 0 else '负面' if s < 0 else '中性' for s in scores]

        detailed_file = os.path.join(self.output_dir, 'detailed_sentiment_analysis.csv')
        detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')

        # 2. 保存总体统计
        stats_df = pd.DataFrame(overall_stats).T
        stats_file = os.path.join(self.output_dir, 'overall_sentiment_statistics.csv')
        stats_df.to_csv(stats_file, encoding='utf-8-sig')

        # 3. 保存车型统计
        model_stats_list = []
        for model, stats in model_stats.items():
            for aspect, values in stats.items():
                model_stats_list.append({
                    'car_model': model,
                    'aspect': aspect,
                    **values
                })

        model_stats_df = pd.DataFrame(model_stats_list)
        model_stats_file = os.path.join(self.output_dir, 'model_sentiment_statistics.csv')
        model_stats_df.to_csv(model_stats_file, index=False, encoding='utf-8-sig')

        # 4. 保存JSON格式的完整结果
        full_results = {
            'overall_statistics': overall_stats,
            'model_statistics': model_stats,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_records': len(self.df)
        }

        json_file = os.path.join(self.output_dir, 'complete_analysis_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"所有结果已保存到: {self.output_dir}")

    def generate_summary_report(self, overall_stats, model_stats):
        """生成摘要报告"""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("汽车评论情感分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总记录数: {len(self.df)}")
        report_lines.append(f"车型数量: {len(model_stats)}")
        report_lines.append("")

        # 总体统计
        report_lines.append("总体情感分布:")
        report_lines.append("-" * 40)

        for aspect, stats in overall_stats.items():
            report_lines.append(
                f"{aspect:15} | 正面: {stats['positive_count']:3d} ({stats['positive_ratio']:6.2%}) | "
                f"负面: {stats['negative_count']:3d} ({stats['negative_ratio']:6.2%}) | "
                f"中性: {stats['neutral_count']:3d} ({stats['neutral_ratio']:6.2%}) | "
                f"平均分: {stats['average_score']:+.4f}"
            )

        report_lines.append("")

        # 车型统计
        report_lines.append("各车型情感分析:")
        report_lines.append("-" * 40)

        for model, stats in model_stats.items():
            report_lines.append(f"\n车型: {model}")
            for aspect, values in stats.items():
                report_lines.append(
                    f"  {aspect:15} | 正面: {values['positive_count']:3d} | "
                    f"负面: {values['negative_count']:3d} | 中性: {values['neutral_count']:3d} | "
                    f"平均分: {values['average_score']:+.4f}"
                )

        report_text = "\n".join(report_lines)

        # 保存报告
        report_file = os.path.join(self.output_dir, 'analysis_summary_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # 打印报告
        print(report_text)

        return report_text

    def run_analysis(self):
        """运行完整的分析流程"""
        self.logger.info("开始汽车评论情感分析...")

        # 1. 分析所有文本列
        analysis_results = self.analyze_all_columns()

        # 2. 计算统计信息
        overall_stats = self.calculate_statistics(analysis_results)

        # 3. 按车型分析
        model_stats = self.analyze_by_car_model(analysis_results)

        # 4. 保存结果
        self.save_results(analysis_results, overall_stats, model_stats)

        # 5. 生成报告
        report = self.generate_summary_report(overall_stats, model_stats)

        self.logger.info("汽车评论情感分析完成!")

        return {
            'analysis_results': analysis_results,
            'overall_stats': overall_stats,
            'model_stats': model_stats,
            'report': report
        }


# 使用示例
def main():
    # 数据文件路径
    data_path = "10.汽车口碑数据/merged_data_V2.xlsx"

    # 输出目录
    output_dir = "car_review_analysis_results"

    # 创建分析器
    analyzer = CarReviewAnalyzer(data_path, output_dir)

    # 运行分析
    results = analyzer.run_analysis()

    print("\n" + "=" * 80)
    print("分析完成！结果已保存到以下文件：")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()