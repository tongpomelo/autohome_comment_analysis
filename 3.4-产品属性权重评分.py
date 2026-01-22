import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple


class ProductAttributeWeightCalculator:
    def __init__(self, scored_comments_path: str, attributes_path: str,
                 comment_attributes_path: str, word_freq_path: str = None,
                 output_dir: str = None):
        """
        优化的产品属性权重计算器

        参数:
        scored_comments_path: 情感得分评论文件路径
        attributes_path: 产品属性分类表路径
        comment_attributes_path: 评论-属性匹配数据路径
        word_freq_path: 词频统计文件路径(可选)
        output_dir: 输出目录
        """
        # 设置输出目录
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'attribute_analysis_results')
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化日志
        self._init_logging()

        # 加载数据
        self.scored_comments, self.attributes, self.comment_attributes = self._load_data(
            scored_comments_path, attributes_path, comment_attributes_path
        )

        # 加载词频数据(如果有)
        self.word_freq = self._load_word_freq(word_freq_path) if word_freq_path else None

        # 构建属性映射
        self.attribute_full_paths = self._build_attribute_full_paths()

        # 构建属性权重字典
        self.attribute_weights = self._build_attribute_weights()

        self.logger.info("产品属性权重计算器初始化完成")

    def _init_logging(self):
        """初始化日志系统"""
        log_file = os.path.join(self.output_dir,
                                f'attribute_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_data(self, scored_comments_path: str, attributes_path: str,
                   comment_attributes_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载所有必要的数据文件"""

        def try_read_csv(file_path: str) -> pd.DataFrame:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(f"成功读取文件 {file_path}，编码: {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"编码 {encoding} 读取失败: {e}")
                    continue

            # 如果所有编码都失败，尝试指定错误处理
            try:
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
                self.logger.warning(f"使用错误替换模式读取文件 {file_path}")
                return df
            except Exception as e:
                self.logger.error(f"所有编码尝试都失败: {e}")
                raise

        def try_read_excel(file_path: str) -> pd.DataFrame:
            """处理Excel文件"""
            try:
                # 如果是Excel文件
                if file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                    self.logger.info(f"成功读取Excel文件 {file_path}")
                    return df
                else:
                    # 如果是CSV文件，使用CSV读取方法
                    return try_read_csv(file_path)
            except Exception as e:
                self.logger.error(f"读取Excel文件失败: {e}")
                raise

        try:
            # 分别处理不同文件
            scored_comments = try_read_csv(scored_comments_path)
            attributes = try_read_excel(attributes_path)  # 你的属性文件是Excel格式
            comment_attributes = try_read_csv(comment_attributes_path)

            self.logger.info(f"成功加载数据: 评论{len(scored_comments)}条, "
                             f"属性{len(attributes)}个, 属性匹配{len(comment_attributes)}条")
            return scored_comments, attributes, comment_attributes

        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

    def _load_word_freq(self, word_freq_path: str) -> Dict[str, int]:
        """加载词频数据"""
        try:
            # 尝试多种编码读取词频文件
            encodings = ['utf-8', 'gbk', 'gb2312']
            word_freq_df = None

            for encoding in encodings:
                try:
                    word_freq_df = pd.read_csv(word_freq_path, encoding=encoding)
                    self.logger.info(f"成功加载词频数据，编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if word_freq_df is None:
                # 如果所有编码都失败，尝试自动检测
                word_freq_df = pd.read_csv(word_freq_path, encoding='utf-8', errors='replace')
                self.logger.warning("使用错误替换模式加载词频数据")

            # 查找词和频率列
            word_col = None
            freq_col = None
            possible_word_cols = ['word', '词汇', '词语', '词']
            possible_freq_cols = ['frequency', 'freq', '频率', '词频']

            for col in word_freq_df.columns:
                if col in possible_word_cols:
                    word_col = col
                if col in possible_freq_cols:
                    freq_col = col

            if not word_col:
                word_col = word_freq_df.columns[0]
            if not freq_col and len(word_freq_df.columns) >= 2:
                freq_col = word_freq_df.columns[1]

            word_freq_dict = dict(zip(word_freq_df[word_col], word_freq_df[freq_col]))
            self.logger.info(f"成功加载词频数据，共 {len(word_freq_dict)} 个词汇")
            return word_freq_dict

        except Exception as e:
            self.logger.warning(f"加载词频数据失败: {e}")
            return None

    def _build_attribute_full_paths(self) -> Dict[str, Dict[str, str]]:
        """构建三级分类完整路径映射"""
        full_paths = {}
        for _, row in self.attributes.iterrows():
            full_path = f"{row['一级分类']}-{row['二级分类']}-{row['三级分类']}"
            full_paths[row['三级分类']] = {
                'full_path': full_path,
                'level1': row['一级分类'],
                'level2': row['二级分类'],
                'level3': row['三级分类']
            }
        self.logger.info(f"构建属性路径映射完成，共 {len(full_paths)} 个属性")
        return full_paths

    def calculate_decision_weights(self) -> Dict[str, float]:
        """
        计算决策参考权重
        基于词频数据或评论中出现频率
        """
        if self.word_freq is not None:
            return self._calculate_weights_from_word_freq()
        else:
            return self._calculate_weights_from_comments()

    def _calculate_weights_from_word_freq(self) -> Dict[str, float]:
        """基于词频数据计算决策参考权重"""
        self.logger.info("基于词频数据计算决策参考权重...")

        # 按一级分类统计总词频
        level1_totals = defaultdict(int)
        attribute_freq = {}

        for _, row in self.attributes.iterrows():
            level1 = row['一级分类']
            level3 = row['三级分类']

            # 获取词频
            freq = self.word_freq.get(level3, 0)
            attribute_freq[level3] = freq
            level1_totals[level1] += freq

        # 计算权重百分比
        decision_weights = {}
        for level3, freq in attribute_freq.items():
            level1 = None
            for _, row in self.attributes.iterrows():
                if row['三级分类'] == level3:
                    level1 = row['一级分类']
                    break

            if level1 and level1_totals[level1] > 0:
                weight_percent = (freq / level1_totals[level1]) * 100
                decision_weights[level3] = round(weight_percent, 2)
            else:
                decision_weights[level3] = 0.0

        self.logger.info(f"词频权重计算完成，共 {len(decision_weights)} 个属性")
        return decision_weights

    def _calculate_weights_from_comments(self) -> Dict[str, float]:
        """基于评论中出现频率计算决策参考权重"""
        self.logger.info("基于评论数据计算决策参考权重...")

        # 统计每个三级分类在评论中出现的次数
        attribute_counts = defaultdict(int)

        for _, row in self.comment_attributes.iterrows():
            attribute = row['命中属性词']
            attribute_counts[attribute] += 1

        # 按一级分类统计总出现次数
        level1_totals = defaultdict(int)
        for level3, count in attribute_counts.items():
            if level3 in self.attribute_full_paths:
                level1 = self.attribute_full_paths[level3]['level1']
                level1_totals[level1] += count

        # 计算权重百分比
        decision_weights = {}
        for level3, count in attribute_counts.items():
            if level3 in self.attribute_full_paths:
                level1 = self.attribute_full_paths[level3]['level1']
                if level1_totals[level1] > 0:
                    weight_percent = (count / level1_totals[level1]) * 100
                    decision_weights[level3] = round(weight_percent, 2)
                else:
                    decision_weights[level3] = 0.0

        self.logger.info(f"评论频率权重计算完成，共 {len(decision_weights)} 个属性")
        return decision_weights

    def _build_attribute_weights(self) -> Dict[str, float]:
        """构建属性权重字典 - 使用计算得到的决策参考权重"""
        decision_weights = self.calculate_decision_weights()

        weights = {}
        for _, row in self.attributes.iterrows():
            level3 = row['三级分类']
            weight = decision_weights.get(level3, 0.0) / 100  # 转换为小数
            weights[level3] = weight

        self.logger.info(f"属性权重字典构建完成，共 {len(weights)} 个属性")
        return weights

    def select_representative_attribute(self, comment_id: str) -> str:
        """
        为每条评论选择代表性属性（决策参考权重最高的）
        """
        # 查找评论ID列名
        comment_id_col = None
        possible_id_cols = ['评论ID', 'comment_id', 'id', '评论id']

        for col in self.comment_attributes.columns:
            if col in possible_id_cols:
                comment_id_col = col
                break

        if not comment_id_col:
            comment_id_col = self.comment_attributes.columns[0]  # 使用第一列作为备选

        # 查找属性词列名
        attribute_col = None
        possible_attribute_cols = ['命中属性词', 'attribute_word', 'attribute', '属性词']

        for col in self.comment_attributes.columns:
            if col in possible_attribute_cols:
                attribute_col = col
                break

        if not attribute_col:
            attribute_col = self.comment_attributes.columns[1]  # 使用第二列作为备选

        # 获取该评论命中的所有属性
        comment_attrs = self.comment_attributes[
            self.comment_attributes[comment_id_col] == comment_id
            ]

        if comment_attrs.empty:
            return None

        # 计算每个属性的综合权重
        attr_scores = {}
        for _, attr_row in comment_attrs.iterrows():
            attr_word = attr_row[attribute_col]
            if attr_word in self.attribute_weights:
                attr_scores[attr_word] = self.attribute_weights[attr_word]

        if not attr_scores:
            return None

        # 选择权重最高的属性
        representative_attr = max(attr_scores.items(), key=lambda x: x[1])[0]
        return representative_attr

    def calculate_attribute_scores(self) -> Dict[str, Dict[str, Any]]:
        """
        计算产品属性权重评分

        返回:
        各属性的平均情感得分和统计信息
        """
        # 首先检查数据框的列名
        self.logger.info(f"scored_comments 列名: {list(self.scored_comments.columns)}")
        self.logger.info(f"comment_attributes 列名: {list(self.comment_attributes.columns)}")

        # 查找情感得分列（可能有不同的列名）
        sentiment_col = None
        comment_id_col = None
        comment_content_col = None

        # 常见的情感得分列名
        possible_sentiment_cols = ['情感得分', 'sentiment_score', 'score', '情感分数', 'sentiment']
        possible_id_cols = ['评论ID', 'comment_id', 'id', '评论id']
        possible_content_cols = ['评论内容', 'comment_content', 'content', '评论', 'comment']

        for col in self.scored_comments.columns:
            if col in possible_sentiment_cols:
                sentiment_col = col
            if col in possible_id_cols:
                comment_id_col = col
            if col in possible_content_cols:
                comment_content_col = col

        # 如果没有找到标准列名，使用第一列作为备选
        if not sentiment_col and len(self.scored_comments.columns) >= 2:
            sentiment_col = self.scored_comments.columns[1]  # 假设第二列是情感得分
        if not comment_id_col:
            comment_id_col = self.scored_comments.columns[0]  # 假设第一列是评论ID
        if not comment_content_col and len(self.scored_comments.columns) >= 3:
            comment_content_col = self.scored_comments.columns[2]  # 假设第三列是评论内容

        self.logger.info(
            f"使用列名 - 情感得分: {sentiment_col}, 评论ID: {comment_id_col}, 评论内容: {comment_content_col}")

        if not sentiment_col:
            raise KeyError("无法找到情感得分列，请检查数据文件列名")

        # 只考虑情感得分非0的评论
        non_zero_scores = self.scored_comments[
            self.scored_comments[sentiment_col] != 0
            ].copy()

        self.logger.info(f"参与计算的评论数量: {len(non_zero_scores)}")

        # 为每条评论选择代表性属性
        non_zero_scores['代表性属性'] = non_zero_scores[comment_id_col].apply(
            self.select_representative_attribute
        )

        # 移除没有代表性属性的评论
        valid_scores = non_zero_scores.dropna(subset=['代表性属性'])
        self.logger.info(f"有效评论数量(有代表性属性): {len(valid_scores)}")

        # 计算每个属性的统计
        attribute_stats = defaultdict(lambda: {
            'total_score': 0,
            'count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'comments': []
        })

        for _, row in valid_scores.iterrows():
            attr = row['代表性属性']
            score = row[sentiment_col]
            comment = row[comment_content_col] if comment_content_col else "无内容"

            attribute_stats[attr]['total_score'] += score
            attribute_stats[attr]['count'] += 1

            if score > 0:
                attribute_stats[attr]['positive_count'] += 1
            elif score < 0:
                attribute_stats[attr]['negative_count'] += 1

            attribute_stats[attr]['comments'].append({
                'comment': comment,
                'score': score
            })

        # 计算平均得分和其他统计
        result = {}
        for attr, stats in attribute_stats.items():
            if stats['count'] > 0:
                avg_score = stats['total_score'] / stats['count']
                positive_ratio = stats['positive_count'] / stats['count']
                negative_ratio = stats['negative_count'] / stats['count']

                result[attr] = {
                    '平均情感得分': round(avg_score, 4),
                    '评论数量': stats['count'],
                    '总情感得分': round(stats['total_score'], 4),
                    '正面评论数': stats['positive_count'],
                    '负面评论数': stats['negative_count'],
                    '正面评论比例': round(positive_ratio, 4),
                    '负面评论比例': round(negative_ratio, 4),
                    '决策参考权重': round(self.attribute_weights.get(attr, 0) * 100, 2),
                    '示例评论': stats['comments'][:3]  # 保存前3条示例评论
                }

        self.logger.info(f"属性评分计算完成，共 {len(result)} 个属性有评分")
        return result

    def calculate_by_category(self, attribute_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
        """
        按分类级别计算权重评分
        """
        # 按一级分类聚合
        level1_scores = defaultdict(lambda: {'total_score': 0, 'count': 0, 'attributes': []})
        level2_scores = defaultdict(lambda: {'total_score': 0, 'count': 0, 'attributes': []})

        for attr, scores in attribute_scores.items():
            if attr in self.attribute_full_paths:
                path_info = self.attribute_full_paths[attr]
                level1 = path_info['level1']
                level2 = f"{level1}-{path_info['level2']}"

                # 一级分类统计
                level1_scores[level1]['total_score'] += scores['总情感得分']
                level1_scores[level1]['count'] += scores['评论数量']
                level1_scores[level1]['attributes'].append(attr)

                # 二级分类统计
                level2_scores[level2]['total_score'] += scores['总情感得分']
                level2_scores[level2]['count'] += scores['评论数量']
                level2_scores[level2]['attributes'].append(attr)

        # 计算平均分
        level1_result = {}
        for category, stats in level1_scores.items():
            if stats['count'] > 0:
                level1_result[category] = {
                    '平均情感得分': round(stats['total_score'] / stats['count'], 4),
                    '评论数量': stats['count'],
                    '包含属性数': len(stats['attributes']),
                    '属性示例': stats['attributes'][:5]
                }

        level2_result = {}
        for category, stats in level2_scores.items():
            if stats['count'] > 0:
                level2_result[category] = {
                    '平均情感得分': round(stats['total_score'] / stats['count'], 4),
                    '评论数量': stats['count'],
                    '包含属性数': len(stats['attributes']),
                    '属性列表': stats['attributes']
                }

        return {
            '一级分类': level1_result,
            '二级分类': level2_result
        }

    def generate_decision_weight_report(self) -> pd.DataFrame:
        """生成决策参考权重报告"""
        decision_weights = self.calculate_decision_weights()

        report_data = []
        for _, row in self.attributes.iterrows():
            level3 = row['三级分类']
            weight = decision_weights.get(level3, 0.0)

            report_data.append({
                '一级分类': row['一级分类'],
                '二级分类': row['二级分类'],
                '三级分类': level3,
                '决策参考权重(%)': weight
            })

        report_df = pd.DataFrame(report_data)

        # 按一级分类和权重排序
        report_df = report_df.sort_values(['一级分类', '决策参考权重(%)'], ascending=[True, False])

        self.logger.info("决策参考权重报告生成完成")
        return report_df

    def save_results(self, attribute_scores: Dict[str, Dict[str, Any]],
                     category_scores: Dict[str, Dict],
                     decision_weight_report: pd.DataFrame) -> Dict[str, str]:
        """保存完整结果"""

        # 1. 保存属性级评分结果
        attr_results = []
        for attr, scores in attribute_scores.items():
            if attr in self.attribute_full_paths:
                path_info = self.attribute_full_paths[attr]
                attr_results.append({
                    '一级分类': path_info['level1'],
                    '二级分类': path_info['level2'],
                    '三级分类': attr,
                    '平均情感得分': scores['平均情感得分'],
                    '评论数量': scores['评论数量'],
                    '总情感得分': scores['总情感得分'],
                    '正面评论数': scores['正面评论数'],
                    '负面评论数': scores['负面评论数'],
                    '正面评论比例': scores['正面评论比例'],
                    '负面评论比例': scores['负面评论比例'],
                    '决策参考权重(%)': scores['决策参考权重']
                })

        attr_df = pd.DataFrame(attr_results)
        attr_file = os.path.join(self.output_dir, 'attribute_scores_detailed.csv')
        attr_df.to_csv(attr_file, index=False, encoding='utf-8-sig')

        # 2. 保存分类级评分结果
        category_file = os.path.join(self.output_dir, 'category_scores.json')
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(category_scores, f, ensure_ascii=False, indent=2)

        # 3. 保存决策参考权重报告
        decision_file = os.path.join(self.output_dir, 'decision_weights_report.csv')
        decision_weight_report.to_csv(decision_file, index=False, encoding='utf-8-sig')

        # 4. 生成汇总报告
        summary_file = self._generate_summary_report(attribute_scores, category_scores, decision_weight_report)

        self.logger.info(f"所有结果已保存到: {self.output_dir}")

        return {
            'attribute_scores': attr_file,
            'category_scores': category_file,
            'decision_weights': decision_file,
            'summary_report': summary_file
        }

    def _generate_summary_report(self, attribute_scores: Dict[str, Dict[str, Any]],
                                 category_scores: Dict[str, Dict],
                                 decision_weight_report: pd.DataFrame) -> str:
        """生成分析摘要报告"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("产品属性权重评分分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总属性数量: {len(self.attributes)}")
        report_lines.append(f"有评分属性数量: {len(attribute_scores)}")
        report_lines.append(f"参与分析评论数量: {len(self.scored_comments)}")
        report_lines.append("")

        # TOP10 正面属性
        positive_attrs = [(k, v) for k, v in attribute_scores.items() if v['平均情感得分'] > 0]
        positive_attrs.sort(key=lambda x: x[1]['平均情感得分'], reverse=True)

        report_lines.append("TOP10 正面评价属性:")
        report_lines.append("-" * 80)
        for attr, scores in positive_attrs[:10]:
            report_lines.append(f"{attr:20} | 得分: {scores['平均情感得分']:+.4f} | "
                                f"评论数: {scores['评论数量']:4d} | "
                                f"正面率: {scores['正面评论比例']:6.2%} | "
                                f"权重: {scores['决策参考权重']:6.2f}%")

        report_lines.append("")

        # TOP10 负面属性
        negative_attrs = [(k, v) for k, v in attribute_scores.items() if v['平均情感得分'] < 0]
        negative_attrs.sort(key=lambda x: x[1]['平均情感得分'])

        report_lines.append("TOP10 负面评价属性:")
        report_lines.append("-" * 80)
        for attr, scores in negative_attrs[:10]:
            report_lines.append(f"{attr:20} | 得分: {scores['平均情感得分']:+.4f} | "
                                f"评论数: {scores['评论数量']:4d} | "
                                f"负面率: {scores['负面评论比例']:6.2%} | "
                                f"权重: {scores['决策参考权重']:6.2f}%")

        report_lines.append("")

        # 一级分类表现
        report_lines.append("一级分类表现:")
        report_lines.append("-" * 80)
        for category, scores in category_scores['一级分类'].items():
            report_lines.append(f"{category:15} | 平均得分: {scores['平均情感得分']:+.4f} | "
                                f"评论数: {scores['评论数量']:5d} | "
                                f"属性数: {scores['包含属性数']:3d}")

        # 保存报告
        report_file = os.path.join(self.output_dir, 'analysis_summary_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # 打印报告
        print('\n'.join(report_lines))

        self.logger.info(f"摘要报告已保存: {report_file}")
        return report_file

    def run_analysis(self) -> Dict[str, Any]:
        """
        运行完整的分析流程

        返回:
        分析结果汇总
        """
        self.logger.info("开始产品属性权重评分分析...")

        try:
            # 1. 计算属性权重评分
            attribute_scores = self.calculate_attribute_scores()

            # 2. 按分类聚合
            category_scores = self.calculate_by_category(attribute_scores)

            # 3. 生成决策参考权重报告
            decision_weight_report = self.generate_decision_weight_report()

            # 4. 保存所有结果
            result_files = self.save_results(attribute_scores, category_scores, decision_weight_report)

            # 汇总结果
            results = {
                'attribute_scores': attribute_scores,
                'category_scores': category_scores,
                'decision_weights': dict(zip(decision_weight_report['三级分类'],
                                             decision_weight_report['决策参考权重(%)'])),
                'result_files': result_files
            }

            self.logger.info("产品属性权重评分分析完成!")
            return results

        except Exception as e:
            self.logger.error(f"分析过程中出现错误: {e}")
            raise


def main():
    """主函数示例"""
    # 文件路径配置
    scored_comments_path = "car_review_analysis_results/detailed_sentiment_analysis.csv"  # 来自情感分析程序
    attributes_path = "30.自定义词典/汽车描述词库V1.xlsx"  # 您的三级词库文件
    comment_attributes_path = "car_review_analysis_results/comment_attributes.csv"  # 评论-属性匹配数据
    word_freq_path = "word_frequency.csv"  # 可选:词频统计文件

    # 输出目录
    output_dir = "product_attribute_analysis"

    try:
        # 创建计算器
        calculator = ProductAttributeWeightCalculator(
            scored_comments_path=scored_comments_path,
            attributes_path=attributes_path,
            comment_attributes_path=comment_attributes_path,
            word_freq_path=word_freq_path,
            output_dir=output_dir
        )

        # 运行完整分析
        results = calculator.run_analysis()

        print("\n" + "=" * 80)
        print("分析完成！生成的文件:")
        for file_type, file_path in results['result_files'].items():
            print(f"  {file_type}: {file_path}")
        print("=" * 80)

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()