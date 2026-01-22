import pandas as pd
import jieba
import re
from typing import List, Set, Tuple, Dict
import os
import logging
from datetime import datetime


class AttributeMatcher:
    def __init__(self, attributes_path: str, custom_dict_path: str = None):
        """
        属性匹配器

        参数:
        attributes_path: 属性词库路径
        custom_dict_path: 自定义词典路径(可选)
        """
        # 加载属性词库
        self.attributes = self._load_attributes(attributes_path)

        # 初始化jieba
        if custom_dict_path and os.path.exists(custom_dict_path):
            jieba.load_userdict(custom_dict_path)

    def _load_attributes(self, attributes_path: str) -> Set[str]:
        """加载属性词库"""
        try:
            # 根据文件扩展名选择读取方式
            if attributes_path.endswith('.xlsx'):
                # 读取Excel文件
                df = pd.read_excel(attributes_path, sheet_name='汽车三级词库')
                # 使用三级分类列作为属性词
                attributes = set(df['三级分类'].dropna().astype(str).unique())
                print(f"从Excel文件加载属性词库: {len(attributes)} 个属性词")
            elif attributes_path.endswith('.csv'):
                # 读取CSV文件
                df = pd.read_csv(attributes_path)
                # 假设三级分类列包含具体的属性词
                attributes = set(df['三级分类'].dropna().astype(str).unique())
                print(f"从CSV文件加载属性词库: {len(attributes)} 个属性词")
            else:
                # 文本文件
                with open(attributes_path, 'r', encoding='utf-8') as f:
                    attributes = set(line.strip() for line in f if line.strip())
                print(f"从文本文件加载属性词库: {len(attributes)} 个属性词")

            # 过滤掉空字符串和None
            attributes = {attr for attr in attributes if attr and str(attr).strip()}
            print(f"有效属性词: {len(attributes)} 个")

            # 添加属性词到jieba词典
            for attr in attributes:
                jieba.add_word(attr)
            print(f"已向jieba添加 {len(attributes)} 个属性词")

            return attributes

        except Exception as e:
            print(f"加载属性词库失败: {e}")
            return set()

    def extract_attributes_from_comment(self, comment: str) -> List[str]:
        """从单条评论中提取属性词"""
        if pd.isna(comment) or not str(comment).strip():
            return []

        comment_str = str(comment)
        found_attributes = []

        # 直接匹配属性词
        for attr in self.attributes:
            if attr in comment_str:
                found_attributes.append(attr)

        return found_attributes

    def batch_match_attributes(self, comments_df: pd.DataFrame,
                               comment_id_col: str = '评论ID',
                               comment_text_col: str = '评论内容') -> pd.DataFrame:
        """批量匹配评论属性"""
        results = []

        # 检查必要的列是否存在
        if comment_id_col not in comments_df.columns:
            print(f"警告: 数据框中没有 '{comment_id_col}' 列，使用索引作为评论ID")
            comments_df = comments_df.reset_index()
            comments_df[comment_id_col] = 'comment_' + comments_df['index'].astype(str)

        if comment_text_col not in comments_df.columns:
            print(f"错误: 数据框中没有 '{comment_text_col}' 列")
            available_cols = list(comments_df.columns)
            print(f"可用列: {available_cols}")
            return pd.DataFrame()

        total_comments = len(comments_df)
        print(f"开始处理 {total_comments} 条评论...")

        for idx, row in comments_df.iterrows():
            comment_id = row[comment_id_col]
            comment_text = row[comment_text_col]

            attributes = self.extract_attributes_from_comment(comment_text)

            for attr in attributes:
                results.append({
                    '评论ID': comment_id,
                    '命中属性词': attr,
                    '评论内容': comment_text[:100] + '...' if len(str(comment_text)) > 100 else comment_text
                    # 保存部分评论内容用于调试
                })

            # 进度显示
            if (idx + 1) % 1000 == 0:
                print(f"已处理 {idx + 1}/{total_comments} 条评论")

        return pd.DataFrame(results)

    def save_matching_results(self, matched_df: pd.DataFrame, output_path: str):
        """保存匹配结果"""
        if len(matched_df) > 0:
            matched_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"属性匹配结果已保存: {output_path}, 共 {len(matched_df)} 条匹配记录")
        else:
            print("警告: 没有找到任何匹配的属性，不保存空文件")


def generate_comment_attributes():
    """生成评论-属性匹配文件的主函数"""

    # 文件路径配置 - 根据您的实际文件调整
    scored_comments_path = "car_review_analysis_results/detailed_sentiment_analysis.csv"  # 情感分析结果文件
    attributes_path = "30.自定义词典/汽车描述词库V1.xlsx"  # 使用您提供的Excel文件
    output_path = "comment_attributes.csv"

    try:
        # 加载情感分析结果
        print(f"正在加载评论数据: {scored_comments_path}")
        scored_comments = pd.read_csv(scored_comments_path, encoding='utf-8-sig')
        print(f"加载评论数据成功: {len(scored_comments)} 条")

        # 显示数据列信息
        print(f"数据列: {list(scored_comments.columns)}")

        # 创建属性匹配器
        print(f"正在加载属性词库: {attributes_path}")
        matcher = AttributeMatcher(attributes_path)

        if not matcher.attributes:
            print("错误: 属性词库加载失败，无法继续")
            return

        # 批量匹配属性
        print("开始属性匹配...")

        # 确定评论内容列 - 检查可用的文本列
        text_columns = ['最满意', '最不满意', '空间评论', '驾驶感受评论', '续航评论',
                        '外观评论', '内饰评论', '性价比评论', '智能化评论']

        available_text_cols = [col for col in text_columns if col in scored_comments.columns]
        print(f"可用的评论列: {available_text_cols}")

        # 合并所有评论内容到一个列中用于匹配
        scored_comments['合并评论'] = ''
        for col in available_text_cols:
            scored_comments['合并评论'] += ' ' + scored_comments[col].fillna('').astype(str)

        # 去除多余空格
        scored_comments['合并评论'] = scored_comments['合并评论'].str.strip()

        # 为评论生成唯一ID（如果不存在）
        if '评论ID' not in scored_comments.columns:
            scored_comments['评论ID'] = ['comment_' + str(i) for i in range(len(scored_comments))]
            print("已生成评论ID")

        # 进行属性匹配
        matched_attributes = matcher.batch_match_attributes(
            scored_comments,
            comment_id_col='评论ID',
            comment_text_col='合并评论'
        )

        # 保存结果
        matcher.save_matching_results(matched_attributes, output_path)

        # 统计信息
        if len(matched_attributes) > 0:
            unique_comments = matched_attributes['评论ID'].nunique()
            unique_attributes = matched_attributes['命中属性词'].nunique()

            print(f"\n匹配统计:")
            print(f"- 涉及评论: {unique_comments} 条")
            print(f"- 匹配属性: {unique_attributes} 种")
            print(f"- 总匹配数: {len(matched_attributes)} 次")

            # 显示最常见的属性
            top_attributes = matched_attributes['命中属性词'].value_counts().head(10)
            print(f"\n最常见的10个属性:")
            for attr, count in top_attributes.items():
                print(f"  {attr}: {count} 次")

            # 显示匹配率
            match_rate = (unique_comments / len(scored_comments)) * 100
            print(f"\n评论匹配率: {match_rate:.2f}%")
        else:
            print("警告: 没有找到任何属性匹配")

    except Exception as e:
        print(f"生成评论属性匹配文件失败: {e}")
        import traceback
        traceback.print_exc()


def analyze_matching_quality(matched_file: str):
    """分析匹配质量"""
    try:
        matched_df = pd.read_csv(matched_file, encoding='utf-8-sig')

        print(f"\n匹配质量分析:")
        print(f"总匹配记录: {len(matched_df)}")

        # 分析属性词长度分布
        matched_df['属性词长度'] = matched_df['命中属性词'].str.len()
        length_stats = matched_df['属性词长度'].describe()
        print(f"\n属性词长度统计:")
        print(f"平均长度: {length_stats['mean']:.2f}")
        print(f"最短: {length_stats['min']}")
        print(f"最长: {length_stats['max']}")

        # 分析重复匹配
        duplicate_matches = matched_df.duplicated(subset=['评论ID', '命中属性词']).sum()
        if duplicate_matches > 0:
            print(f"重复匹配: {duplicate_matches} 条")

    except Exception as e:
        print(f"分析匹配质量失败: {e}")


if __name__ == "__main__":
    # 生成评论属性匹配文件
    generate_comment_attributes()

    # 如果生成了文件，分析匹配质量
    if os.path.exists("comment_attributes.csv"):
        analyze_matching_quality("comment_attributes.csv")