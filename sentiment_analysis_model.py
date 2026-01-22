import pandas as pd
import numpy as np
import jieba
import re
import os
import json
from datetime import datetime
import logging
from typing import List, Set, Tuple, Dict, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


class CommentProcessor:
    def __init__(self, professional_dict_path: str, sentiment_dict_paths: Dict[str, str],
                 degree_words_path: str, output_dir: str = None):
        """
        评论处理器初始化 - 基于HowNet情感计算模型

        参数:
        professional_dict_path: 专业词典路径
        sentiment_dict_paths: 情感词典路径字典 {'positive': path, 'negative': path, ...}
        degree_words_path: 程度级别词路径
        output_dir: 输出目录
        """
        # 设置输出目录
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'comment_processing_output')
        os.makedirs(self.output_dir, exist_ok=True)

        # 先初始化日志系统
        self._init_logging()

        # 然后加载词典
        self.professional_dict = self.load_dictionary(professional_dict_path)
        self.sentiment_dict = self.load_sentiment_dictionaries(sentiment_dict_paths)
        self.degree_words = self.load_degree_words(degree_words_path)

        self.logger.info(f"加载专业词典: {len(self.professional_dict)} 词")
        self.logger.info(f"加载情感词典: {len(self.sentiment_dict['positive'])} 正面词, "
                         f"{len(self.sentiment_dict['negative'])} 负面词, "
                         f"{len(self.sentiment_dict['perception'])} 主张词")
        self.logger.info(f"加载程度级别词: {len(self.degree_words)} 词")

    def _init_logging(self):
        """初始化日志系统"""
        log_file = os.path.join(self.output_dir,
                                f'comment_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_dictionary(self, file_path: str) -> Set[str]:
        """加载词典文件，尝试多种编码"""
        dictionary = set()
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']  # 尝试多种编码

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            dictionary.add(word)
                self.logger.info(f"成功加载词典 {file_path}，使用编码: {encoding}")
                return dictionary
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"加载词典失败 {file_path}，使用编码{encoding}: {e}")
                continue

        # 如果所有编码都失败
        self.logger.error(f"所有编码方式都失败，无法加载词典: {file_path}")
        return set()

    def load_sentiment_dictionaries(self, file_paths: Dict[str, str]) -> Dict[str, Set[str]]:
        """
        加载情感词典 - 基于HowNet分类
        """
        sentiment_dict = {
            'positive': set(),
            'negative': set(),
            'perception': set(),  # 主张/感知词
            'positive_eval': set(),  # 正面评价词
            'negative_eval': set()  # 负面评价词
        }

        # 加载正面情感词
        if 'positive' in file_paths:
            sentiment_dict['positive'] = self.load_dictionary(file_paths['positive'])

        # 加载负面情感词
        if 'negative' in file_paths:
            sentiment_dict['negative'] = self.load_dictionary(file_paths['negative'])

        # 加载主张词
        if 'perception' in file_paths:
            sentiment_dict['perception'] = self.load_dictionary(file_paths['perception'])

        # 加载正面评价词
        if 'positive_eval' in file_paths:
            sentiment_dict['positive_eval'] = self.load_dictionary(file_paths['positive_eval'])

        # 加载负面评价词
        if 'negative_eval' in file_paths:
            sentiment_dict['negative_eval'] = self.load_dictionary(file_paths['negative_eval'])

        return sentiment_dict

    def load_degree_words(self, file_path: str) -> Dict[str, float]:
        """
        加载程度级别词及其权重 - 基于HowNet程度级别分类
        修正版本：正确解析带标题和分类的文件格式
        """
        degree_words = {}
        degree_levels = {
            'extreme': 2.0,  # 最/极/非常
            'very': 1.5,  # 很/非常
            'more': 1.2,  # 较/更
            'ish': 0.8,  # 稍/有点
            'insufficiently': 0.5,  # 欠/不够
            'over': 1.8  # 过/太
        }

        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                    # 根据您提供的清晰文本结构进行解析
                    sections = content.split('\n\n')  # 按空行分割不同级别

                    for section in sections:
                        lines = section.strip().split('\n')
                        if not lines:
                            continue

                        # 解析级别标题行
                        title_line = lines[0]
                        current_level = None

                        # 检测程度级别
                        if 'extreme' in title_line or '最' in title_line:
                            current_level = 'extreme'
                        elif 'very' in title_line or '很' in title_line:
                            current_level = 'very'
                        elif 'more' in title_line or '较' in title_line:
                            current_level = 'more'
                        elif 'ish' in title_line or '稍' in title_line:
                            current_level = 'ish'
                        elif 'insufficiently' in title_line or '欠' in title_line:
                            current_level = 'insufficiently'
                        elif 'over' in title_line or '超' in title_line:
                            current_level = 'over'

                        if current_level:
                            # 处理该级别下的所有词语行
                            for line in lines[1:]:  # 跳过标题行
                                word = line.strip()
                                if word and not word.isdigit():  # 过滤数字和空行
                                    degree_words[word] = degree_levels[current_level]

                self.logger.info(f"成功加载程度级别词 {file_path}，使用编码: {encoding}")
                self.logger.info(f"加载程度词统计: {len(degree_words)} 个词")
                return degree_words

            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"加载程度级别词失败 {file_path}，使用编码{encoding}: {e}")
                continue

        # 如果文件加载失败，使用基于您提供文本的硬编码版本
        self.logger.warning("文件加载失败，使用内置程度词字典")
        return self.get_complete_degree_words_from_text()

    def get_complete_degree_words_from_text(self) -> Dict[str, float]:
        """
        基于您提供的清晰文本构建完整程度词字典
        """
        degree_words = {}
        degree_levels = {
            'extreme': 2.0,  # 最/极/非常
            'very': 1.5,  # 很/非常
            'more': 1.2,  # 较/更
            'ish': 0.8,  # 稍/有点
            'insufficiently': 0.5,  # 欠/不够
            'over': 1.8  # 过/太
        }

        # extreme 级别词语
        extreme_words = [
            '百分之百', '倍加', '备至', '不得了', '不堪', '不可开交', '不亦乐乎', '不折不扣',
            '彻头彻尾', '充分', '到头', '地地道道', '非常', '极', '极度', '极端', '极其',
            '极为', '截然', '尽', '惊人地', '绝', '绝顶', '绝对', '绝对化', '刻骨', '酷',
            '满', '满贯', '满心', '莫大', '奇', '入骨', '甚为', '十二分', '十分', '十足',
            '死', '滔天', '痛', '透', '完全', '完完全全', '万', '万般', '万分', '万万',
            '无比', '无度', '无可估量', '无以复加', '无以伦比', '要命', '要死', '已极',
            '已甚', '异常', '逾常', '贼', '之极', '之至', '至极', '卓绝', '最为', '佼佼',
            '郅', '綦', '齁', '最'
        ]

        # very 级别词语
        very_words = [
            '不过', '不少', '不胜', '惨', '沉', '沉沉', '出奇', '大为', '多', '多多', '多加',
            '多么', '分外', '格外', '够瞧的', '够戗', '好', '好不', '何等', '很', '很是',
            '坏', '可', '老', '老大', '良', '颇', '颇为', '甚', '实在', '太', '太甚', '特',
            '特别', '尤', '尤其', '尤为', '尤以', '远', '着实', '曷', '碜'
        ]

        # more 级别词语
        more_words = [
            '大不了', '多', '更', '更加', '更进一步', '更为', '还', '还要', '较', '较比',
            '较为', '进一步', '那般', '那么', '那样', '强', '如斯', '益', '益发', '尤甚',
            '逾', '愈', '愈 ... 愈', '愈发', '愈加', '愈来愈', '愈益', '远远', '越 ... 越',
            '越发', '越加', '越来越', '越是', '这般', '这样', '足', '足足'
        ]

        # ish 级别词语
        ish_words = [
            '点点滴滴', '多多少少', '怪', '好生', '还', '或多或少', '略', '略加', '略略',
            '略微', '略为', '蛮', '稍', '稍稍', '稍微', '稍为', '稍许', '挺', '未免', '相当',
            '些', '些微', '些小', '一点', '一点儿', '一些', '有点', '有点儿', '有些'
        ]

        # insufficiently 级别词语
        insufficiently_words = [
            '半点', '不大', '不丁点儿', '不甚', '不怎么', '聊', '没怎么', '轻度', '弱', '丝毫', '微', '相对'
        ]

        # over 级别词语
        over_words = [
            '不为过', '超', '超额', '超外差', '超微结构', '超物质', '出头', '多', '浮', '过',
            '过度', '过分', '过火', '过劲', '过了头', '过猛', '过热', '过甚', '过头', '过于',
            '过逾', '何止', '何啻', '开外', '苦', '老', '偏', '强', '溢', '忒'
        ]

        # 构建完整字典
        for word in extreme_words:
            degree_words[word] = degree_levels['extreme']
        for word in very_words:
            degree_words[word] = degree_levels['very']
        for word in more_words:
            degree_words[word] = degree_levels['more']
        for word in ish_words:
            degree_words[word] = degree_levels['ish']
        for word in insufficiently_words:
            degree_words[word] = degree_levels['insufficiently']
        for word in over_words:
            degree_words[word] = degree_levels['over']

        self.logger.info(f"使用内置程度词字典，共 {len(degree_words)} 个词")
        return degree_words

    def get_default_degree_words(self) -> Dict[str, float]:
        """获取默认程度级别词"""
        return {
            '最': 2.0, '极': 2.0, '非常': 2.0, '十分': 2.0, '超级': 2.0,
            '很': 1.5, '挺': 1.5, '相当': 1.5, '特别': 1.5,
            '较': 1.2, '比较': 1.2, '更加': 1.2, '更为': 1.2,
            '稍': 0.8, '有点': 0.8, '有些': 0.8, '略微': 0.8,
            '欠': 0.5, '不够': 0.5, '不太': 0.5, '略欠': 0.5,
            '过': 1.8, '太': 1.8, '过于': 1.8, '过分': 1.8
        }

    # 其余方法保持不变...
    def calculate_sentiment_score(self, sentence: str) -> float:
        """
        基于HowNet公式计算情感得分: Score = A × ES × L

        参数:
        sentence: 单个句子

        返回:
        情感得分
        """
        # A: 判断是否含有专业词 (1为有，0为没有)
        A = 1.0 if any(word in sentence for word in self.professional_dict) else 0.0

        if A == 0:
            return 0.0  # 不含专业词，直接返回0分

        # ES: 情感极性 (1为正面，-1为负面，0为中性)
        positive_count = sum(1 for word in self.sentiment_dict['positive'] |
                             self.sentiment_dict['positive_eval'] if word in sentence)
        negative_count = sum(1 for word in self.sentiment_dict['negative'] |
                             self.sentiment_dict['negative_eval'] if word in sentence)

        if positive_count > negative_count:
            ES = 1.0
        elif negative_count > positive_count:
            ES = -1.0
        else:
            ES = 0.0

        # L: 程度级别词权重 (默认为1.0)
        L = 1.0
        for degree_word, weight in self.degree_words.items():
            if degree_word in sentence:
                L = weight
                break

        # 计算最终得分: Score = A × ES × L
        score = A * ES * L

        # 记录计算详情用于调试
        self.logger.debug(f"句子: {sentence}")
        self.logger.debug(f"A(专业词): {A}, ES(情感极性): {ES}, L(程度权重): {L}, 得分: {score}")

        return score

    def segment_sentences(self, comments: List[str]) -> List[str]:
        """
        将评论分割成短句
        """
        sentences = []
        for comment in comments:
            # 使用标点符号分割句子
            split_sentences = re.split(r'[，。！？；,\.!?;]', comment)
            for sentence in split_sentences:
                cleaned_sentence = sentence.strip()
                if len(cleaned_sentence) >= 2:  # 过滤掉太短的句子
                    sentences.append(cleaned_sentence)

        self.logger.info(f"将评论分割为 {len(sentences)} 个短句")
        return sentences

    def check_usefulness(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        筛选有用评论 - 基于专业词和情感词
        """
        useful_sentences = []
        useless_sentences = []

        for sentence in sentences:
            has_professional_word = any(word in sentence for word in self.professional_dict)
            has_sentiment_word = any(word in sentence for word in
                                     self.sentiment_dict['positive'] | self.sentiment_dict['negative'] |
                                     self.sentiment_dict['positive_eval'] | self.sentiment_dict['negative_eval'])

            if has_professional_word and has_sentiment_word:
                useful_sentences.append(sentence)
            else:
                useless_sentences.append(sentence)

        self.logger.info(f"有用评论: {len(useful_sentences)} 条")
        self.logger.info(f"无用评论: {len(useless_sentences)} 条")

        return useful_sentences, useless_sentences

    def batch_sentiment_scoring(self, useful_sentences: List[str]) -> List[Tuple[str, float]]:
        """
        批量计算情感得分
        """
        scored_sentences = []

        for sentence in useful_sentences:
            score = self.calculate_sentiment_score(sentence)
            scored_sentences.append((sentence, score))

        # 统计得分分布
        positive_scores = [score for _, score in scored_sentences if score > 0]
        negative_scores = [score for _, score in scored_sentences if score < 0]
        neutral_scores = [score for _, score in scored_sentences if score == 0]

        self.logger.info(f"情感得分分布 - 正面: {len(positive_scores)}, "
                         f"负面: {len(negative_scores)}, "
                         f"中立: {len(neutral_scores)}")

        return scored_sentences

    def extract_product_attributes(self, scored_sentences: List[Tuple[str, float]]) -> Dict[
        str, List[Tuple[str, float]]]:
        """
        提取产品属性及其情感得分
        """
        attribute_scores = defaultdict(list)

        for sentence, score in scored_sentences:
            if score == 0:  # 跳过中性评论
                continue

            # 找出句子中出现的专业词（产品属性）
            found_attributes = [word for word in self.professional_dict if word in sentence]

            for attribute in found_attributes:
                attribute_scores[attribute].append((sentence, score))

        self.logger.info(f"提取到 {len(attribute_scores)} 个产品属性的情感评价")

        return dict(attribute_scores)

    def calculate_attribute_weights(self, attribute_scores: Dict[str, List[Tuple[str, float]]]) -> Dict[
        str, Dict[str, Any]]:
        """
        计算产品属性权重
        """
        attribute_stats = {}

        for attribute, scores in attribute_scores.items():
            if not scores:
                continue

            score_values = [score for _, score in scores]
            total_score = sum(score_values)
            frequency = len(scores)
            avg_score = total_score / frequency if frequency > 0 else 0

            attribute_stats[attribute] = {
                'frequency': frequency,
                'total_score': total_score,
                'average_score': avg_score,
                'score_distribution': {
                    'positive': len([s for s in score_values if s > 0]),
                    'negative': len([s for s in score_values if s < 0]),
                    'neutral': len([s for s in score_values if s == 0])
                }
            }

        # 按频率排序
        sorted_attributes = sorted(attribute_stats.items(),
                                   key=lambda x: x[1]['frequency'], reverse=True)

        self.logger.info("产品属性权重计算完成")

        return dict(sorted_attributes[:50])  # 返回前50个最重要的属性

    def save_results(self, useful_sentences: List[str],
                     scored_sentences: List[Tuple[str, float]],
                     attribute_weights: Dict[str, Dict[str, Any]]):
        """保存处理结果"""

        # 保存有用评论
        useful_file = os.path.join(self.output_dir, 'useful_comments.txt')
        with open(useful_file, 'w', encoding='utf-8') as f:
            for sentence in useful_sentences:
                f.write(sentence + '\n')

        # 保存情感得分结果
        scored_file = os.path.join(self.output_dir, 'scored_comments.csv')
        df_scored = pd.DataFrame(scored_sentences, columns=['comment', 'sentiment_score'])
        df_scored.to_csv(scored_file, index=False, encoding='utf-8-sig')

        # 保存属性权重
        weights_file = os.path.join(self.output_dir, 'attribute_weights.json')
        with open(weights_file, 'w', encoding='utf-8') as f:
            json.dump(attribute_weights, f, ensure_ascii=False, indent=2)

        # 生成属性权重可视化
        self.visualize_attribute_weights(attribute_weights)

        # 生成情感得分分布图
        self.visualize_sentiment_distribution(scored_sentences)

        self.logger.info(f"所有结果已保存到: {self.output_dir}")

    def visualize_attribute_weights(self, attribute_weights: Dict[str, Dict[str, Any]]):
        """可视化属性权重"""
        if not attribute_weights:
            return

        # 提取前20个属性进行可视化
        top_attributes = list(attribute_weights.keys())[:20]
        frequencies = [attribute_weights[attr]['frequency'] for attr in top_attributes]
        avg_scores = [attribute_weights[attr]['average_score'] for attr in top_attributes]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 频率条形图
        bars = ax1.barh(top_attributes, frequencies, color='skyblue')
        ax1.set_xlabel('出现频率')
        ax1.set_title('产品属性出现频率TOP20')
        ax1.grid(True, alpha=0.3, axis='x')

        # 在条形上添加数值
        for bar, freq in zip(bars, frequencies):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     str(freq), ha='left', va='center')

        # 平均得分条形图
        colors = ['green' if score > 0 else 'red' for score in avg_scores]
        bars2 = ax2.barh(top_attributes, avg_scores, color=colors, alpha=0.7)
        ax2.set_xlabel('平均情感得分')
        ax2.set_title('产品属性平均情感得分TOP20')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # 在条形上添加数值
        for bar, score in zip(bars2, avg_scores):
            ax2.text(bar.get_width() + (0.1 if score >= 0 else -0.3),
                     bar.get_y() + bar.get_height() / 2,
                     f'{score:.2f}', ha='left' if score >= 0 else 'right', va='center')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'attribute_weights_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"属性权重可视化图表已保存: {plot_path}")

    def visualize_sentiment_distribution(self, scored_sentences: List[Tuple[str, float]]):
        """可视化情感得分分布"""
        scores = [score for _, score in scored_sentences]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 情感得分分布直方图
        ax1.hist(scores, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('情感得分')
        ax1.set_ylabel('频数')
        ax1.set_title('情感得分分布直方图')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # 情感极性饼图
        positive_count = len([s for s in scores if s > 0])
        negative_count = len([s for s in scores if s < 0])
        neutral_count = len([s for s in scores if s == 0])

        labels = ['正面', '负面', '中性']
        sizes = [positive_count, negative_count, neutral_count]
        colors = ['#66b3ff', '#ff6666', '#c2c2f0']

        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('情感极性分布')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'sentiment_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"情感分布可视化图表已保存: {plot_path}")

    def process_comments(self, comments: List[str]) -> Dict[str, Any]:
        """
        完整的评论处理流程

        参数:
        comments: 原始评论列表

        返回:
        处理结果汇总
        """
        self.logger.info("开始处理评论...")

        # 1. 分割句子
        sentences = self.segment_sentences(comments)

        # 2. 筛选有用评论
        useful_sentences, useless_sentences = self.check_usefulness(sentences)

        # 3. 计算情感得分
        scored_sentences = self.batch_sentiment_scoring(useful_sentences)

        # 4. 提取产品属性
        attribute_scores = self.extract_product_attributes(scored_sentences)

        # 5. 计算属性权重
        attribute_weights = self.calculate_attribute_weights(attribute_scores)

        # 6. 保存结果
        self.save_results(useful_sentences, scored_sentences, attribute_weights)

        # 返回汇总结果
        result_summary = {
            'total_comments': len(comments),
            'total_sentences': len(sentences),
            'useful_sentences': len(useful_sentences),
            'useless_sentences': len(useless_sentences),
            'sentiment_distribution': {
                'positive': len([s for _, s in scored_sentences if s > 0]),
                'negative': len([s for _, s in scored_sentences if s < 0]),
                'neutral': len([s for _, s in scored_sentences if s == 0])
            },
            'top_attributes': attribute_weights
        }

        self.logger.info("评论处理完成!")
        return result_summary


# 使用示例
def main():
    # 文件路径配置 - 基于提供的词典文件
    professional_dict_path = "30.自定义词典/汽车专业词典_最终版.txt"  # 使用正斜杠
    sentiment_dict_paths = {
        'positive': "31.情感词典/知网Hownet情感词典-中文/正面情感词语（中文）.txt",
        'negative': "31.情感词典/知网Hownet情感词典-中文/负面情感词语（中文）.txt",
        'perception': "31.情感词典/知网Hownet情感词典-中文/主张词语（中文）.txt",
        'positive_eval': "31.情感词典/知网Hownet情感词典-中文/正面评价词语（中文）.txt",
        'negative_eval': "31.情感词典/知网Hownet情感词典-中文/负面评价词语（中文）.txt"
    }
    degree_words_path = "31.情感词典/知网Hownet情感词典-中文/程度级别词语（中文）.txt"

    # 输出目录
    output_dir = "comment_analysis_results"

    # 创建评论处理器
    processor = CommentProcessor(
        professional_dict_path=professional_dict_path,
        sentiment_dict_paths=sentiment_dict_paths,
        degree_words_path=degree_words_path,
        output_dir=output_dir
    )

    # 示例评论数据
    sample_comments = [
        "小鹏汽车的续航表现非常出色，我特别喜欢它的电池管理系统。",
        "内饰做工一般，座椅舒适度有待提高。",
        "智能驾驶功能很好用，NGP系统特别智能。",
        "外观设计很时尚，车灯造型很有科技感。",
        "后排空间有点拥挤，长途乘坐不太舒服。",
        "充电速度很快，超充站覆盖也很广。",
        "音响效果不错，隔音性能也很好。",
        "价格有点高，性价比不是很高。"
    ]

    # 处理评论
    results = processor.process_comments(sample_comments)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("评论处理结果摘要 - 基于HowNet情感计算模型")
    print("=" * 60)
    print(f"总评论数: {results['total_comments']}")
    print(f"总句子数: {results['total_sentences']}")
    print(f"有用评论: {results['useful_sentences']}")
    print(f"无用评论: {results['useless_sentences']}")
    print(f"情感分布 - 正面: {results['sentiment_distribution']['positive']}, "
          f"负面: {results['sentiment_distribution']['negative']}, "
          f"中性: {results['sentiment_distribution']['neutral']}")

    print("\nTOP 10 产品属性:")
    top_attributes = list(results['top_attributes'].items())[:10]
    for attr, stats in top_attributes:
        print(f"  {attr}: 频率={stats['frequency']}, 平均得分={stats['average_score']:.2f}")


if __name__ == "__main__":
    main()