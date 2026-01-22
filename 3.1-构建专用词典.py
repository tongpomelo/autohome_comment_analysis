import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from gensim.models import Word2Vec
from collections import Counter , defaultdict
import re
import os
import json
from datetime import datetime
import logging
from typing import List, Set, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class AutomotiveDictionaryBuilder:
    def __init__(self, excel_path, stopwords_paths, custom_dict_paths,
                 automotive_stopwords_path, output_dir=None):
        """
        初始化汽车词典构建器

        参数:
        excel_path: Excel数据文件路径
        stopwords_paths: 停用词文件路径列表
        custom_dict_paths: 自定义词典文件路径列表
        automotive_stopwords_path: 汽车专用停用词文件路径
        output_dir: 输出目录,默认为当前目录下的output文件夹
        """
        self.excel_path = excel_path
        self.stopwords_paths = stopwords_paths
        self.custom_dict_paths = custom_dict_paths
        self.automotive_stopwords_path = automotive_stopwords_path

        # 设置输出目录
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化日志
        self._init_logging()

        # 迭代历史记录
        self.iteration_history = []

        # 加载数据
        self.data = self.load_data()

        # 加载词典
        self.stopwords = self.load_stopwords()
        self.custom_dict = self.load_custom_dict()
        self.automotive_stopwords = self.load_automotive_stopwords()

        # 初始化jieba
        self.init_jieba()

    def _init_logging(self):
        """初始化日志系统"""
        log_file = os.path.join(self.output_dir,
                                f'dict_building_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
            df = pd.read_excel(self.excel_path, sheet_name='clean_data_xlsx')
            self.logger.info(f"成功加载数据,共{len(df)}条记录")
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return None

    def load_stopwords(self):
        """加载停用词"""
        stopwords = set()
        for path in self.stopwords_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    words = {line.strip() for line in f if line.strip()}
                    stopwords.update(words)
                self.logger.info(f"从 {os.path.basename(path)} 加载 {len(words)} 个停用词")
            except Exception as e:
                self.logger.warning(f"加载停用词失败 {path}: {e}")
        return stopwords

    def load_custom_dict(self):
        """加载自定义词典"""
        custom_dict = set()
        for path in self.custom_dict_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    words = {line.strip() for line in f if line.strip()}
                    custom_dict.update(words)
                self.logger.info(f"从 {os.path.basename(path)} 加载 {len(words)} 个自定义词")
            except Exception as e:
                self.logger.warning(f"加载自定义词典失败 {path}: {e}")
        return custom_dict

    def load_automotive_stopwords(self):
        """加载汽车专用停用词"""
        automotive_stopwords = set()
        try:
            with open(self.automotive_stopwords_path, 'r', encoding='utf-8') as f:
                automotive_stopwords = {line.strip() for line in f if line.strip()}
            self.logger.info(f"加载 {len(automotive_stopwords)} 个汽车专用停用词")
        except Exception as e:
            self.logger.warning(f"加载汽车专用停用词失败: {e}")
        return automotive_stopwords

    def init_jieba(self):
        """初始化jieba分词器"""
        for word in self.custom_dict:
            jieba.add_word(word)
        self.logger.info(f"已向jieba添加 {len(self.custom_dict)} 个自定义词汇")

    def extract_comments(self):
        """提取所有评论文本"""
        comment_columns = [
            '最满意', '最不满意', '空间评论', '驾驶感受评论',
            '续航评论', '外观评论', '内饰评论', '性价比评论', '智能化评论'
        ]

        all_comments = []
        for col in comment_columns:
            if col in self.data.columns:
                comments = self.data[col].dropna().astype(str).tolist()
                all_comments.extend(comments)

        self.logger.info(f"提取到 {len(all_comments)} 条评论")
        return all_comments

    def clean_text(self, text):
        """清洗文本"""
        # 去除特殊字符和标点,保留中英文和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def segment_comments(self, comments, current_dict=None):
        """使用当前词典进行分词"""
        segmented_comments = []

        for comment in comments:
            clean_comment = self.clean_text(comment)
            if not clean_comment:
                continue

            # 使用jieba分词
            words = jieba.cut(clean_comment)
            # 过滤停用词和单字
            filtered_words = [
                word for word in words
                if (word not in self.stopwords and
                    word not in self.automotive_stopwords and
                    len(word) > 1)
            ]

            if filtered_words:
                segmented_comments.append(filtered_words)

        return segmented_comments

    def train_word2vec(self, segmented_comments):
        """训练Word2Vec模型"""
        self.logger.info("训练Word2Vec模型...")
        model = Word2Vec(
            sentences=segmented_comments,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=10
        )
        return model

    def find_similar_words(self, model, target_word, topn=10, threshold=0.6):
        """查找相似词"""
        try:
            similar_words = model.wv.most_similar(target_word, topn=topn)
            return [word for word, score in similar_words if score > threshold]
        except KeyError:
            return []

    def calculate_pmi(self, bigram_freq, word_freq, total_words):
        """计算点互信息(PMI)用于评估词组合的紧密度"""
        pmi_scores = {}
        for bigram, freq in bigram_freq.items():
            word1, word2 = bigram
            p_xy = freq / total_words
            p_x = word_freq[word1] / total_words
            p_y = word_freq[word2] / total_words

            if p_x > 0 and p_y > 0:
                pmi = np.log2(p_xy / (p_x * p_y))
                pmi_scores[bigram] = pmi

        return pmi_scores

    def find_mis_segmented_words(self, segmented_comments, current_dict,
                                 top_n=50, use_pmi=True):
        """发现误切分的专业表达"""
        # 统计词频
        word_freq = Counter()
        for comment in segmented_comments:
            word_freq.update(comment)

        # 统计n-gram
        bigram_freq = Counter()
        trigram_freq = Counter()

        for comment in segmented_comments:
            # 2-gram
            for i in range(len(comment) - 1):
                bigram = comment[i] + comment[i + 1]
                bigram_freq[bigram] += 1

            # 3-gram
            for i in range(len(comment) - 2):
                trigram = comment[i] + comment[i + 1] + comment[i + 2]
                trigram_freq[trigram] += 1

        mis_segmented = []

        # 检查2-gram (更严格的筛选)
        for bigram, freq in bigram_freq.most_common(top_n * 2):
            if (freq >= 5 and  # 提高频率阈值
                    bigram not in current_dict and
                    4 <= len(bigram) <= 10 and  # 长度范围限制
                    not any(char.isdigit() for char in bigram)):  # 排除包含数字的
                mis_segmented.append((bigram, freq, 'bigram'))

        # 检查3-gram
        for trigram, freq in trigram_freq.most_common(top_n):
            if (freq >= 3 and
                    trigram not in current_dict and
                    6 <= len(trigram) <= 15 and
                    not any(char.isdigit() for char in trigram)):
                mis_segmented.append((trigram, freq, 'trigram'))

        # 按频率排序
        mis_segmented.sort(key=lambda x: x[1], reverse=True)
        return mis_segmented[:top_n]

    def filter_quality_words(self, words_with_freq: List[Tuple[str, int]],
                             current_dict: Set[str]) -> List[Tuple[str, int]]:
        """过滤低质量词汇"""
        filtered = []

        for word, freq in words_with_freq:
            # 基本过滤规则
            if (len(word) < 2 or  # 太短
                    len(word) > 20 or  # 太长
                    word in current_dict or  # 已存在
                    word.isdigit() or  # 纯数字
                    any(char in word for char in ['xx', 'XX', '***'])):  # 占位符
                continue

            # 检查是否包含有意义的汉字
            chinese_chars = sum(1 for c in word if '\u4e00' <= c <= '\u9fa5')
            if chinese_chars < len(word) * 0.5:  # 至少50%是汉字
                continue

            filtered.append((word, freq))

        return filtered

    def evaluate_iteration_quality(self, supplement_words: Set[str],
                                   word_freq: Counter) -> Dict:
        """评估本轮迭代的词典质量"""
        metrics = {
            'total_words': len(supplement_words),
            'avg_length': sum(len(w) for w in supplement_words) / len(supplement_words) if supplement_words else 0,
            'avg_freq': sum(word_freq.get(w, 0) for w in supplement_words) / len(
                supplement_words) if supplement_words else 0,
            'chinese_ratio': sum(1 for w in supplement_words if all('\u4e00' <= c <= '\u9fa5' for c in w)) / len(
                supplement_words) if supplement_words else 0
        }

        return metrics

    def check_convergence(self, iteration: int, supplement_words: Set[str],
                          history: List[Dict]) -> Tuple[bool, str]:
        """检查是否收敛"""
        if not supplement_words:
            return True, "补充词为空"

        if len(supplement_words) < 5:
            return True, f"补充词过少({len(supplement_words)}个)"

        # 检查连续两轮的增长率
        if len(history) >= 2:
            current_count = len(supplement_words)
            prev_count = history[-1]['metrics']['total_words']

            if prev_count > 0:
                growth_rate = (current_count - prev_count) / prev_count
                if abs(growth_rate) < 0.1:  # 增长率小于10%
                    return True, f"增长率过低({growth_rate:.2%})"

        return False, ""

    def iterative_dictionary_building(self, max_iterations=5,
                                      min_freq_threshold=3,
                                      similarity_threshold=0.65):
        """迭代构建专业词典"""
        comments = self.extract_comments()
        current_dict = set(self.custom_dict)

        self.logger.info(f"开始迭代构建,初始词典: {len(current_dict)} 词")
        self.logger.info(
            f"参数设置: max_iter={max_iterations}, min_freq={min_freq_threshold}, sim_threshold={similarity_threshold}")

        for iteration in range(max_iterations):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"第 {iteration + 1}/{max_iterations} 轮迭代")
            self.logger.info(f"{'=' * 60}")

            # 分词
            segmented_comments = self.segment_comments(comments, current_dict)

            if not segmented_comments:
                self.logger.warning("没有可分词的评论,结束迭代")
                break

            # 训练Word2Vec
            model = self.train_word2vec(segmented_comments)

            # 统计词频
            all_words = []
            for comment in segmented_comments:
                all_words.extend(comment)
            word_freq = Counter(all_words)

            # 1. 提取高频词
            high_freq_words = [
                (word, freq) for word, freq in word_freq.most_common(200)
                if word not in current_dict and freq >= min_freq_threshold
            ]
            high_freq_words = self.filter_quality_words(high_freq_words, current_dict)
            self.logger.info(f"发现高频词: {len(high_freq_words)} 个")

            # 2. 发现误切分表达
            mis_segmented = self.find_mis_segmented_words(segmented_comments, current_dict)
            self.logger.info(f"发现误切分表达: {len(mis_segmented)} 个")

            # 3. 查找近义词
            similar_words = []
            for word, freq in high_freq_words[:30]:
                similars = self.find_similar_words(model, word, threshold=similarity_threshold)
                for similar in similars:
                    if similar not in current_dict:
                        similar_words.append((similar, word_freq.get(similar, 0)))

            similar_words = self.filter_quality_words(similar_words, current_dict)
            self.logger.info(f"发现近义词: {len(similar_words)} 个")

            # 构建补充词集合
            supplement_words = set()

            # 添加各类词汇
            for word, freq, _ in mis_segmented:
                supplement_words.add(word)

            for word, freq in high_freq_words[:50]:  # 只取前50个高频词
                supplement_words.add(word)

            for word, freq in similar_words[:30]:  # 只取前30个近义词
                supplement_words.add(word)

            # 评估质量
            metrics = self.evaluate_iteration_quality(supplement_words, word_freq)

            # 检查收敛
            converged, reason = self.check_convergence(iteration, supplement_words,
                                                       self.iteration_history)

            # 记录历史
            self.iteration_history.append({
                'iteration': iteration + 1,
                'supplement_words': list(supplement_words),
                'metrics': metrics,
                'dict_size': len(current_dict) + len(supplement_words)
            })

            self.logger.info(f"本轮补充: {len(supplement_words)} 词")
            self.logger.info(f"质量指标: 平均长度={metrics['avg_length']:.2f}, "
                             f"平均频次={metrics['avg_freq']:.2f}, "
                             f"纯中文比例={metrics['chinese_ratio']:.2%}")

            # 显示示例
            if supplement_words:
                examples = sorted(supplement_words,
                                  key=lambda w: word_freq.get(w, 0),
                                  reverse=True)[:20]
                self.logger.info(f"补充词示例: {', '.join(examples[:10])}")

            if converged:
                self.logger.info(f"迭代收敛: {reason}")
                break

            # 更新词典
            current_dict.update(supplement_words)
            for word in supplement_words:
                jieba.add_word(word)

            self.logger.info(f"词典规模: {len(current_dict)} 词")

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"迭代完成,最终词典: {len(current_dict)} 词")
        self.logger.info(f"{'=' * 60}")

        return current_dict

    def save_dictionary(self, dictionary, output_path):
        """保存词典到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for word in sorted(dictionary, key=lambda x: len(x), reverse=True):
                    f.write(word + '\n')
            self.logger.info(f"词典已保存: {output_path}")
        except Exception as e:
            self.logger.error(f"保存词典失败: {e}")

    def save_iteration_report(self):
        """保存迭代报告"""
        report_path = os.path.join(self.output_dir, 'iteration_report.json')

        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'initial_dict_size': len(self.custom_dict),
            'iterations': self.iteration_history
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"迭代报告已保存: {report_path}")

    def visualize_iteration_process(self):
        """可视化迭代过程"""
        if not self.iteration_history:
            return

        iterations = [h['iteration'] for h in self.iteration_history]
        dict_sizes = [h['dict_size'] for h in self.iteration_history]
        supplement_counts = [h['metrics']['total_words'] for h in self.iteration_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 词典规模变化
        ax1.plot(iterations, dict_sizes, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('迭代轮次', fontsize=12)
        ax1.set_ylabel('词典规模', fontsize=12)
        ax1.set_title('词典规模增长曲线', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 每轮补充词数量
        ax2.bar(iterations, supplement_counts, color='steelblue', alpha=0.7)
        ax2.set_xlabel('迭代轮次', fontsize=12)
        ax2.set_ylabel('补充词数量', fontsize=12)
        ax2.set_title('每轮补充词数量', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, 'iteration_process.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"可视化图表已保存: {plot_path}")
        plt.close()


def main():
    # 文件路径配置
    excel_path = r"10.汽车口碑数据\merged_data_V2.xlsx"
    stopwords_paths = [r"20.停用词词库\stopwords_cn.txt",r"20.停用词词库\stopwords_full.txt"]
    custom_dict_paths = [r"30.自定义词典\汽车词库1.txt"]
    automotive_stopwords_path = r"30.自定义词典\汽车词库2.txt"

    # 输出目录
    output_dir = r"40.词典构建结果"

    # 创建词典构建器
    builder = AutomotiveDictionaryBuilder(
        excel_path=excel_path,
        stopwords_paths=stopwords_paths,
        custom_dict_paths=custom_dict_paths,
        automotive_stopwords_path=automotive_stopwords_path,
        output_dir=output_dir
    )

    # 执行迭代构建
    if builder.data is not None:
        final_dictionary = builder.iterative_dictionary_building(
            max_iterations=5,
            min_freq_threshold=3,
            similarity_threshold=0.65
        )

        # 保存最终词典
        output_dict_path = os.path.join(output_dir, '汽车专业词典_最终版.txt')
        builder.save_dictionary(final_dictionary, output_dict_path)

        # 保存迭代报告
        builder.save_iteration_report()

        # 生成可视化
        builder.visualize_iteration_process()

        # 统计信息
        print(f"\n{'=' * 60}")
        print(f"词典构建完成!")
        print(f"{'=' * 60}")
        print(f"初始词典: {len(builder.custom_dict)} 词")
        print(f"最终词典: {len(final_dictionary)} 词")
        print(f"新增词汇: {len(final_dictionary - builder.custom_dict)} 词")
        print(f"迭代轮数: {len(builder.iteration_history)} 轮")
        print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()