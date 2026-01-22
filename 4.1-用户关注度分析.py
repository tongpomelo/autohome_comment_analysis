import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from collections import defaultdict
import os
import warnings
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


class AutomotiveAttentionAnalyzer:
    """
    基于三级词库的汽车用户关注度趋势分析
    """

    def __init__(self, review_data, three_level_dict_path, output_dir='50. 用户关注度分析'):
        """
        初始化分析器

        参数:
        review_data: 评论数据DataFrame
        three_level_dict_path: 三级词库Excel文件路径
        output_dir: 输出目录
        """
        self.df = review_data.copy()
        self.output_dir = output_dir

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载三级词库
        self.load_three_level_dict(three_level_dict_path)

        # 预处理数据
        self.preprocess_data()

        # 分析结果存储
        self.results = {}

    def load_three_level_dict(self, dict_path):
        """加载三级词库"""
        print("正在加载三级词库...")

        try:
            # 尝试读取第一个sheet（汽车评价思维导图）
            dict_df = pd.read_excel(dict_path, sheet_name='sheet1')

            # 检查列名
            if len(dict_df.columns) >= 3:
                dict_df.columns = ['一级分类', '二级分类', '三级分类'][:len(dict_df.columns)]
            else:
                # 尝试读取第二个sheet
                dict_df = pd.read_excel(dict_path, sheet_name=1)
                if len(dict_df.columns) >= 4:
                    dict_df = dict_df[['一级分类', '二级分类', '三级分类']]
                else:
                    print("警告：词库格式不符合预期")
                    return

        except Exception as e:
            print(f"加载词库失败: {e}")
            # 尝试读取默认sheet
            dict_df = pd.read_excel(dict_path)
            if len(dict_df.columns) >= 3:
                dict_df.columns = ['一级分类', '二级分类', '三级分类'][:len(dict_df.columns)]

        # 清理数据
        dict_df = dict_df.dropna(subset=['三级分类'])
        dict_df = dict_df[['一级分类', '二级分类', '三级分类']].copy()

        # 去除三级分类中的数字前缀和空格
        dict_df['三级分类'] = dict_df['三级分类'].astype(str).str.strip()

        self.dict_df = dict_df
        print(f"成功加载三级词库: {len(self.dict_df)} 个词条")

        # 构建多级索引
        self._build_dict_index()

    def _build_dict_index(self):
        """构建词库索引"""
        # 一级分类映射
        self.level1_categories = sorted(self.dict_df['一级分类'].dropna().unique())

        # 二级分类映射
        self.level2_categories = {}
        for level1 in self.level1_categories:
            level2_list = self.dict_df[self.dict_df['一级分类'] == level1]['二级分类'].unique()
            self.level2_categories[level1] = sorted(level2_list)

        # 三级关键词映射
        self.keyword_mapping = {}
        for _, row in self.dict_df.iterrows():
            keyword = str(row['三级分类']).strip()
            if keyword:
                self.keyword_mapping[keyword] = {
                    'level1': row['一级分类'],
                    'level2': row['二级分类']
                }

        print(f"词库构建完成: {len(self.level1_categories)} 个一级分类, "
              f"{sum(len(v) for v in self.level2_categories.values())} 个二级分类, "
              f"{len(self.keyword_mapping)} 个三级关键词")

    def preprocess_data(self):
        """预处理评论数据"""
        print("\n正在预处理评论数据...")

        # 提取文本列
        text_cols = [
            '最满意', '最不满意', '空间评论', '驾驶感受评论',
            '续航评论', '外观评论', '内饰评论', '性价比评论', '智能化评论'
        ]
        self.text_columns = [col for col in text_cols if col in self.df.columns]
        print(f"可用文本列: {len(self.text_columns)} 个")

        # 处理时间数据
        time_columns = ['购买时间']  # 只使用购买时间
        self.time_column = None
        for col in time_columns:
            if col in self.df.columns:
                self.time_column = col
                break

        if self.time_column:
            self.df[self.time_column] = pd.to_datetime(self.df[self.time_column], errors='coerce')

            # 筛选时间大于2021-01-01的数据（新能源汽车普及后）
            start_date = pd.Timestamp('2021-01-01')
            original_count = len(self.df)
            self.df = self.df[self.df[self.time_column] >= start_date].copy()
            filtered_count = len(self.df)
            print(f"时间筛选: 保留 {filtered_count}/{original_count} 条数据 (>=2021-01-01)")

            # 如果没有符合时间条件的数据，显示警告
            if filtered_count == 0:
                print("警告: 筛选后无符合时间条件的数据，将使用所有数据")
                self.df[self.time_column] = pd.to_datetime(self.df[self.time_column], errors='coerce')
                # 获取实际的时间范围
                time_range = self.df[self.time_column].dropna()
                if len(time_range) > 0:
                    print(f"实际时间范围: {time_range.min().date()} 到 {time_range.max().date()}")

            # 提取时间维度
            self.df['年份'] = self.df[self.time_column].dt.year
            self.df['季度'] = self.df[self.time_column].dt.quarter
            self.df['年份季度'] = self.df['年份'].astype(str) + '-Q' + self.df['季度'].astype(str)
            self.df['月份'] = self.df[self.time_column].dt.month

            # 统计时间分布
            valid_time_count = self.df[self.time_column].notna().sum()
            print(f"有效时间数据: {valid_time_count}/{len(self.df)} 条")

            if valid_time_count > 0:
                time_range = self.df[self.time_column].dropna()
                print(f"分析时间范围: {time_range.min().date()} 到 {time_range.max().date()}")
        else:
            print("警告: 未找到时间列")

    def extract_keywords_from_text(self, text):
        """从文本中提取三级关键词"""
        if pd.isna(text):
            return []

        text = str(text).lower()
        found_keywords = []

        # 按长度从长到短排序，避免短关键词匹配到长关键词的子串
        sorted_keywords = sorted(self.keyword_mapping.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text:
                # 避免重复计数
                if not any(kw in keyword_lower for kw in found_keywords if kw != keyword_lower):
                    found_keywords.append(keyword)

        return found_keywords

    def analyze_by_time_period(self, time_period='年份季度'):
        """
        按时间段分析关注度

        参数:
        time_period: 时间维度，可选 '年份', '季度', '年份季度', '月份'
        """
        if time_period not in self.df.columns:
            print(f"警告: 时间维度 '{time_period}' 不存在")
            return None

        print(f"\n开始按 {time_period} 分析关注度...")

        results = {
            'periods': [],
            'total_records': [],
            'total_keywords': [],
            'level1_attention': defaultdict(lambda: defaultdict(int)),
            'level2_attention': defaultdict(lambda: defaultdict(int)),
            'level3_attention': defaultdict(lambda: defaultdict(int)),
            # 新增：百分比数据
            'level1_percentage': defaultdict(lambda: defaultdict(float)),
            'level2_percentage': defaultdict(lambda: defaultdict(float)),
            'level3_percentage': defaultdict(lambda: defaultdict(float))
        }

        # 获取所有时间段
        periods = sorted(self.df[time_period].dropna().unique())

        for period in periods:
            period_data = self.df[self.df[time_period] == period]

            if len(period_data) == 0:
                continue

            period_keywords_count = 0
            period_keywords_list = []

            # 统计每个关键词的出现次数
            for _, row in period_data.iterrows():
                all_keywords = []

                # 从所有文本列中提取关键词
                for col in self.text_columns:
                    text = row[col]
                    if pd.isna(text):
                        continue

                    keywords = self.extract_keywords_from_text(text)
                    all_keywords.extend(keywords)

                # 去重
                unique_keywords = list(set(all_keywords))
                period_keywords_count += len(unique_keywords)
                period_keywords_list.extend(unique_keywords)

                # 统计各级分类
                for keyword in unique_keywords:
                    mapping = self.keyword_mapping.get(keyword)
                    if mapping:
                        level1 = mapping['level1']
                        level2 = mapping['level2']

                        results['level3_attention'][period][keyword] += 1
                        results['level2_attention'][period][f"{level1} - {level2}"] += 1
                        results['level1_attention'][period][level1] += 1

            results['periods'].append(period)
            results['total_records'].append(len(period_data))
            results['total_keywords'].append(period_keywords_count)

            # 计算百分比（均衡化处理）
            if period_keywords_count > 0:
                # 计算一级分类百分比
                for level1 in results['level1_attention'][period]:
                    count = results['level1_attention'][period][level1]
                    results['level1_percentage'][period][level1] = count / period_keywords_count * 100

                # 计算二级分类百分比
                for level2 in results['level2_attention'][period]:
                    count = results['level2_attention'][period][level2]
                    results['level2_percentage'][period][level2] = count / period_keywords_count * 100

                # 计算三级关键词百分比
                for keyword in results['level3_attention'][period]:
                    count = results['level3_attention'][period][keyword]
                    results['level3_percentage'][period][keyword] = count / period_keywords_count * 100

        self.results[time_period] = results
        print(f"分析完成: {len(periods)} 个时间段")
        print(f"总关键词提及次数: {sum(results['total_keywords'])}")

        # 显示各时间段关键词数量分布
        if len(periods) > 1:
            print("\n各时间段关键词数量分布:")
            for i, period in enumerate(periods):
                keywords_count = results['total_keywords'][i]
                records_count = results['total_records'][i]
                avg_keywords = keywords_count / records_count if records_count > 0 else 0
                print(
                    f"  {period}: {records_count}条评论, {keywords_count}个关键词, 平均每条{avg_keywords:.2f}个关键词")

        return results

    def save_analysis_results(self, time_period='年份季度'):
        """保存分析结果到Excel"""
        if time_period not in self.results:
            print("请先运行 analyze_by_time_period()")
            return

        results = self.results[time_period]
        output_file = os.path.join(self.output_dir, f'关注度分析_{time_period}.xlsx')

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. 时间段汇总
            summary_df = pd.DataFrame({
                '时间段': results['periods'],
                '样本数量': results['total_records'],
                '关键词总数': results['total_keywords'],
                '平均关键词数': np.array(results['total_keywords']) / np.array(results['total_records'])
            })
            summary_df.to_excel(writer, sheet_name='时间段汇总', index=False)

            # 2. 一级分类趋势（包含频次和百分比）
            level1_data = []
            for period in results['periods']:
                period_idx = results['periods'].index(period)
                total_keywords = results['total_keywords'][period_idx]

                for level1 in results['level1_attention'][period]:
                    count = results['level1_attention'][period][level1]
                    percentage = results['level1_percentage'][period].get(level1, 0)

                    level1_data.append({
                        '时间段': period,
                        '一级分类': level1,
                        '提及次数': count,
                        '占比(%)': percentage,
                        '标准化得分': count / total_keywords * 100 if total_keywords > 0 else 0
                    })

            level1_df = pd.DataFrame(level1_data)
            level1_df.to_excel(writer, sheet_name='一级分类趋势', index=False)

            # 3. 二级分类趋势（包含频次和百分比）
            level2_data = []
            for period in results['periods']:
                period_idx = results['periods'].index(period)
                total_keywords = results['total_keywords'][period_idx]

                for level2 in results['level2_attention'][period]:
                    count = results['level2_attention'][period][level2]
                    percentage = results['level2_percentage'][period].get(level2, 0)

                    level2_data.append({
                        '时间段': period,
                        '二级分类': level2,
                        '提及次数': count,
                        '占比(%)': percentage,
                        '标准化得分': count / total_keywords * 100 if total_keywords > 0 else 0
                    })

            level2_df = pd.DataFrame(level2_data)
            level2_df.to_excel(writer, sheet_name='二级分类趋势', index=False)

            # 4. 三级关键词趋势（包含频次和百分比）
            all_keywords = set()
            for period in results['periods']:
                all_keywords.update(results['level3_attention'][period].keys())

            # 计算每个关键词的总提及次数
            keyword_totals = {}
            for keyword in all_keywords:
                total = sum(results['level3_attention'][period].get(keyword, 0) for period in results['periods'])
                keyword_totals[keyword] = total

            # 取前100个最活跃的关键词
            top_keywords = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:100]
            top_keyword_list = [kw for kw, _ in top_keywords]

            level3_data = []
            for period in results['periods']:
                period_idx = results['periods'].index(period)
                total_keywords = results['total_keywords'][period_idx]

                for keyword in top_keyword_list:
                    count = results['level3_attention'][period].get(keyword, 0)
                    if count > 0:
                        mapping = self.keyword_mapping.get(keyword, {})
                        percentage = results['level3_percentage'][period].get(keyword, 0)

                        level3_data.append({
                            '时间段': period,
                            '一级分类': mapping.get('level1', ''),
                            '二级分类': mapping.get('level2', ''),
                            '三级关键词': keyword,
                            '提及次数': count,
                            '占比(%)': percentage,
                            '标准化得分': count / total_keywords * 100 if total_keywords > 0 else 0
                        })

            level3_df = pd.DataFrame(level3_data)
            level3_df.to_excel(writer, sheet_name='三级关键词趋势', index=False)

            # 5. 均衡化分析报告
            balanced_data = []
            for period in results['periods']:
                period_idx = results['periods'].index(period)
                total_keywords = results['total_keywords'][period_idx]
                total_records = results['total_records'][period_idx]

                # 计算均衡化指标
                if total_keywords > 0 and total_records > 0:
                    # 关键词密度
                    keyword_density = total_keywords / total_records

                    # 分类多样性（一级分类的数量）
                    level1_diversity = len(results['level1_attention'][period])

                    # 关键词集中度（前3个一级分类占比）
                    level1_counts = list(results['level1_attention'][period].values())
                    level1_counts.sort(reverse=True)
                    top3_concentration = sum(level1_counts[:3]) / total_keywords * 100 if len(
                        level1_counts) >= 3 else 100

                    balanced_data.append({
                        '时间段': period,
                        '样本数量': total_records,
                        '关键词总数': total_keywords,
                        '关键词密度': keyword_density,
                        '分类多样性': level1_diversity,
                        '前3分类集中度(%)': top3_concentration,
                        '均衡化指数': (keyword_density * level1_diversity) / (top3_concentration + 1) * 100
                    })

            if balanced_data:
                balanced_df = pd.DataFrame(balanced_data)
                balanced_df.to_excel(writer, sheet_name='均衡化分析', index=False)

        print(f"分析结果已保存到: {output_file}")

        return output_file

    def plot_level1_trends(self, time_period='年份季度', top_n=8, figsize=(24, 30)):
        """绘制一级分类趋势图（包含频次和百分比视图）"""
        if time_period not in self.results:
            print("请先运行 analyze_by_time_period()")
            return

        results = self.results[time_period]
        periods = results['periods']

        # 计算每个一级分类的总提及次数
        level1_totals = defaultdict(int)
        for period in periods:
            for level1, count in results['level1_attention'][period].items():
                level1_totals[level1] += count

        # 选择Top N个一级分类
        top_level1 = sorted(level1_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_level1_list = [level1 for level1, _ in top_level1]

        print(f"显示前 {top_n} 个一级分类: {top_level1_list}")

        # 准备绘图数据
        plot_data_counts = []
        plot_data_percentages = []
        for level1 in top_level1_list:
            counts = []
            percentages = []
            for period in periods:
                counts.append(results['level1_attention'][period].get(level1, 0))
                percentages.append(results['level1_percentage'][period].get(level1, 0))
            plot_data_counts.append((level1, counts))
            plot_data_percentages.append((level1, percentages))

        # 创建图形 - 使用GridSpec实现复杂布局
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 2, height_ratios=[2, 2, 2, 2], hspace=0.3, wspace=0.2)

        # 子图1: 堆叠面积图（频次）
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(periods))
        bottom = np.zeros(len(periods))

        # 创建颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data_counts)))

        for i, (level1, counts) in enumerate(plot_data_counts):
            ax1.fill_between(x, bottom, bottom + counts, label=level1, alpha=0.8, color=colors[i])
            bottom += counts

        ax1.set_title(f'一级分类关注度趋势 - 频次堆叠图 ({time_period})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间段', fontsize=12)
        ax1.set_ylabel('提及次数', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 子图2: 堆叠面积图（百分比）
        ax2 = fig.add_subplot(gs[0, 1])
        bottom = np.zeros(len(periods))

        for i, (level1, percentages) in enumerate(plot_data_percentages):
            ax2.fill_between(x, bottom, bottom + percentages, label=level1, alpha=0.8, color=colors[i])
            bottom += percentages

        ax2.set_title(f'一级分类关注度趋势 - 百分比堆叠图 ({time_period})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间段', fontsize=12)
        ax2.set_ylabel('占比(%)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])

        # 子图3: 折线图（频次）
        ax3 = fig.add_subplot(gs[1, 0])
        for i, (level1, counts) in enumerate(plot_data_counts):
            ax3.plot(x, counts, marker='o', linewidth=2, markersize=6,
                     label=level1, color=colors[i])

        ax3.set_title('一级分类关注度趋势 - 频次折线图', fontsize=14, fontweight='bold')
        ax3.set_xlabel('时间段', fontsize=12)
        ax3.set_ylabel('提及次数', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 子图4: 折线图（百分比）
        ax4 = fig.add_subplot(gs[1, 1])
        for i, (level1, percentages) in enumerate(plot_data_percentages):
            ax4.plot(x, percentages, marker='o', linewidth=2, markersize=6,
                     label=level1, color=colors[i])

        ax4.set_title('一级分类关注度趋势 - 百分比折线图', fontsize=14, fontweight='bold')
        ax4.set_xlabel('时间段', fontsize=12)
        ax4.set_ylabel('占比(%)', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, max([max(p) for _, p in plot_data_percentages]) * 1.2])

        # 子图5: 热力图（频次）
        ax5 = fig.add_subplot(gs[2, 0])
        heatmap_data_counts = []
        heatmap_labels_counts = []
        for level1, counts in plot_data_counts:
            # 只显示有数据的一级分类
            if max(counts) > 0:
                heatmap_data_counts.append(counts)
                heatmap_labels_counts.append(level1)

        if heatmap_data_counts:
            heatmap_data_counts = np.array(heatmap_data_counts)
            im1 = ax5.imshow(heatmap_data_counts, cmap='YlOrRd', aspect='auto',
                             vmin=0, vmax=np.max(heatmap_data_counts))

            # 简化标签显示
            simplified_labels = []
            for label in heatmap_labels_counts:
                simplified = label.replace('1.', '').replace('2.', '').replace('3.', '') \
                    .replace('4.', '').replace('5.', '').replace('6.', '') \
                    .replace('7.', '').replace('8.', '')
                if len(simplified) > 10:
                    simplified = simplified[:8] + '...'
                simplified_labels.append(simplified)

            ax5.set_xticks(range(len(periods)))
            ax5.set_xticklabels(periods, rotation=45, ha='right', fontsize=9)
            ax5.set_yticks(range(len(heatmap_labels_counts)))
            ax5.set_yticklabels(simplified_labels, fontsize=9)

            # 添加数值标签
            for i in range(len(heatmap_labels_counts)):
                for j in range(len(periods)):
                    value = heatmap_data_counts[i, j]
                    if value > 0:
                        ax5.text(j, i, f'{int(value)}',
                                 ha="center", va="center",
                                 color="black" if value < np.max(heatmap_data_counts) / 2 else "white",
                                 fontsize=8, fontweight='bold')

            ax5.set_title('一级分类关注度热力图 (频次)', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=ax5, label='提及次数', shrink=0.8)
        else:
            ax5.text(0.5, 0.5, '无显著数据', ha='center', va='center', fontsize=12)
            ax5.axis('off')

        # 子图6: 热力图（百分比）
        ax6 = fig.add_subplot(gs[2, 1])
        heatmap_data_percentages = []
        heatmap_labels_percentages = []
        for level1, percentages in plot_data_percentages:
            # 只显示有数据的一级分类
            if max(percentages) > 0.5:  # 设置0.5%的阈值
                heatmap_data_percentages.append(percentages)
                heatmap_labels_percentages.append(level1)

        if heatmap_data_percentages:
            heatmap_data_percentages = np.array(heatmap_data_percentages)
            im2 = ax6.imshow(heatmap_data_percentages, cmap='YlOrRd', aspect='auto',
                             vmin=0, vmax=np.max(heatmap_data_percentages))

            # 简化标签显示
            simplified_labels = []
            for label in heatmap_labels_percentages:
                simplified = label.replace('1.', '').replace('2.', '').replace('3.', '') \
                    .replace('4.', '').replace('5.', '').replace('6.', '') \
                    .replace('7.', '').replace('8.', '')
                if len(simplified) > 10:
                    simplified = simplified[:8] + '...'
                simplified_labels.append(simplified)

            ax6.set_xticks(range(len(periods)))
            ax6.set_xticklabels(periods, rotation=45, ha='right', fontsize=9)
            ax6.set_yticks(range(len(heatmap_labels_percentages)))
            ax6.set_yticklabels(simplified_labels, fontsize=9)

            # 添加数值标签
            for i in range(len(heatmap_labels_percentages)):
                for j in range(len(periods)):
                    value = heatmap_data_percentages[i, j]
                    if value > 0:
                        ax6.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center",
                                 color="black" if value < 50 else "white",
                                 fontsize=8, fontweight='bold')

            ax6.set_title('一级分类关注度热力图 (占比%)', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=ax6, label='占比(%)', shrink=0.8)
        else:
            ax6.text(0.5, 0.5, '无显著数据 (占比均 < 0.5%)', ha='center', va='center', fontsize=12)
            ax6.axis('off')

        # 子图7: 雷达图（对比各时间段）
        ax7 = fig.add_subplot(gs[3, 0], polar=True)

        # 准备雷达图数据（使用百分比数据）
        angles = np.linspace(0, 2 * np.pi, len(top_level1_list), endpoint=False).tolist()

        # 选择第一个和最后一个时间段进行对比
        if len(periods) >= 2:
            first_period = periods[0]
            last_period = periods[-1]

            first_period_data = []
            last_period_data = []

            for level1 in top_level1_list:
                first_period_data.append(results['level1_percentage'][first_period].get(level1, 0))
                last_period_data.append(results['level1_percentage'][last_period].get(level1, 0))

            # 闭合数据
            first_period_data = first_period_data + [first_period_data[0]]
            last_period_data = last_period_data + [last_period_data[0]]
            angles_plot = angles + [angles[0]]

            ax7.plot(angles_plot, first_period_data, 'o-', linewidth=2, label=first_period)
            ax7.plot(angles_plot, last_period_data, 'o-', linewidth=2, label=last_period)
            ax7.fill(angles_plot, first_period_data, alpha=0.25)
            ax7.fill(angles_plot, last_period_data, alpha=0.25)

            ax7.set_xticks(angles)
            # 简化标签显示
            radar_labels = []
            for label in top_level1_list:
                simplified = label.replace('1.', '').replace('2.', '').replace('3.', '') \
                    .replace('4.', '').replace('5.', '').replace('6.', '') \
                    .replace('7.', '').replace('8.', '')
                if len(simplified) > 8:
                    simplified = simplified[:6] + '...'
                radar_labels.append(simplified)
            ax7.set_xticklabels(radar_labels, fontsize=9)
            ax7.set_title(f'关注度分布对比雷达图\n{first_period} vs {last_period}', fontsize=12, fontweight='bold')
            ax7.legend(loc='upper right', fontsize=9)
            ax7.grid(True)
        else:
            ax7.text(0.5, 0.5, '数据不足生成雷达图\n(需要至少2个时间段)',
                     ha='center', va='center', fontsize=12, transform=ax7.transAxes)
            ax7.axis('off')

        # 子图8: 箱线图（显示各时间段分布）
        ax8 = fig.add_subplot(gs[3, 1])

        # 收集所有时间段的数据
        all_periods_data = []
        for level1 in top_level1_list:
            level_data = []
            for period in periods:
                level_data.append(results['level1_percentage'][period].get(level1, 0))
            all_periods_data.append(level_data)

        # 绘制箱线图
        bp = ax8.boxplot(all_periods_data, labels=[f'分类{i + 1}' for i in range(len(top_level1_list))])
        ax8.set_title('各一级分类占比分布箱线图', fontsize=12, fontweight='bold')
        ax8.set_xlabel('一级分类', fontsize=10)
        ax8.set_ylabel('占比(%)', fontsize=10)
        ax8.grid(True, alpha=0.3, axis='y')

        # 在箱线图上添加均值点
        means = [np.mean(data) for data in all_periods_data]
        ax8.plot(range(1, len(means) + 1), means, 'ro', label='均值')

        plt.tight_layout()

        # 保存图片，不显示
        chart_path = os.path.join(self.output_dir, f'一级分类趋势图_{time_period}.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"一级分类趋势图已保存到: {chart_path}")

        # 额外生成每个一级分类的单独趋势图（频次和百分比对比）
        self._plot_individual_level1_trends(time_period, top_level1_list)

        return top_level1_list

    def _plot_individual_level1_trends(self, time_period, top_level1_list):
        """为每个一级分类生成单独的趋势图（频次和百分比对比）"""
        if time_period not in self.results:
            return

        results = self.results[time_period]
        periods = results['periods']

        for level1 in top_level1_list:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            counts = []
            percentages = []
            for period in periods:
                counts.append(results['level1_attention'][period].get(level1, 0))
                percentages.append(results['level1_percentage'][period].get(level1, 0))

            x = range(len(periods))

            # 子图1: 频次柱状图
            bars1 = ax1.bar(x, counts, color=plt.cm.Set3(np.random.random()), alpha=0.8)
            ax1.set_title(f'{level1} - 关注度趋势 (频次) ({time_period})', fontsize=14, fontweight='bold')
            ax1.set_xlabel('时间段', fontsize=12)
            ax1.set_ylabel('提及次数', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')

            # 在柱子上添加数值
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{int(height)}', ha='center', va='bottom', fontsize=9)

            # 子图2: 百分比折线图
            ax2.plot(x, percentages, 'o-', linewidth=2, markersize=8, color='red', alpha=0.8)
            ax2.fill_between(x, 0, percentages, alpha=0.3, color='red')
            ax2.set_title(f'{level1} - 关注度趋势 (占比) ({time_period})', fontsize=14, fontweight='bold')
            ax2.set_xlabel('时间段', fontsize=12)
            ax2.set_ylabel('占比(%)', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, max(percentages) * 1.2])

            # 在折线上添加百分比数值
            for i, pct in enumerate(percentages):
                if pct > 0:
                    ax2.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()

            # 保存单独图表，不显示
            safe_name = level1.replace('/', '_').replace('\\', '_').replace(':', '_')
            chart_path = os.path.join(self.output_dir, f'单独趋势_{safe_name}_{time_period}.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"已生成 {len(top_level1_list)} 个单独的一级分类趋势图")

    def plot_level2_trends_with_radar(self, level1_category=None, time_period='年份季度', figsize=(20, 16)):
        """绘制二级分类趋势图（包含雷达图）"""
        if time_period not in self.results:
            print("请先运行 analyze_by_time_period()")
            return

        results = self.results[time_period]
        periods = results['periods']

        # 如果没有指定一级分类，选择最活跃的那个
        if level1_category is None:
            level1_totals = defaultdict(int)
            for period in periods:
                for level1, count in results['level1_attention'][period].items():
                    level1_totals[level1] += count

            if level1_totals:
                level1_category = max(level1_totals.items(), key=lambda x: x[1])[0]
            else:
                print("没有找到一级分类数据")
                return

        # 获取该一级分类下的所有二级分类
        level2_categories = self.level2_categories.get(level1_category, [])

        if not level2_categories:
            print(f"一级分类 '{level1_category}' 下没有找到二级分类")
            return

        # 准备绘图数据（频次和百分比）
        level2_data_counts = {}
        level2_data_percentages = {}
        for level2 in level2_categories:
            full_level2_name = f"{level1_category} - {level2}"
            counts = []
            percentages = []
            for period in periods:
                counts.append(results['level2_attention'][period].get(full_level2_name, 0))
                percentages.append(results['level2_percentage'][period].get(full_level2_name, 0))
            level2_data_counts[level2] = counts
            level2_data_percentages[level2] = percentages

        # 只保留有数据的二级分类
        active_level2 = {k: v for k, v in level2_data_counts.items() if sum(v) > 0}
        active_level2_percentages = {k: v for k, v in level2_data_percentages.items() if sum(v) > 0}

        if not active_level2:
            print(f"一级分类 '{level1_category}' 下的二级分类没有数据")
            return

        print(f"一级分类 '{level1_category}' 下有 {len(active_level2)} 个有数据的二级分类")

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        x = range(len(periods))
        colors = plt.cm.tab20(np.linspace(0, 1, len(active_level2)))

        # 子图1: 频次折线图
        ax1 = axes[0, 0]
        for i, (level2, counts) in enumerate(active_level2.items()):
            ax1.plot(x, counts, marker='o', linewidth=2, markersize=6,
                     label=level2, color=colors[i])

        ax1.set_title(f'{level1_category} - 二级分类关注度趋势 (频次)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间段', fontsize=12)
        ax1.set_ylabel('提及次数', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 子图2: 百分比折线图
        ax2 = axes[0, 1]
        for i, (level2, percentages) in enumerate(active_level2_percentages.items()):
            ax2.plot(x, percentages, marker='o', linewidth=2, markersize=6,
                     label=level2, color=colors[i])

        ax2.set_title(f'{level1_category} - 二级分类关注度趋势 (占比)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间段', fontsize=12)
        ax2.set_ylabel('占比(%)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 子图3: 频次堆叠柱状图
        ax3 = axes[1, 0]
        bottom = np.zeros(len(periods))
        for i, (level2, counts) in enumerate(active_level2.items()):
            ax3.bar(x, counts, bottom=bottom, label=level2, color=colors[i], alpha=0.8)
            bottom += counts

        ax3.set_title(f'{level1_category} - 二级分类关注度分布 (频次)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('时间段', fontsize=12)
        ax3.set_ylabel('提及次数', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(periods, rotation=45, ha='right', fontsize=10)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加总数值
        for j in range(len(periods)):
            total = sum(counts[j] for _, counts in active_level2.items())
            if total > 0:
                ax3.text(j, total, str(int(total)), ha='center', va='bottom',
                         fontsize=9, fontweight='bold')

        # 子图4: 雷达图（展示最后一个时间段的二级分类分布）
        ax4 = axes[1, 1]

        if len(periods) > 0:
            last_period = periods[-1]

            # 获取最后一个时间段的二级分类百分比数据
            radar_data = []
            radar_labels = []

            for level2 in active_level2_percentages.keys():
                full_name = f"{level1_category} - {level2}"
                pct = results['level2_percentage'][last_period].get(full_name, 0)
                if pct > 0:
                    radar_data.append(pct)
                    # 简化标签显示
                    simplified_label = level2
                    if len(simplified_label) > 10:
                        simplified_label = simplified_label[:8] + '...'
                    radar_labels.append(simplified_label)

            if len(radar_data) > 2:  # 雷达图需要至少3个数据点
                # 转换为极坐标
                angles = np.linspace(0, 2 * np.pi, len(radar_data), endpoint=False).tolist()
                radar_data.append(radar_data[0])
                angles.append(angles[0])

                # 创建雷达图
                ax4 = plt.subplot(2, 2, 4, polar=True)
                ax4.plot(angles, radar_data, 'o-', linewidth=2)
                ax4.fill(angles, radar_data, alpha=0.25)
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(radar_labels, fontsize=9)
                ax4.set_title(f'{level1_category} - 二级分类分布雷达图\n({last_period})',
                              fontsize=12, fontweight='bold', pad=20)
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, '数据不足生成雷达图\n(需要至少3个有数据的二级分类)',
                         ha='center', va='center', fontsize=12, transform=ax4.transAxes)
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, '无时间段数据',
                     ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.axis('off')

        plt.tight_layout()

        # 保存图片，不显示
        safe_category = level1_category.replace('/', '_').replace('\\', '_').replace(':', '_')
        chart_path = os.path.join(self.output_dir, f'二级分类趋势图_{safe_category}_{time_period}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"二级分类趋势图已保存到: {chart_path}")

        return list(active_level2.keys())

    def plot_word_cloud_evolution(self, time_period='年份季度', top_n_per_period=8, figsize=(20, 30)):
        """绘制关键词词云演化图（改进版，包含频次和百分比）"""
        if time_period not in self.results:
            print("请先运行 analyze_by_time_period()")
            return

        results = self.results[time_period]
        periods = results['periods']

        # 计算每个时间段最活跃的关键词（频次和百分比）
        top_keywords_by_period_counts = {}
        top_keywords_by_period_percentages = {}

        for period in periods:
            period_keywords_counts = results['level3_attention'][period]
            period_keywords_percentages = results['level3_percentage'][period]

            # 按提及次数排序，取前N个
            sorted_keywords_counts = sorted(period_keywords_counts.items(), key=lambda x: x[1], reverse=True)[
                                     :top_n_per_period]
            sorted_keywords_percentages = sorted(period_keywords_percentages.items(), key=lambda x: x[1], reverse=True)[
                                          :top_n_per_period]

            top_keywords_by_period_counts[period] = {kw: count for kw, count in sorted_keywords_counts}
            top_keywords_by_period_percentages[period] = {kw: pct for kw, pct in sorted_keywords_percentages}

        # 分批显示时间段，每批最多3个
        batch_size = 3
        n_batches = (len(periods) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(periods))
            batch_periods = periods[start_idx:end_idx]

            # 计算子图的行列数（每行最多3个）
            n_periods_batch = len(batch_periods)
            n_cols = min(3, n_periods_batch)
            n_rows = n_periods_batch * 2  # 每个时间段有2个子图：频次和百分比

            # 调整图形大小
            batch_figsize = (figsize[0], 12 * n_periods_batch)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=batch_figsize)

            # 扁平化axes数组以便索引
            if n_rows * n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            # 创建统一的颜色映射
            all_counts = []
            all_percentages = []
            for period in batch_periods:
                all_counts.extend(top_keywords_by_period_counts[period].values())
                all_percentages.extend(top_keywords_by_period_percentages[period].values())
            max_count = max(all_counts) if all_counts else 1
            max_percentage = max(all_percentages) if all_percentages else 1

            for idx, period in enumerate(batch_periods):
                # 频次图
                ax_counts = axes[idx * 2]
                keywords_counts = top_keywords_by_period_counts[period]

                if not keywords_counts:
                    ax_counts.text(0.5, 0.5, f'{period}\n无数据', ha='center', va='center', fontsize=12)
                    ax_counts.axis('off')
                else:
                    # 创建水平条形图（频次）
                    sorted_items = sorted(keywords_counts.items(), key=lambda x: x[1], reverse=True)
                    words = [item[0] for item in sorted_items]
                    counts = [item[1] for item in sorted_items]

                    # 截断过长的关键词
                    max_word_length = 15
                    display_words = []
                    for word in words:
                        if len(word) > max_word_length:
                            display_words.append(word[:max_word_length] + '...')
                        else:
                            display_words.append(word)

                    y_pos = range(len(words))

                    # 根据频率设置颜色深浅
                    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(words)))

                    bars = ax_counts.barh(y_pos, counts, color=colors)

                    ax_counts.set_yticks(y_pos)
                    ax_counts.set_yticklabels(display_words, fontsize=9)
                    ax_counts.invert_yaxis()  # 最高的在最上面
                    ax_counts.set_xlabel('提及次数', fontsize=10)
                    ax_counts.set_title(f'{period} - 频次', fontsize=12, fontweight='bold', pad=15)

                    # 在条形上添加数值
                    for i, (bar, count) in enumerate(zip(bars, counts)):
                        width = bar.get_width()
                        ax_counts.text(width + max_count * 0.02, bar.get_y() + bar.get_height() / 2,
                                       f'{count}', ha='left', va='center', fontsize=8, fontweight='bold')

                    # 设置x轴范围
                    ax_counts.set_xlim([0, max_count * 1.2])

                    # 添加网格
                    ax_counts.grid(True, alpha=0.3, axis='x')

                # 百分比图
                ax_percentages = axes[idx * 2 + 1]
                keywords_percentages = top_keywords_by_period_percentages[period]

                if not keywords_percentages:
                    ax_percentages.text(0.5, 0.5, f'{period}\n无数据', ha='center', va='center', fontsize=12)
                    ax_percentages.axis('off')
                else:
                    # 创建水平条形图（百分比）
                    sorted_items = sorted(keywords_percentages.items(), key=lambda x: x[1], reverse=True)
                    words = [item[0] for item in sorted_items]
                    percentages = [item[1] for item in sorted_items]

                    # 截断过长的关键词
                    max_word_length = 15
                    display_words = []
                    for word in words:
                        if len(word) > max_word_length:
                            display_words.append(word[:max_word_length] + '...')
                        else:
                            display_words.append(word)

                    y_pos = range(len(words))

                    # 根据百分比设置颜色深浅
                    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(words)))

                    bars = ax_percentages.barh(y_pos, percentages, color=colors)

                    ax_percentages.set_yticks(y_pos)
                    ax_percentages.set_yticklabels(display_words, fontsize=9)
                    ax_percentages.invert_yaxis()  # 最高的在最上面
                    ax_percentages.set_xlabel('占比(%)', fontsize=10)
                    ax_percentages.set_title(f'{period} - 占比', fontsize=12, fontweight='bold', pad=15)

                    # 在条形上添加数值
                    for i, (bar, pct) in enumerate(zip(bars, percentages)):
                        width = bar.get_width()
                        ax_percentages.text(width + max_percentage * 0.02, bar.get_y() + bar.get_height() / 2,
                                            f'{pct:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')

                    # 设置x轴范围
                    ax_percentages.set_xlim([0, max_percentage * 1.2])

                    # 添加网格
                    ax_percentages.grid(True, alpha=0.3, axis='x')

            # 隐藏多余的子图
            for idx in range(len(batch_periods) * 2, len(axes)):
                axes[idx].axis('off')

            plt.suptitle(f'关键词关注度演化 - 批次 {batch_idx + 1} ({time_period})',
                         fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()

            # 保存批次图片，不显示
            chart_path = os.path.join(self.output_dir, f'关键词演化图_批次{batch_idx + 1}_{time_period}.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"关键词演化图批次 {batch_idx + 1} 已保存到: {chart_path}")

    def plot_balanced_analysis(self, time_period='年份季度', figsize=(16, 12)):
        """绘制均衡化分析图表"""
        if time_period not in self.results:
            print("请先运行 analyze_by_time_period()")
            return

        results = self.results[time_period]
        periods = results['periods']

        # 计算均衡化指标
        balanced_metrics = {
            'periods': [],
            'keyword_density': [],  # 关键词密度
            'diversity': [],  # 分类多样性
            'concentration': [],  # 集中度（前3分类占比）
            'balance_index': []  # 均衡化指数
        }

        for i, period in enumerate(periods):
            total_keywords = results['total_keywords'][i]
            total_records = results['total_records'][i]

            if total_keywords > 0 and total_records > 0:
                # 关键词密度
                keyword_density = total_keywords / total_records

                # 分类多样性（一级分类的数量）
                diversity = len(results['level1_attention'][period])

                # 集中度（前3个一级分类占比）
                level1_counts = list(results['level1_attention'][period].values())
                level1_counts.sort(reverse=True)
                concentration = sum(level1_counts[:3]) / total_keywords * 100 if len(level1_counts) >= 3 else 100

                # 均衡化指数（多样性/集中度 * 密度）
                balance_index = (diversity * keyword_density) / (concentration + 1) * 100

                balanced_metrics['periods'].append(period)
                balanced_metrics['keyword_density'].append(keyword_density)
                balanced_metrics['diversity'].append(diversity)
                balanced_metrics['concentration'].append(concentration)
                balanced_metrics['balance_index'].append(balance_index)

        if not balanced_metrics['periods']:
            print("无法计算均衡化指标")
            return

        # 创建均衡化分析图表
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        x = range(len(balanced_metrics['periods']))

        # 子图1: 关键词密度
        axes[0, 0].plot(x, balanced_metrics['keyword_density'], 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].fill_between(x, 0, balanced_metrics['keyword_density'], alpha=0.3, color='blue')
        axes[0, 0].set_title('关键词密度趋势', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('时间段', fontsize=12)
        axes[0, 0].set_ylabel('平均关键词数/评论', fontsize=12)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(balanced_metrics['periods'], rotation=45, ha='right', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # 添加数值标签
        for i, density in enumerate(balanced_metrics['keyword_density']):
            axes[0, 0].text(i, density, f'{density:.2f}', ha='center', va='bottom', fontsize=9)

        # 子图2: 分类多样性
        axes[0, 1].bar(x, balanced_metrics['diversity'], color='green', alpha=0.7)
        axes[0, 1].set_title('分类多样性趋势', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('时间段', fontsize=12)
        axes[0, 1].set_ylabel('一级分类数量', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(balanced_metrics['periods'], rotation=45, ha='right', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, diversity in enumerate(balanced_metrics['diversity']):
            axes[0, 1].text(i, diversity, str(diversity), ha='center', va='bottom', fontsize=9)

        # 子图3: 集中度
        axes[1, 0].plot(x, balanced_metrics['concentration'], 'o-', linewidth=2, markersize=8, color='red')
        axes[1, 0].fill_between(x, 0, balanced_metrics['concentration'], alpha=0.3, color='red')
        axes[1, 0].set_title('关注度集中度趋势', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('时间段', fontsize=12)
        axes[1, 0].set_ylabel('前3分类占比(%)', fontsize=12)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(balanced_metrics['periods'], rotation=45, ha='right', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])

        # 添加数值标签
        for i, concentration in enumerate(balanced_metrics['concentration']):
            axes[1, 0].text(i, concentration, f'{concentration:.1f}%', ha='center', va='bottom', fontsize=9)

        # 子图4: 均衡化指数
        axes[1, 1].plot(x, balanced_metrics['balance_index'], 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].fill_between(x, 0, balanced_metrics['balance_index'], alpha=0.3, color='purple')
        axes[1, 1].set_title('均衡化指数趋势', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('时间段', fontsize=12)
        axes[1, 1].set_ylabel('均衡化指数', fontsize=12)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(balanced_metrics['periods'], rotation=45, ha='right', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        # 添加数值标签
        for i, index in enumerate(balanced_metrics['balance_index']):
            axes[1, 1].text(i, index, f'{index:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存图表，不显示
        chart_path = os.path.join(self.output_dir, f'均衡化分析_{time_period}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"均衡化分析图表已保存到: {chart_path}")

        # 输出均衡化分析报告
        print("\n均衡化分析报告:")
        print("-" * 50)
        for i, period in enumerate(balanced_metrics['periods']):
            print(f"{period}:")
            print(f"  关键词密度: {balanced_metrics['keyword_density'][i]:.2f} 关键词/评论")
            print(f"  分类多样性: {balanced_metrics['diversity'][i]} 个一级分类")
            print(f"  关注度集中度: {balanced_metrics['concentration'][i]:.1f}%")
            print(f"  均衡化指数: {balanced_metrics['balance_index'][i]:.1f}")

            # 评价
            density_score = "高" if balanced_metrics['keyword_density'][i] > 3 else "中" if \
                balanced_metrics['keyword_density'][i] > 1.5 else "低"
            diversity_score = "高" if balanced_metrics['diversity'][i] > 6 else "中" if balanced_metrics['diversity'][
                                                                                            i] > 3 else "低"
            concentration_score = "低" if balanced_metrics['concentration'][i] < 60 else "中" if \
                balanced_metrics['concentration'][i] < 80 else "高"
            balance_score = "优" if balanced_metrics['balance_index'][i] > 50 else "良" if \
                balanced_metrics['balance_index'][i] > 30 else "中" if balanced_metrics['balance_index'][
                                                                           i] > 15 else "差"

            print(
                f"  评价: 密度({density_score}), 多样性({diversity_score}), 集中度({concentration_score}), 均衡性({balance_score})")
            print()

        return balanced_metrics

    def generate_comprehensive_report(self):
        """生成综合分析报告（增强版）"""
        report_lines = []

        report_lines.append("=" * 100)
        report_lines.append("汽车用户关注度趋势综合分析报告")
        report_lines.append("=" * 100)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据样本: {len(self.df)} 条")
        report_lines.append(f"三级词库: {len(self.dict_df)} 个词条")
        report_lines.append(
            f"分析时间范围: {self.df[self.time_column].min().date() if self.time_column and not self.df[self.time_column].isna().all() else 'N/A'} 到 {self.df[self.time_column].max().date() if self.time_column and not self.df[self.time_column].isna().all() else 'N/A'}")
        report_lines.append("")

        # 总体统计
        report_lines.append("一、总体统计")
        report_lines.append("-" * 50)

        # 计算关键词覆盖情况
        total_keywords_found = 0
        keywords_used = set()

        for _, row in self.df.iterrows():
            for col in self.text_columns:
                text = row[col]
                if pd.isna(text):
                    continue

                found = self.extract_keywords_from_text(text)
                keywords_used.update(found)
                total_keywords_found += len(found)

        report_lines.append(f"总关键词提及次数: {total_keywords_found}")
        report_lines.append(
            f"使用的关键词数量: {len(keywords_used)} / {len(self.keyword_mapping)} ({len(keywords_used) / len(self.keyword_mapping) * 100:.1f}%)")
        report_lines.append(f"平均每条记录关键词数: {total_keywords_found / len(self.df):.2f}")
        report_lines.append("")

        # 各时间段分析
        for time_period, results in self.results.items():
            report_lines.append(f"\n二、{time_period} 分析结果")
            report_lines.append("-" * 50)

            report_lines.append(f"分析时间段数: {len(results['periods'])}")
            report_lines.append(f"总样本数: {sum(results['total_records'])}")
            report_lines.append(f"总关键词数: {sum(results['total_keywords'])}")
            report_lines.append("")

            # 一级分类排名（频次和百分比）
            report_lines.append("一级分类排名 (按总提及次数):")
            level1_totals = defaultdict(int)
            level1_totals_percentage = defaultdict(float)

            for period in results['periods']:
                for level1, count in results['level1_attention'][period].items():
                    level1_totals[level1] += count
                for level1, pct in results['level1_percentage'][period].items():
                    level1_totals_percentage[level1] = max(level1_totals_percentage[level1], pct)

            sorted_level1 = sorted(level1_totals.items(), key=lambda x: x[1], reverse=True)
            for i, (level1, total) in enumerate(sorted_level1[:10], 1):
                proportion = total / sum(results['total_keywords']) * 100 if sum(results['total_keywords']) > 0 else 0
                max_percentage = level1_totals_percentage.get(level1, 0)
                report_lines.append(f"  {i}. {level1}: {total}次 ({proportion:.1f}%), 峰值占比: {max_percentage:.1f}%")

            report_lines.append("")

            # 趋势变化（同时考虑频次和百分比）
            if len(results['periods']) >= 2:
                report_lines.append("趋势变化分析 (频次和百分比):")

                # 计算每个一级分类的变化率
                first_period = results['periods'][0]
                last_period = results['periods'][-1]

                for level1 in list(level1_totals.keys())[:5]:
                    first_count = results['level1_attention'][first_period].get(level1, 0)
                    last_count = results['level1_attention'][last_period].get(level1, 0)
                    first_pct = results['level1_percentage'][first_period].get(level1, 0)
                    last_pct = results['level1_percentage'][last_period].get(level1, 0)

                    if first_count > 0:
                        count_change_rate = (last_count - first_count) / first_count * 100
                    else:
                        count_change_rate = 100 if last_count > 0 else 0

                    if first_pct > 0:
                        pct_change_rate = (last_pct - first_pct) / first_pct * 100
                    else:
                        pct_change_rate = 100 if last_pct > 0 else 0

                    count_trend = "↑" if count_change_rate > 0 else "↓"
                    pct_trend = "↑" if pct_change_rate > 0 else "↓"

                    report_lines.append(f"  {level1}:")
                    report_lines.append(
                        f"    频次: {first_count} → {last_count} ({count_trend} {abs(count_change_rate):.1f}%)")
                    report_lines.append(
                        f"    占比: {first_pct:.1f}% → {last_pct:.1f}% ({pct_trend} {abs(pct_change_rate):.1f}%)")

        # 均衡化分析
        report_lines.append("\n三、均衡化分析")
        report_lines.append("-" * 50)

        if self.results:
            time_period = list(self.results.keys())[0]
            results = self.results[time_period]

            if len(results['periods']) >= 2:
                # 计算各时间段的均衡化指标
                balanced_data = []
                for i, period in enumerate(results['periods']):
                    total_keywords = results['total_keywords'][i]
                    total_records = results['total_records'][i]

                    if total_keywords > 0 and total_records > 0:
                        # 关键词密度
                        keyword_density = total_keywords / total_records

                        # 分类多样性
                        diversity = len(results['level1_attention'][period])

                        # 集中度
                        level1_counts = list(results['level1_attention'][period].values())
                        level1_counts.sort(reverse=True)
                        concentration = sum(level1_counts[:3]) / total_keywords * 100 if len(
                            level1_counts) >= 3 else 100

                        balanced_data.append({
                            'period': period,
                            'density': keyword_density,
                            'diversity': diversity,
                            'concentration': concentration
                        })

                if balanced_data:
                    report_lines.append("各时间段均衡化指标:")
                    for data in balanced_data:
                        report_lines.append(
                            f"  {data['period']}: 密度={data['density']:.2f}, 多样性={data['diversity']}, 集中度={data['concentration']:.1f}%")

                    # 找出最均衡和最不均衡的时间段
                    if len(balanced_data) > 1:
                        balanced_data_sorted = sorted(balanced_data,
                                                      key=lambda x: x['diversity'] / (x['concentration'] + 1) * x[
                                                          'density'], reverse=True)
                        most_balanced = balanced_data_sorted[0]
                        least_balanced = balanced_data_sorted[-1]

                        report_lines.append(
                            f"\n最均衡时间段: {most_balanced['period']} (密度={most_balanced['density']:.2f}, 多样性={most_balanced['diversity']}, 集中度={most_balanced['concentration']:.1f}%)")
                        report_lines.append(
                            f"最不均衡时间段: {least_balanced['period']} (密度={least_balanced['density']:.2f}, 多样性={least_balanced['diversity']}, 集中度={least_balanced['concentration']:.1f}%)")

        # 洞察与建议
        report_lines.append("\n四、关键洞察与建议")
        report_lines.append("-" * 50)

        if self.results:
            # 找到增长最快的一级分类（同时考虑频次和百分比）
            time_period = list(self.results.keys())[0]
            results = self.results[time_period]

            if len(results['periods']) >= 2:
                first_period = results['periods'][0]
                last_period = results['periods'][-1]

                growth_rates_count = {}
                growth_rates_pct = {}
                for level1 in set(results['level1_attention'][first_period].keys()) | set(
                        results['level1_attention'][last_period].keys()):
                    first_count = results['level1_attention'][first_period].get(level1, 0)
                    last_count = results['level1_attention'][last_period].get(level1, 0)
                    first_pct = results['level1_percentage'][first_period].get(level1, 0)
                    last_pct = results['level1_percentage'][last_period].get(level1, 0)

                    if first_count > 0:
                        growth_rate_count = (last_count - first_count) / first_count * 100
                    else:
                        growth_rate_count = 100 if last_count > 0 else 0

                    if first_pct > 0:
                        growth_rate_pct = (last_pct - first_pct) / first_pct * 100
                    else:
                        growth_rate_pct = 100 if last_pct > 0 else 0

                    growth_rates_count[level1] = growth_rate_count
                    growth_rates_pct[level1] = growth_rate_pct

                # 找到增长最快和下降最快的（频次）
                fastest_growth_count = max(growth_rates_count.items(), key=lambda x: x[1])
                fastest_decline_count = min(growth_rates_count.items(), key=lambda x: x[1])

                # 找到增长最快和下降最快的（百分比）
                fastest_growth_pct = max(growth_rates_pct.items(), key=lambda x: x[1])
                fastest_decline_pct = min(growth_rates_pct.items(), key=lambda x: x[1])

                report_lines.append(
                    f"增长最快的关注点 (频次): {fastest_growth_count[0]} (+{fastest_growth_count[1]:.1f}%)")
                report_lines.append(
                    f"下降最快的关注点 (频次): {fastest_decline_count[0]} ({fastest_decline_count[1]:.1f}%)")
                report_lines.append(f"增长最快的关注点 (占比): {fastest_growth_pct[0]} (+{fastest_growth_pct[1]:.1f}%)")
                report_lines.append(
                    f"下降最快的关注点 (占比): {fastest_decline_pct[0]} ({fastest_decline_pct[1]:.1f}%)")
                report_lines.append("")

                # 建议
                report_lines.append("建议:")
                report_lines.append("1. 加强在增长领域的投入，满足用户日益增长的需求")
                report_lines.append("2. 关注下降领域，分析原因并及时调整策略")
                report_lines.append("3. 针对稳定高关注领域，持续优化产品和服务")
                report_lines.append("4. 使用百分比分析均衡数据量差异，避免大样本主导结论")
                report_lines.append("5. 关注分类多样性，避免关注点过于集中")
                report_lines.append("6. 定期监测用户关注度变化，快速响应市场变化")

        # 保存报告
        report_text = "\n".join(report_lines)
        report_file = os.path.join(self.output_dir, '综合分析报告.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"综合分析报告已保存到: {report_file}")

        return report_text

    def run_full_analysis(self, time_periods=['年份季度']):
        """运行完整分析流程"""
        print("=" * 60)
        print("开始汽车用户关注度趋势分析")
        print("=" * 60)

        # 1. 按时间段分析
        for period in time_periods:
            print(f"\n分析时间维度: {period}")
            self.analyze_by_time_period(period)

            # 2. 保存结果
            self.save_analysis_results(period)

            # 3. 绘制图表
            print("\n生成可视化图表...")
            top_level1 = self.plot_level1_trends(period, top_n=8)  # 限制显示8个一级分类

            if top_level1:
                # 为前3个一级分类生成二级分类图（包含雷达图）
                for level1 in top_level1[:3]:
                    self.plot_level2_trends_with_radar(level1, period)

            self.plot_word_cloud_evolution(period, top_n_per_period=8)  # 限制每个时间段显示8个关键词

            # 新增：均衡化分析
            self.plot_balanced_analysis(period)

        # 4. 生成报告
        print("\n生成综合分析报告...")
        self.generate_comprehensive_report()

        print("\n" + "=" * 60)
        print("分析完成！所有结果已保存到:")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)

        return self


def main():
    """主函数"""
    # 文件路径
    review_data_path = "10.汽车口碑数据/merged_data_V2.xlsx"
    dict_path = "30.自定义词典/汽车三级词库V3.xlsx"
    output_dir = "50. 用户关注度分析"

    print("正在加载数据...")

    try:
        # 加载评论数据
        review_df = pd.read_excel(review_data_path, sheet_name='clean_data_xlsx')
        print(f"成功加载评论数据: {review_df.shape}")
    except Exception as e:
        print(f"加载评论数据失败: {e}")
        return

    try:
        # 创建分析器
        analyzer = AutomotiveAttentionAnalyzer(review_df, dict_path, output_dir)

        # 运行完整分析
        analyzer.run_full_analysis(time_periods=['年份季度'])

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()