import json
import matplotlib.pyplot as plt
import numpy as np

with open('results_bm25spyrs.json') as f:
    bm25spyrs = json.load(f)
with open('results_BM25s.json') as f:
    bm25s = json.load(f)

# indexing time
datasets = [entry['dataset'] for entry in bm25spyrs]
time_spyrs = [entry['indexing_time'] for entry in bm25spyrs]
time_bm25s = [entry['indexing_time'] for entry in bm25s]

x = np.arange(len(datasets))
bar_width = 0.35

plt.figure(figsize=(15, 7))
plt.bar(x - bar_width/2, time_spyrs, width=bar_width, label='bm25spyrs', alpha=0.8)
plt.bar(x + bar_width/2, time_bm25s, width=bar_width, label='BM25s', alpha=0.8)

plt.title('Indexing Time Comparison (BM25 variants)', fontsize=14)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Indexing Time (seconds)', fontsize=12)
plt.xticks(x, datasets, rotation=45, ha='right')
plt.yscale('log')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# retrieval time
queries_count_spyrs = [entry['queries_count'] for entry in bm25spyrs]
qps_spyrs = [entry['queries_count'] / entry['retrieval_total_time'] for entry in bm25spyrs]
qps_bm25s = [entry['queries_count'] / entry['retrieval_total_time'] for entry in bm25s]

# Normalize QPS values using log10 transformation to handle large differences
qps_spyrs_normalized = np.log10(qps_spyrs)
qps_bm25s_normalized = np.log10(qps_bm25s)

new_labels = [f"{dataset}\n@{queries}" for dataset, queries in zip(datasets, queries_count_spyrs)]

plt.figure(figsize=(15, 8))
bars1 = plt.bar(x - bar_width/2, qps_spyrs_normalized, width=bar_width,
                label='bm25spyrs', alpha=0.8, color='#1f77b4')
bars2 = plt.bar(x + bar_width/2, qps_bm25s_normalized, width=bar_width,
                label='BM25s', alpha=0.8, color='#ff7f0e')

plt.title('Queries Per Second (QPS) Comparison - Log Scale Normalized', fontsize=14)
plt.xlabel('Dataset (@number of queries)', fontsize=12)
plt.ylabel('Log10(QPS)', fontsize=12)
plt.xticks(x, new_labels, rotation=45, ha='right')
plt.legend()

# Add actual QPS values as text on bars
for bar, qps in zip(bars1, qps_spyrs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{qps:.1f}',
             ha='center', va='bottom', fontsize=8, color='black', rotation=0)

for bar, qps in zip(bars2, qps_bm25s):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{qps:.1f}',
             ha='center', va='bottom', fontsize=8, color='black', rotation=0)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Alternative: Side-by-side comparison with relative performance
plt.figure(figsize=(15, 8))

# Create subplot for better comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Left plot: Raw QPS values (log scale)
bars1_raw = ax1.bar(x - bar_width/2, qps_spyrs, width=bar_width,
                    label='bm25spyrs', alpha=0.8, color='#1f77b4')
bars2_raw = ax1.bar(x + bar_width/2, qps_bm25s, width=bar_width,
                    label='BM25s', alpha=0.8, color='#ff7f0e')

ax1.set_title('Queries Per Second (QPS) - Raw Values', fontsize=14)
ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_ylabel('QPS (log scale)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=45, ha='right')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on bars
for bar, qps in zip(bars1_raw, qps_spyrs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
             f'{qps:.1f}',
             ha='center', va='bottom', fontsize=8, color='black')

for bar, qps in zip(bars2_raw, qps_bm25s):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
             f'{qps:.1f}',
             ha='center', va='bottom', fontsize=8, color='black')

# Right plot: Relative performance (ratio)
qps_ratio = [spyrs/bm25s for spyrs, bm25s in zip(qps_spyrs, qps_bm25s)]
colors = ['green' if ratio > 1 else 'red' for ratio in qps_ratio]

bars_ratio = ax2.bar(x, qps_ratio, width=bar_width*1.5, alpha=0.8, color=colors)
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal Performance')

ax2.set_title('Relative Performance (bm25spyrs/BM25s)', fontsize=14)
ax2.set_xlabel('Dataset', fontsize=12)
ax2.set_ylabel('QPS Ratio', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add ratio values on bars
for bar, ratio in zip(bars_ratio, qps_ratio):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.,
             height + (0.1 if height > 0 else -0.1),
             f'{ratio:.2f}x',
             ha='center', va='bottom' if height > 0 else 'top',
             fontsize=10, color='black')

plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns

# effectiveness
metrics = ['ndcg', 'map', 'recall', 'p']
df_list = []

for m in metrics:
    for entry_spyrs, entry_s in zip(bm25spyrs, bm25s):
        dataset = entry_spyrs['dataset']
        for k in entry_spyrs[m].keys():
            df_list.append({
                'dataset': dataset,
                'metric': f"{m.upper()}{k.split('@')[1]}",
                'bm25spyrs': entry_spyrs[m][k],
                'BM25s': entry_s[m][k]
            })

df = pd.DataFrame(df_list)

plt.figure(figsize=(15, 12))
sns.set_theme(style="whitegrid")

g = sns.FacetGrid(df, col="metric", col_wrap=4, height=3.5, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="bm25spyrs", y="BM25s", s=100, alpha=0.8,
                hue="dataset", palette="tab20", edgecolor='w', legend=False)
g.set_axis_labels("bm25spyrs Score", "BM25s Score")

for ax in g.axes.flat:
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))

plt.suptitle('BEIR Metrics Correlation Between Models', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# matrix size
matrix_spyrs = [entry['matrix_size'] for entry in bm25spyrs]
matrix_bm25s = [entry['matrix_size'] for entry in bm25s]

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(x=matrix_spyrs, y=matrix_bm25s, s=150,
                     hue=datasets, palette="tab20", legend='full')

lim = max(max(matrix_spyrs), max(matrix_bm25s)) * 1.1
plt.plot([0, lim], [0, lim], 'k--', alpha=0.5)
plt.xlabel('bm25spyrs Matrix Size (MB)')
plt.ylabel('BM25s Matrix Size (MB)')
plt.title('Index Size Parity Plot', fontsize=14)
plt.xlim(0, lim)
plt.ylim(0, lim)
plt.grid(alpha=0.3)

for i, (x, y) in enumerate(zip(matrix_spyrs, matrix_bm25s)):
    if abs(x - y) > 0.1 * max(x, y):  # Highlight >10% differences
        plt.annotate(datasets[i], (x, y),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# grid for effectiveness
datasets = [entry['dataset'] for entry in bm25spyrs]
precision_metrics = ['P@1', 'P@10', 'P@100']

comparison_data = []
for i, dataset in enumerate(datasets):
    row = {'Dataset': dataset}
    for metric in precision_metrics:
        bm25s_val = bm25s[i]['p'][metric]
        spyrs_val = bm25spyrs[i]['p'][metric]
        row[f'{metric}_BM25s'] = bm25s_val
        row[f'{metric}_bm25spyrs'] = spyrs_val
        row[f'{metric}_diff'] = abs(bm25s_val - spyrs_val)
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('BEIR Precision Metrics: BM25s vs bm25spyrs Comparison', fontsize=16, fontweight='bold')

ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Dataset', 'P@1 (BM25s)', 'P@1 (spyrs)', 'P@10 (BM25s)', 'P@10 (spyrs)', 'P@100 (BM25s)', 'P@100 (spyrs)']
table_data.append(headers)

for i, dataset in enumerate(datasets):
    row = [
        dataset,
        f"{bm25s[i]['p']['P@1']:.5f}",
        f"{bm25spyrs[i]['p']['P@1']:.5f}",
        f"{bm25s[i]['p']['P@10']:.5f}",
        f"{bm25spyrs[i]['p']['P@10']:.5f}",
        f"{bm25s[i]['p']['P@100']:.5f}",
        f"{bm25spyrs[i]['p']['P@100']:.5f}"
    ]
    table_data.append(row)

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(1, len(table_data)):
    for j in range(1, len(headers)):
        if j % 2 == 0:  # spyrs columns
            bm25s_idx = j - 1
            diff = abs(float(table_data[i][j]) - float(table_data[i][bm25s_idx]))
            if diff < 0.0001:
                table[(i, j)].set_facecolor('#90EE90')  # Light green for identical
                table[(i, bm25s_idx)].set_facecolor('#90EE90')
            elif diff < 0.001:
                table[(i, j)].set_facecolor('#FFE4B5')  # Light orange for small diff
                table[(i, bm25s_idx)].set_facecolor('#FFE4B5')

for j in range(len(headers)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(weight='bold', color='white')

plt.tight_layout()
plt.show()
