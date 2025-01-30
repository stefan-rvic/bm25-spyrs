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
retrieval_spyrs = [entry['retrieval_total_time'] for entry in bm25spyrs]
retrieval_bm25s = [entry['retrieval_total_time'] for entry in bm25s]
queries_count_spyrs = [entry['queries_count'] for entry in bm25spyrs]
qps_spyrs = [entry['queries_count'] / entry['retrieval_total_time'] for entry in bm25spyrs]
qps_bm25s = [entry['queries_count'] / entry['retrieval_total_time'] for entry in bm25s]

new_labels = [f"{dataset} @{queries}" for dataset, queries in zip(datasets, queries_count_spyrs)]

plt.figure(figsize=(15, 7))
bars1 = plt.bar(x - bar_width/2, retrieval_spyrs, width=bar_width, label='bm25spyrs', alpha=0.8, color='#1f77b4')
bars2 = plt.bar(x + bar_width/2, retrieval_bm25s, width=bar_width, label='BM25s', alpha=0.8, color='#ff7f0e')

plt.title('Total Retrieval Time Comparison (BM25 variants)', fontsize=14)
plt.xlabel('Dataset (@number of queries)', fontsize=12)
plt.ylabel('Retrieval Time (seconds)', fontsize=12)
plt.xticks(x, new_labels, rotation=45, ha='right')
plt.yscale('log')
plt.legend()

for bar, qps in zip(bars1, qps_spyrs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{qps:.1f} QPS',
             ha='center', va='bottom', fontsize=8, color='black')

for bar, qps in zip(bars2, qps_bm25s):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{qps:.1f} QPS',
             ha='center', va='bottom', fontsize=8, color='black')

plt.grid(axis='y', linestyle='--', alpha=0.7)
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