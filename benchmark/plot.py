import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    models = ['bm25spyrs', 'BM25s']
    dfs = []

    for model in models:
        with open(f'results_{model}.json') as f:
            data = json.load(f)

            for entry in data:
                for metric in ['ndcg', 'map', 'recall', 'p']:
                    if metric in entry:
                        for k, v in entry[metric].items():
                            entry[k] = v
                        del entry[metric]
            df = pd.DataFrame(data)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_visualizations(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    df['queries_per_sec'] = df['queries_count'] / df['retrieval_total_time']

    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(data=df, x='dataset', y='indexing_time', hue='model_name', ax=axes[0])
    axes[0].set_title('Indexing Time Comparison')
    axes[0].set_ylabel('Seconds')
    axes[0].set_xlabel('Dataset')

    sns.barplot(data=df, x='dataset', y='queries_per_sec', hue='model_name', ax=axes[1])
    axes[1].set_title('Query Throughput Comparison')
    axes[1].set_ylabel('Queries per Second')
    axes[1].set_xlabel('Dataset')

    sns.barplot(data=df, x='dataset', y='approx_mem', hue='model_name', ax=axes[2])
    axes[2].set_title('Memory Usage Comparison')
    axes[2].set_ylabel('MB')
    axes[2].set_xlabel('Dataset')

    plt.tight_layout()

    metrics = ['NDCG', 'MAP', 'Recall', 'P']
    cutoffs = ['@1', '@10', '@100']

    fig2, axes = plt.subplots(4, 3, figsize=(20, 15))

    for i, metric in enumerate(metrics):
        for j, cutoff in enumerate(cutoffs):
            col = f"{metric}{cutoff}"
            sns.barplot(data=df, x='dataset', y=col, hue='model_name',
                        ax=axes[i, j], palette='Set2')
            axes[i, j].set_title(f'{col} Comparison')
            axes[i, j].set_ylabel('Score')
            axes[i, j].set_xlabel('Dataset')
            if i < 3 or j < 2:
                axes[i, j].get_legend().remove()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_data()
    create_visualizations(df)
