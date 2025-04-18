# visualize.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import pandas as pd

def plot_post_addiction_distribution(df):
    # define bins and original numeric labels
    bins   = [-0.1, 1, 4, 7]
    numeric_labels = ['0‑1', '2‑4', '5‑7']
    category_names = ['Low (0–1)', 'Medium (2–4)', 'High (5–7)']

    # compute category percentages
    pct = (df
           .assign(cat=pd.cut(df['AddictionLevel'], bins=bins, labels=numeric_labels))
           ['cat']
           .value_counts(normalize=True)
           .reindex(numeric_labels)
           .mul(100))

    # choose a light pastel color palette
    colors = ['#B3E5FC',  # light blue for Low
              '#FFECB3',  # light yellow for Medium
              '#E1BEE7']  # light lavender for High

    # plot pie chart without slice labels (we'll use a legend)
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        pct,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        colors=colors
    )

    # add title
    ax.set_title('Distribution of Addiction Level Label')

    # ensure it's a circle
    ax.axis('equal')

    # add legend in upper right corner
    ax.legend(
        wedges,
        category_names,
        title='Addiction Level',
        loc='upper right',
        bbox_to_anchor=(1.15, 1)
    )

    plt.tight_layout()
    plt.savefig("plots/post_addiction_level.png", bbox_inches='tight')
    plt.close()
def plot_correlation(df):
    #  keep numeric columns only
    numeric_df = df.select_dtypes(include='number')

    # 3 ▸ compute correlations
    corr = numeric_df.corr()

    # 4 ▸ draw heat‑map
    plt.figure(figsize=(12, 9))
    im = plt.imshow(corr, interpolation='nearest')  # default colormap
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Correlation Heatmap')
    plt.tight_layout()

    # 5 ▸ save figure
    plt.savefig('plots/correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_addiction_polar(df, column='AddictionLevel', save_path='plots\Addiction Level Polar.png'):
    """
    Plots the proportion of each category in `column` as a polar area chart.

    Parameters:
    - data: pandas DataFrame containing the data.
    - column: name of the categorical column to plot.
    - save_path: filename to save the generated plot.
    """
    # Calculate proportions
    counts = df[column].value_counts().sort_index()
    proportions = counts / counts.sum()
    labels = [str(l) for l in counts.index]
    sizes = proportions.values

    # Angles and bar widths
    N = len(sizes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inner_radius = 0.2
    max_radius = 1.0
    radii = inner_radius + (sizes / sizes.max()) * (max_radius - inner_radius)
    width = 2 * np.pi / N * 0.8

    # Colors
    cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, N))

    # Create polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw bars
    bars = ax.bar(angles, radii - inner_radius, width=width, bottom=inner_radius,
                  color=colors, edgecolor='white', linewidth=1.5)

    # Central circle with label
    circle = plt.Circle((0, 0), inner_radius, transform=ax.transData._b, color='white', zorder=10)
    ax.add_artist(circle)
    ax.text(0, 0, 'Addiction\nLevel', ha='center', va='center', fontsize=14, weight='bold')

    # Add labels with updated text above each bar
    for angle, radius, label, size, color in zip(angles, radii, labels, sizes, colors):
        angle_deg = np.degrees(angle)
        alignment = 'left' if np.cos(angle) >= 0 else 'right'
        rotation = angle_deg if -90 <= angle_deg <= 90 else angle_deg + 180
        ax.text(angle, radius + 0.05,
                f"Addiction Level = {label}\n{size * 100:.1f}%",
                ha=alignment, va='center', rotation=rotation,
                rotation_mode='anchor', fontsize=11, color=color)

    # Legend mapping colors to labels and percentages
    legend_labels = [f"{lab}: {prop * 100:.1f}%" for lab, prop in zip(labels, proportions)]
    ax.legend(bars, legend_labels, title="Addiction Level",
              bbox_to_anchor=(1.1, 1.05), loc='upper left')

    # Hide axes
    ax.set_axis_off()

    save_path = 'plots\Addiction Level Polar.png'
    # Save and show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def model_arch():
    """
        Run following commands if error.
        pip install graphviz
        brew install graphviz
    """
    import os
    from graphviz import Digraph

    # Ensure output folder exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create a directed graph with top‑to‑bottom orientation
    dot = Digraph(format='png')
    dot.attr(rankdir='TB', size='5,10')

    # Nodes
    dot.node('A', 'Raw Dataset', shape='cylinder', style='filled', fillcolor='#ffcccc')
    dot.node('B', 'Preprocessing', shape='box', style='filled', fillcolor='#ccffcc')
    dot.node('C', 'Train/Test Split\n(stratified 80/20)', shape='box', style='filled', fillcolor='#ccccff')
    dot.node('D', 'Baseline RF\n100 trees', shape='box', style='filled', fillcolor='#ffe4b5')
    dot.node('E', 'Improved Preprocessing\n Features and Encoding', shape='box', style='filled', fillcolor='#ccffcc')
    dot.node('F', 'SMOTENC\nOversample', shape='box', style='filled', fillcolor='#ccffcc')
    dot.node('G1', 'Hypertuned GBDT', shape='box', style='filled', fillcolor='#99ccff')
    dot.node('G2', '(RF + LR) Stacking', shape='box', style='filled', fillcolor='#99ccff')
    dot.node('H', 'Evaluation\nMetrics', shape='box', style='filled', fillcolor='#f2f2f2')

    # Edges
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D', xlabel='Baseline Pipeline',
             fontcolor='black', fontsize='12')
    dot.edge('C', 'E', label='Improvement Pipeline', fontsize='12')
    dot.edge('E', 'F')
    dot.edge('F', 'G1')
    dot.edge('F', 'G2')
    dot.edge('D', 'H')
    dot.edge('G1', 'H')
    dot.edge('G2', 'H')

    # Render and save to plots/
    output_path = dot.render(filename=os.path.join(output_dir, 'pipeline_vertical'),
                             cleanup=True)
    print(f"Saved vertical flowchart to {output_path}.png")

def main():
    path = "data/social_media_dataset.csv"
    df = df = pd.read_csv(path)

    # original addiction level distribution
    plot_addiction_polar(df)
    # correlation heatmap
    plot_correlation(df)
    # Addiction Level distribution after relabelling.
    plot_post_addiction_distribution(df)
    # visualize of model architecture
    model_arch()

if __name__ == "__main__":
    main()