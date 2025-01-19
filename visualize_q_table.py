import os
import pickle
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
import umap
import plotly.express as px
import matplotlib.patches as patches

# Ensure plots use a style that is visually appealing
sns.set(style='whitegrid')


def find_latest_q_table(directory='.'):
    """
    Finds the latest Q-table pickle file in the given directory.
    
    Args:
        directory (str): The directory to search in.
    
    Returns:
        str: The filename of the latest Q-table pickle file.
    """
    pkl_files = [f for f in os.listdir(directory) if f.startswith('q_table-') and f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError("No Q-table pickle files found in the specified directory.")
    latest_file = max(pkl_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)


def load_q_table(file_path):
    """
    Loads the Q-table from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        tuple: The Q-table dictionary and epsilon value.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        q_table = data.get('q_table', {})
        epsilon = data.get('epsilon', None)
    print(f"Loaded Q-table from {file_path}. Total states: {len(q_table)}. Epsilon: {epsilon}")
    return q_table, epsilon


def q_table_to_dataframe(q_table):
    """
    Converts the Q-table dictionary to a pandas DataFrame.
    
    Args:
        q_table (dict): The Q-table dictionary.
    
    Returns:
        pd.DataFrame: DataFrame with state features, actions, and Q-values.
    """
    records = []
    for state, actions in q_table.items():
        state_features = state  # Assuming state is a tuple of feature labels
        for action, q_value in actions.items():
            record = {
                'state': state,
                'action': action.name if hasattr(action, 'name') else str(action),
                'q_value': q_value
            }
            # Optionally, split state features into separate columns
            for idx, feature in enumerate(state_features):
                record[f'state_feature_{idx+1}'] = feature
            records.append(record)
    df = pd.DataFrame(records)
    print(f"Converted Q-table to DataFrame with {len(df)} records.")
    return df


def visualize_policy(df):
    """
    Visualizes the optimal policy on a grid if states have spatial features.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
    """
    # Extract spatial features
    feature_cols = [col for col in df.columns if col.startswith('state_feature_')]
    
    if len(feature_cols) < 2:
        print("Policy visualization requires at least two state features for spatial representation.")
        return
    
    # Assuming the first two features are x and y coordinates
    x_col, y_col = feature_cols[:2]
    
    # Determine the best action for each state
    best_actions = df.loc[df.groupby('state')['q_value'].idxmax()]
    
    # Create a grid
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Define action to direction mapping
    action_directions = {
        'up': (0, 0.4),
        'down': (0, -0.4),
        'left': (-0.4, 0),
        'right': (0.4, 0)
    }
    
    # Plot each state with an arrow indicating the best action
    for _, row in best_actions.iterrows():
        x = row[x_col]
        y = row[y_col]
        action = row['action'].lower()
        
        dx, dy = action_directions.get(action, (0, 0))  # Default to no movement if action not recognized
        
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')
        ax.plot(x, y, 'ko')  # Mark the state location
    
    plt.title('Optimal Policy Visualization')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()


def action_specific_heatmaps(df):
    """
    Creates separate heatmaps for each action showing Q-values across two state features.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
    """
    feature_cols = [col for col in df.columns if col.startswith('state_feature_')]
    if len(feature_cols) < 2:
        print("Action-specific heatmaps require at least two state features.")
        return
    
    feature_x, feature_y = feature_cols[:2]
    actions = df['action'].unique()
    
    num_actions = len(actions)
    fig, axes = plt.subplots(1, num_actions, figsize=(6 * num_actions, 5), squeeze=False)
    
    for idx, action in enumerate(actions):
        action_df = df[df['action'] == action]
        pivot_table = action_df.pivot_table(index=feature_y, columns=feature_x, values='q_value', aggfunc='mean')
        sns.heatmap(pivot_table, cmap='viridis', ax=axes[0, idx], cbar=(idx == 0))
        axes[0, idx].set_title(f'Heatmap of Q-Values for Action: {action}')
        axes[0, idx].set_xlabel(feature_x)
        axes[0, idx].set_ylabel(feature_y)
    
    plt.tight_layout()
    plt.show()


def cluster_states_umap(df):
    """
    Clusters states using UMAP and visualizes the clusters.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
    """
    feature_cols = [col for col in df.columns if col.startswith('state_feature_')]
    if not feature_cols:
        print("No state feature columns found for clustering.")
        return
    
    # Aggregate Q-values by state
    state_best = df.loc[df.groupby('state')['q_value'].idxmax()]
    X = state_best[feature_cols].values
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X)
    
    state_best['umap_1'] = embedding[:, 0]
    state_best['umap_2'] = embedding[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='umap_1', y='umap_2',
        hue='action',
        palette='tab10',
        data=state_best,
        legend='full',
        alpha=0.7
    )
    plt.title('UMAP Clustering of States Colored by Best Action')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def interactive_tsne(df):
    """
    Creates an interactive t-SNE visualization using Plotly.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
    """
    feature_columns = [col for col in df.columns if col.startswith('state_feature_')]
    if not feature_columns:
        print("No state feature columns found for t-SNE visualization.")
        return
    
    state_best = df.loc[df.groupby('state')['q_value'].idxmax()]
    state_features = pd.get_dummies(state_best[feature_columns])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(state_features)
    state_best['tsne_1'] = tsne_results[:, 0]
    state_best['tsne_2'] = tsne_results[:, 1]
    
    fig = px.scatter(
        state_best,
        x='tsne_1',
        y='tsne_2',
        color='action',
        title='Interactive t-SNE Visualization of States Colored by Best Action',
        labels={'tsne_1': 't-SNE Dimension 1', 'tsne_2': 't-SNE Dimension 2'},
        hover_data=feature_columns
    )
    fig.update_layout(legend=dict(title='Action'), width=800, height=600)
    fig.show()


def box_violin_plots(df):
    """
    Creates box and violin plots for Q-values across different actions.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
    """
    plt.figure(figsize=(14, 6))
    
    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='action', y='q_value', data=df, palette='Set2')
    plt.title('Box Plot of Q-Values per Action')
    plt.xlabel('Action')
    plt.ylabel('Q-Value')
    plt.xticks(rotation=45)
    
    # Violin Plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x='action', y='q_value', data=df, palette='Set3')
    plt.title('Violin Plot of Q-Values per Action')
    plt.xlabel('Action')
    plt.ylabel('Q-Value')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def correlation_matrix(df):
    """
    Plots a correlation matrix of state features and Q-values.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
    """
    feature_cols = [col for col in df.columns if col.startswith('state_feature_')]
    if not feature_cols:
        print("No state feature columns found for correlation matrix.")
        return
    
    # Aggregate Q-values by state
    state_best = df.loc[df.groupby('state')['q_value'].idxmax()]
    
    # Select relevant columns
    corr_df = state_best[feature_cols + ['q_value']]
    
    # Compute correlation matrix
    corr = corr_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of State Features and Q-Values')
    plt.tight_layout()
    plt.show()


def create_visualizations(df, epsilon):
    """
    Creates and displays various visualizations from the Q-table DataFrame.
    
    Args:
        df (pd.DataFrame): The Q-table DataFrame.
        epsilon (float): The current epsilon value.
    """
    # Q-Value Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['q_value'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Q-Values')
    plt.xlabel('Q-Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    # Average Q-Value per Action
    plt.figure(figsize=(10, 6))
    avg_q_per_action = df.groupby('action')['q_value'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_q_per_action.index, y=avg_q_per_action.values, palette='viridis')
    plt.title('Average Q-Value per Action')
    plt.xlabel('Action')
    plt.ylabel('Average Q-Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Heatmap of Q-Values for two selected state features
    feature_cols = [col for col in df.columns if col.startswith('state_feature_')]
    if len(feature_cols) >= 2:
        feature_x = feature_cols[0]
        feature_y = feature_cols[1]
        pivot_table = df.pivot_table(index=feature_y, columns=feature_x, values='q_value', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='coolwarm', linewidths=.5)
        plt.title(f'Heatmap of Average Q-Values\nFeatures: {feature_x} vs {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough state features for heatmap visualization.")
    
    # Interactive t-SNE Visualization
    interactive_tsne(df)
    
    # Policy Visualization
    visualize_policy(df)
    
    # Action-Specific Q-Value Heatmaps
    action_specific_heatmaps(df)
    
    # UMAP Clustering of States
    cluster_states_umap(df)
    
    # Box and Violin Plots for Q-Values
    box_violin_plots(df)
    
    # Correlation Matrix
    correlation_matrix(df)
    
    # Display Epsilon
    print(f"Final Epsilon: {epsilon}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Q-Table from Q-Learning Brain.')
    parser.add_argument('--file', type=str, default=None, help='Path to the Q-table pickle file.')
    args = parser.parse_args()
    
    # Determine Q-table file path
    if args.file:
        if not os.path.exists(args.file):
            print(f"Specified file {args.file} does not exist.")
            return
        q_table_file = args.file
    else:
        try:
            q_table_file = find_latest_q_table()
            print(f"No file specified. Using the latest Q-table file: {q_table_file}")
        except FileNotFoundError as e:
            print(e)
            return
    
    # Load Q-table
    q_table, epsilon = load_q_table(q_table_file)
    
    # Convert Q-table to DataFrame
    df = q_table_to_dataframe(q_table)
    
    # Create and display visualizations
    create_visualizations(df, epsilon)
    print("All visualizations displayed on screen.")


if __name__ == '__main__':
    main()
