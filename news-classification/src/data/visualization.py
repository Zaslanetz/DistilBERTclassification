import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def create_category_distribution_chart(save_path):
    """
    Creates a stepped bar chart showing the distribution of categories in the dataset
    with exact counts displayed on each bar
    """
    # Dataset category distribution
    categories = ['Business', 'Sport', 'Politics', 'Tech', 'Entertainment']
    counts = [510, 511, 417, 401, 386]
    
    # Sort by count for stepped effect
    sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_data]
    counts = [item[1] for item in sorted_data]
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
    
    # Category-specific colors
    colors = {
        'Business': '#3b82f6',
        'Entertainment': '#ec4899',
        'Politics': '#f59e0b',
        'Sport': '#10b981',
        'Tech': '#8b5cf6'
    }
    
    # Create bars with category-specific colors
    bars = ax.bar(
        categories, 
        counts, 
        width=0.6,
        color=[colors[cat] for cat in categories],
        edgecolor='white',
        linewidth=0.7
    )
    
    # Add exact count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=14,
            fontweight='bold'
        )
    
    # Customize chart appearance
    ax.set_title('BBC News Dataset - Category Distribution', fontsize=18, pad=20)
    ax.set_ylabel('Number of Articles', fontsize=14)
    ax.set_ylim(0, max(counts) + 60)  # Add space for labels
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the chart
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Category distribution chart saved to {save_path}")

# Example usage:
if __name__ == "__main__":
    create_category_distribution_chart('static/category_distribution.png')