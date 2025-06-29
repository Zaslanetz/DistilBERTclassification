# src/data/text_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import seaborn as sns
import os  # Add this import

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def analyze_text_length(df, category_col='category', text_col='text'):
    """Analyze text length by category"""
    # Add character count
    df['char_count'] = df[text_col].apply(len)
    
    # Add word count
    df['word_count'] = df[text_col].apply(lambda x: len(word_tokenize(x)))
    
    # Plot text length distribution
    plt.figure(figsize=(12, 5))
    
    # Character count by category
    plt.subplot(1, 2, 1)
    sns.boxplot(x=category_col, y='char_count', data=df)
    plt.title('Character Count by Category')
    plt.xticks(rotation=45)
    
    # Word count by category
    plt.subplot(1, 2, 2)
    sns.boxplot(x=category_col, y='word_count', data=df)
    plt.title('Word Count by Category')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('static/text_length_analysis.png')
    
    return {
        'char_count_mean': df.groupby(category_col)['char_count'].mean().to_dict(),
        'char_count_median': df.groupby(category_col)['char_count'].median().to_dict(),
        'word_count_mean': df.groupby(category_col)['word_count'].mean().to_dict(),
        'word_count_median': df.groupby(category_col)['word_count'].median().to_dict(),
    }

def generate_wordclouds(df, category_col='category', text_col='text'):
    """Generate wordclouds for each category"""
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Create directory for wordclouds if it doesn't exist
    os.makedirs('static/wordclouds', exist_ok=True)
    
    # Generate wordcloud for each category
    categories = df[category_col].unique()
    for category in categories:
        # Get texts for this category
        texts = ' '.join(df[df[category_col] == category][text_col])
        
        # Generate wordcloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stop_words,
            max_words=100
        ).generate(texts)
        
        # Save wordcloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {category}')
        plt.tight_layout()
        plt.savefig(f'static/wordclouds/{category}_wordcloud.png')
        plt.close()

def get_top_words(df, category_col='category', text_col='text', n=10):
    """Get top N words for each category"""
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Store results
    results = {}
    
    # Get top words for each category
    categories = df[category_col].unique()
    for category in categories:
        # Get texts for this category
        texts = ' '.join(df[df[category_col] == category][text_col])
        
        # Tokenize
        words = word_tokenize(texts.lower())
        
        # Remove stopwords and punctuation
        words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get top N words
        top_words = word_counts.most_common(n)
        
        results[category] = top_words
    
    # Plot top words
    plt.figure(figsize=(15, 10))
    for i, category in enumerate(categories):
        plt.subplot(len(categories), 1, i+1)
        words, counts = zip(*results[category])
        plt.barh(words, counts)
        plt.title(f'Top {n} Words - {category}')
        plt.tight_layout()
    
    plt.savefig('static/top_words.png')
    return results