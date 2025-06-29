import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean and preprocess text"""
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text
    return ""

def load_bbc_news_dataset(data_path, test_size=0.2, val_size=0.1):
    """Load BBC News dataset and split into train/val/test sets"""
    # For BBC News dataset, try various reading methods
    try:
        # Try reading with default parameters but specified separator
        df = pd.read_csv(data_path, sep=',', error_bad_lines=False, warn_bad_lines=True)
    except:
        try:
            # Try with Python engine which is more flexible
            df = pd.read_csv(data_path, sep=',', engine='python')
        except:
            # Last resort: try to read with tab delimiter
            df = pd.read_csv(data_path, sep='\t', engine='python')
    
    # Ensure we have the expected columns
    expected_columns = ['category', 'filename', 'title', 'content']
    if not all(col in df.columns for col in expected_columns):
        print("Warning: Missing expected columns in dataset.")
        # If missing some columns but we have category, we can proceed
        if 'category' in df.columns:
            print("Continuing with available columns...")
        else:
            raise ValueError("Critical column 'category' is missing!")
    
    # Add text column from content
    if 'text' not in df.columns and 'content' in df.columns:
        df['text'] = df['content'].apply(clean_text)
    elif 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['category'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), 
                                         stratify=train_df['category'], random_state=42)
    
    return train_df, val_df, test_df