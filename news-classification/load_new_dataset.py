from src.data.preprocessing import load_news_category_dataset, visualize_first_rows, load_bbc_news_dataset

def prepare_new_data(dataset='bbc_news'):
    
    """
    Load and preprocess the selected dataset.
    dataset: 'news_category' or 'bbc_news'
    """
    if dataset == 'news_category':
        data_path = 'data/News_Category_Dataset_v3.json/News_Category_Dataset_v3.json'
        loader = load_news_category_dataset
    elif dataset == 'bbc_news':
        data_path = 'data/bbc-news-data.csv'  # Update this path as needed
        loader = load_bbc_news_dataset
    else:
        raise ValueError("Unknown dataset. Choose 'news_category' or 'bbc_news'.")

    # Load and preprocess the new dataset   
    train_df, val_df, test_df = loader(data_path)
    print("Train DataFrame (first 10 rows):")
    visualize_first_rows(train_df, 10)

    print("\nValidation DataFrame (first 10 rows):")
    visualize_first_rows(val_df, 10)

    print("\nTest DataFrame (first 10 rows):")
    visualize_first_rows(test_df, 10)

if __name__ == "__main__":
    prepare_new_data()