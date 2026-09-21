"""
Data loader for Amazon Electronics dataset (Optimized)
"""

import pandas as pd
import numpy as np

DATASET_URL = (
    'https://raw.githubusercontent.com/abdelaziztestas/'
    'spark_book/main/amazon_electronics.csv'
)

COLUMN_NAMES = ['user_id', 'product_id', 'rating', 'timestamp']

_cached_df = None


# --------------------------
# Load Data
# --------------------------
def load_data(sample_size=None, shuffle=True):
    """
    Load dataset with optional sampling

    Parameters:
        sample_size (int): number of rows to sample
        shuffle (bool): shuffle before sampling

    Returns:
        DataFrame
    """
    global _cached_df

    # 🔥 load once (cache)
    if _cached_df is None:
        try:
            print("📥 Loading dataset from URL...")
            df = pd.read_csv(DATASET_URL, names=COLUMN_NAMES)

            # حذف مقادیر خراب
            df = df.dropna()

            # تبدیل نوع داده (خیلی مهم برای مدل‌ها)
            df['rating'] = df['rating'].astype(float)

            _cached_df = df

            print(f"✅ Dataset loaded: {df.shape}")

        except Exception as e:
            raise Exception(f"خطا در لود دیتاست: {str(e)}")

    df = _cached_df

    # --------------------------
    # Sampling
    # --------------------------
    if sample_size:

        # جلوگیری از بیشتر شدن از کل دیتا
        sample_size = min(sample_size, len(df))

        # 🔥 shuffle قبل از sample (خیلی مهم)
        if shuffle:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.iloc[:sample_size]

        print(f"📊 Using sample size: {len(df_sample)}")

        return df_sample.copy()

    return df.copy()


# --------------------------
# EDA
# --------------------------
def get_eda_stats(df):
    """Return EDA statistics as a dict"""

    sorted_ratings = np.sort(df['rating'].unique()).tolist()

    top_users = (
        df.groupby('user_id').size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(10)
        .to_dict('records')
    )

    top_products = (
        df.groupby('product_id').size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(10)
        .to_dict('records')
    )

    rating_dist = (
        df.groupby('rating').size()
        .reset_index(name='count')
        .to_dict('records')
    )

    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'unique_ratings': sorted_ratings,
        'top_users': top_users,
        'top_products': top_products,
        'rating_distribution': rating_dist,
        'sample': df.head(5).to_dict('records'),
    }