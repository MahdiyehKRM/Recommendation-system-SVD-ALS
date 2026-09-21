"""
Hybrid recommender system combining SVD and ALS
(Final Clean Version)
"""

from .svd_model import predict_svd
from .als_model import predict_als


SVD_WEIGHT = 0.7
ALS_WEIGHT = 0.3


# --------------------------
# SINGLE PREDICTION
# --------------------------
def hybrid_predict(user_id, product_id):
    """
    Combine SVD and ALS predictions
    خروجی: فقط prediction (float یا None)
    """

    try:
        # SVD → (pred, error)
        svd_pred, _ = predict_svd(user_id, product_id)

        # ALS → فقط pred
        als_pred = predict_als(user_id, product_id)

        # اگر هیچکدوم نبود
        if svd_pred is None and als_pred is None:
            return None

        # fallback
        if svd_pred is None:
            return als_pred

        if als_pred is None:
            return svd_pred

        # ترکیب وزنی
        final_pred = (SVD_WEIGHT * svd_pred) + (ALS_WEIGHT * als_pred)

        return round(final_pred, 2)

    except Exception as e:
        print("🔥 Hybrid Error:", e)
        return None


# --------------------------
# BATCH PREDICTION
# --------------------------
def hybrid_batch_predict(df):

    results = []

    for _, row in df.iterrows():

        pred = hybrid_predict(row['user_id'], row['product_id'])

        results.append({
            'user_id': row['user_id'],
            'product_id': row['product_id'],
            'actual': row.get('rating', None),
            'predicted': pred
        })

    return results


# --------------------------
# EVALUATION
# --------------------------
def evaluate_hybrid(df):

    from math import sqrt
    from sklearn.metrics import mean_squared_error

    y_true = []
    y_pred = []

    for _, row in df.iterrows():

        if 'rating' not in row:
            continue

        pred = hybrid_predict(row['user_id'], row['product_id'])

        if pred is not None:
            y_true.append(row['rating'])
            y_pred.append(pred)

    if len(y_true) == 0:
        return None

    rmse = sqrt(mean_squared_error(y_true, y_pred))

    return round(rmse, 4)