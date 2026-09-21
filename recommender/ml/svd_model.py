"""
Surprise SVD recommender model with Grid Search.
"""
import joblib
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse as surprise_rmse

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svd_model.pkl')

_model = None


def train_svd(df, test_size=0.2, use_grid=True):
    """
    Train SVD with optional Grid Search.
    Returns (model, rmse_score, sample_predictions).
    """
    global _model

    # Step 1: Load data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[['user_id', 'product_id', 'rating']], reader
    )

    # Step 2: Split
    trainset, testset = train_test_split(
        data, test_size=test_size, random_state=42
    )

    # Step 3: Grid Search (اختیاری ولی خیلی مهم 🔥)
    if use_grid:
        param_grid = {
            'n_factors': [50, 100],
            'n_epochs': [20, 30],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.1]
        }

        gs = GridSearchCV(
            SVD,
            param_grid,
            measures=['rmse'],
            cv=3,
            n_jobs=-1
        )

        gs.fit(data)

        best_params = gs.best_params['rmse']
        model = SVD(**best_params, random_state=42)

    else:
        model = SVD(random_state=42)

    # Step 4: Train
    model.fit(trainset)

    # Step 5: Evaluate
    predictions = model.test(testset)
    rmse_score = surprise_rmse(predictions, verbose=False)

    # Step 6: Save model
    joblib.dump(model, MODEL_PATH)
    _model = model

    # Step 7: Sample predictions
    sample = []
    for p in predictions[:5]:
        sample.append({
            'user_id': p.uid,
            'product_id': p.iid,
            'actual': p.r_ui,
            'predicted': round(p.est, 2),
        })

    return model, round(rmse_score, 4), sample


def load_svd_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_svd(user_id, product_id):
    """Predict rating for a user-product pair."""
    model = load_svd_model()
    if model is None:
        return None, 'Model not trained yet. Please train first.'
    pred = model.predict(user_id, product_id)
    return round(pred.est, 2), None