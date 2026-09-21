# ===============================
# ALS MODEL (FINAL VERSION)
# ===============================

import os
import sys

_als_model = None
_spark = None
_user_indexer_model = None
_product_indexer_model = None


# ===============================
# Spark Session
# ===============================
def _get_spark():
    global _spark

    if _spark is None:
        if sys.platform == 'win32':
            os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
            os.environ.setdefault('PYSPARK_DRIVER_PYTHON', sys.executable)

        from pyspark.sql import SparkSession

        _spark = (
            SparkSession.builder
            .master('local[2]')
            .appName('RecommenderSystem')
            .config('spark.driver.memory', '1g')
            .config('spark.ui.enabled', 'false')
            .getOrCreate()
        )

        _spark.sparkContext.setLogLevel('ERROR')

    return _spark


# ===============================
# TRAIN (Grid Search)
# ===============================
def train_als(pandas_df, test_size=0.2):
    global _als_model, _user_indexer_model, _product_indexer_model

    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator
    import itertools

    spark = _get_spark()
    spark_df = spark.createDataFrame(pandas_df)

    # Indexing
    user_indexer = StringIndexer(inputCol='user_id', outputCol='user_index')
    product_indexer = StringIndexer(inputCol='product_id', outputCol='product_index')

    _user_indexer_model = user_indexer.fit(spark_df)
    df = _user_indexer_model.transform(spark_df)

    _product_indexer_model = product_indexer.fit(df)
    df = _product_indexer_model.transform(df)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    evaluator = RegressionEvaluator(labelCol='rating', metricName='rmse')

    # Grid Search Params
    ranks = [5, 10, 20]
    max_iters = [5, 10, 15]
    reg_params = [0.01, 0.1, 1.0]

    best_rmse = float("inf")
    best_model = None
    best_params = None

    print("🚀 Grid Search Started")

    for rank, max_iter, reg in itertools.product(ranks, max_iters, reg_params):
        print(f"Testing → rank={rank}, maxIter={max_iter}, reg={reg}")

        als = ALS(
            userCol='user_index',
            itemCol='product_index',
            ratingCol='rating',
            coldStartStrategy='drop',
            rank=rank,
            maxIter=max_iter,
            regParam=reg
        )

        model = als.fit(train)
        predictions = model.transform(test)

        rmse = evaluator.evaluate(predictions)
        print(f"RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (rank, max_iter, reg)

    _als_model = best_model

    print("✅ Best Model:")
    print(f"RMSE: {best_rmse}")
    print(f"Params: rank={best_params[0]}, maxIter={best_params[1]}, reg={best_params[2]}")

    # Sample
    predictions = _als_model.transform(test)

    sample_rows = predictions.select(
        'user_id', 'product_id', 'rating', 'prediction'
    ).limit(5).collect()

    sample = [
        {
            'user_id': r['user_id'],
            'product_id': r['product_id'],
            'actual': r['rating'],
            'predicted': round(float(r['prediction']), 2)
        }
        for r in sample_rows
    ]

    return round(best_rmse, 4), sample


# ===============================
# PREDICT
# ===============================
def predict_als(user_id, product_id):
    global _als_model, _user_indexer_model, _product_indexer_model

    try:
        if _als_model is None:
            return None, "مدل آموزش داده نشده"

        spark = _get_spark()

        user_df = spark.createDataFrame([(user_id,)], ['user_id'])
        user_df = _user_indexer_model.transform(user_df)

        item_df = spark.createDataFrame([(product_id,)], ['product_id'])
        item_df = _product_indexer_model.transform(item_df)

        df = user_df.crossJoin(item_df).select('user_index', 'product_index')

        pred = _als_model.transform(df).collect()[0]['prediction']

        if pred is None:
            return None, "cold start"

        return float(pred), None

    except Exception as e:
        return None, str(e)


# ===============================
# CHECK
# ===============================
def als_is_trained():
    return _als_model is not None