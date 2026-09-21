import threading
import os

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .ml.data_loader import load_data, get_eda_stats
from .ml.svd_model import train_svd, predict_svd, load_svd_model
from .ml.als_model import train_als, als_is_trained, predict_als


# --------------------------
# وضعیت آموزش
# --------------------------
_training_status = {
    'svd': 'idle',
    'als': 'idle'
}

_training_results = {
    'svd': None,
    'als': None
}


# --------------------------
# صفحه اصلی
# --------------------------
def index(request):
    svd_ready = load_svd_model() is not None
    als_ready = als_is_trained()

    return render(request, 'recommender/index.html', {
        'svd_trained': svd_ready,
        'als_trained': als_ready,
    })


# --------------------------
# EDA
# --------------------------
def eda_view(request):
    sample_size = int(request.GET.get('sample', 500000))

    df = load_data(sample_size=sample_size)
    stats = get_eda_stats(df)

    return render(request, 'recommender/eda.html', {
        'stats': stats,
        'sample_size': sample_size,
    })


# --------------------------
# TRAIN
# --------------------------
@csrf_exempt
def train_view(request):

    if request.method == 'POST':

        model_type = request.POST.get('model_type', 'svd')
        sample_size = int(request.POST.get('sample_size', 500000))

        if _training_status.get(model_type) == 'training':
            return JsonResponse({
                'status': 'busy',
                'message': 'مدل در حال آموزش است...'
            })

        def do_train():
            _training_status[model_type] = 'training'

            try:
                df = load_data(sample_size=sample_size)

                if len(df) == 0:
                    raise Exception("دیتا خالی است!")

                # -------- SVD --------
                if model_type == 'svd':
                    model, rmse, sample = train_svd(df)

                    _training_results['svd'] = {
                        'rmse': round(rmse, 4),
                        'sample': sample,
                        'model': 'SVD',
                        'data_size': len(df)
                    }

                # -------- ALS --------
                elif model_type == 'als':
                    rmse, sample = train_als(df)

                    _training_results['als'] = {
                        'rmse': round(rmse, 4),
                        'sample': sample,
                        'model': 'ALS',
                        'data_size': len(df)
                    }

                _training_status[model_type] = 'done'

            except Exception as e:
                _training_status[model_type] = f'error: {str(e)}'

        t = threading.Thread(target=do_train, daemon=True)
        t.start()

        return JsonResponse({
            'status': 'started',
            'model': model_type,
            'sample_size': sample_size
        })

    return render(request, 'recommender/train.html', {
        'svd_result': _training_results.get('svd'),
        'als_result': _training_results.get('als'),
        'svd_status': _training_status.get('svd'),
        'als_status': _training_status.get('als'),
    })


# --------------------------
# وضعیت آموزش
# --------------------------
def train_status(request):

    model_type = request.GET.get('model', 'svd')

    return JsonResponse({
        'status': _training_status.get(model_type, 'idle'),
        'result': _training_results.get(model_type)
    })


# --------------------------
# پیش‌بینی (ساده شده)
# --------------------------
@csrf_exempt
def predict_view(request):

    prediction = None
    error = None
    pred_int = None

    user_id = ''
    product_id = ''
    model_type = 'svd'

    # وضعیت مدل‌ها
    svd_ready = load_svd_model() is not None
    als_ready = als_is_trained()

    if request.method == 'POST':

        user_id = request.POST.get('user_id', '').strip()
        product_id = request.POST.get('product_id', '').strip()
        model_type = request.POST.get('model_type', 'svd')

        if not user_id or not product_id:
            error = 'هر دو فیلد را وارد کنید.'

        else:
            try:
                # -------- SVD --------
                if model_type == 'svd':
                    if not svd_ready:
                        error = 'مدل SVD موجود نیست.'
                    else:
                        prediction, error = predict_svd(user_id, product_id)

                # -------- ALS --------
                elif model_type == 'als':
                    if not als_ready:
                        error = 'مدل ALS یافت نشد.'
                    else:
                        prediction, error = predict_als(user_id, product_id)

                else:
                    error = 'مدل نامعتبر است.'

                if prediction is not None:
                    pred_int = int(round(prediction))

            except Exception as e:
                error = str(e)

    df = load_data(sample_size=1000)

    return render(request, 'recommender/predict.html', {
        'prediction': prediction,
        'pred_int': pred_int,
        'error': error,
        'user_id': user_id,
        'product_id': product_id,
        'model_type': model_type,
        'sample_users': df['user_id'].unique()[:5].tolist(),
        'sample_products': df['product_id'].unique()[:5].tolist(),
        'svd_trained': svd_ready,
        'als_trained': als_ready,
    })


# --------------------------
# مقایسه
# --------------------------
def compare_view(request):
    return render(request, 'recommender/compare.html', {
        'svd_result': _training_results.get('svd'),
        'als_result': _training_results.get('als'),
    })