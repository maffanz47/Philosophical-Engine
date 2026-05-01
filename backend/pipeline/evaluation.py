from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def compute_metrics(y_true, y_pred, task='classification'):
    if task == 'classification':
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average='weighted')
        }
    elif task == 'regression':
        return {"rmse": mean_squared_error(y_true, y_pred, squared=False)}
    else:
        return {}