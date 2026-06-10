import json

from model_training import train_models


if __name__ == '__main__':
    results, best_name, _ = train_models()
    print('Model comparison results:')
    for name, metrics in results.items():
        print(f"{name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    print('\nBest model selected:', best_name)

    with open('models/flight_fare_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'best_model': best_name, 'results': results}, f, indent=2)
