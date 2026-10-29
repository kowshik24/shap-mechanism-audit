"""
Score new machine-sensor readings with the trained failure model.

Loads the artifacts written by experiments/01_benchmark.py (models/preprocessor.joblib,
models/xgboost_tuned.joblib, models/metadata.joblib) and applies the same feature engineering as
training before predicting.

Usage:
    python src/predict.py --csv path/to/new_readings.csv
    python src/predict.py --csv path/to/new_readings.csv --out predictions.csv

The input CSV needs the raw columns: Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min]
"""
import argparse

import pdm_audit  # noqa: F401  (threading env vars)
import joblib
import pandas as pd

from pdm_audit.config import MODELS_DIR
from pdm_audit.data import add_features


def load_artifacts():
    return (joblib.load(MODELS_DIR / 'preprocessor.joblib'), joblib.load(MODELS_DIR / 'xgboost_tuned.joblib'),
            joblib.load(MODELS_DIR / 'metadata.joblib'))


def predict(df_raw, preprocessor, model, metadata):
    X = preprocessor.transform(add_features(df_raw)[metadata['feature_cols']])
    proba = model.predict_proba(X)[:, 1]
    result = df_raw.copy()
    result['failure_probability'] = proba
    result['predicted_failure'] = (proba >= metadata['threshold']).astype(int)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', required=True, help='CSV of new readings to score')
    parser.add_argument('--out', default=None, help='where to write predictions (defaults to stdout)')
    args = parser.parse_args()

    result = predict(pd.read_csv(args.csv), *load_artifacts())
    if args.out:
        result.to_csv(args.out, index=False)
        print(f'wrote {len(result)} predictions to {args.out}')
    else:
        print(result.to_string(index=False))


if __name__ == '__main__':
    main()
