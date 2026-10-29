"""SHAP failure-mechanism audit for the AI4I 2020 predictive maintenance dataset."""
import os

# numpy/scikit-learn/xgboost/torch each ship a threaded BLAS/OpenMP runtime; running them in one
# process segfaulted on the development machine. Must run before any of them is imported.
for _var in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
