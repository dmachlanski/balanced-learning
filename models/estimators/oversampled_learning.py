import numpy as np
from sklearn import clone
from joblib import Parallel, delayed

class OLearner():
    def __init__(self, cate_estimator, n_runs=10, arm_sample_size=-1, n_jobs=1):
        self.n_runs = n_runs
        self.arm_sample_size = arm_sample_size
        self.n_jobs = n_jobs
        self.est_all = [clone(cate_estimator, safe=False) for _ in range(self.n_runs)]

    def fit(self, X, t, y):
        c_idx = np.where(t == 0)[0]
        t_idx = np.where(t == 1)[0]
        sample_size = np.min([len(c_idx), len(t_idx), self.arm_sample_size])

        def _balanced_fit(id):
            c_idx_batch = np.random.choice(c_idx, size=sample_size, replace=False)
            t_idx_batch = np.random.choice(t_idx, size=sample_size, replace=False)

            X_sample = np.concatenate([X[c_idx_batch], X[t_idx_batch]])
            t_sample = np.concatenate([t[c_idx_batch], t[t_idx_batch]])
            y_sample = np.concatenate([y[c_idx_batch], y[t_idx_batch]])

            self.est_all[id].fit(Y=y_sample, T=t_sample, X=X_sample)

        Parallel(n_jobs=self.n_jobs)(delayed(_balanced_fit)(i) for i in range(self.n_runs))

    def cate(self, X, mode='mean'):
        preds = np.zeros((X.shape[0], self.n_runs))
        for i, est in enumerate(self.est_all):
            preds[:, i] = est.effect(X)

        # Could also return std dev/err.
        if mode == 'mean':
            return np.mean(preds, axis=1)
        elif mode == 'median':
            return np.median(preds, axis=1)