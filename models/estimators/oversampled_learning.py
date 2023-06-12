import numpy as np
from sklearn import clone
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from joblib import Parallel, delayed

class OLearner():
    def __init__(self, cate_estimator, n_runs=10, t_frac=1.0, n_jobs=1):
        self.n_runs = n_runs
        self.t_frac = t_frac
        self.n_jobs = n_jobs
        self.est_all = [clone(cate_estimator, safe=False) for _ in range(self.n_runs)]

    def fit(self, X, t, y):
        c_idx = np.where(t == 0)[0]
        t_idx = np.where(t == 1)[0]

        if self.t_frac > 1.0:
            t_size = int(self.t_frac)
        else:
            t_size = int(len(t_idx) * self.t_frac)
        sample_size = np.min([len(c_idx), len(t_idx), t_size])

        def _balanced_fit(id):
            c_idx_batch = np.random.choice(c_idx, size=sample_size, replace=False)
            t_idx_batch = np.random.choice(t_idx, size=sample_size, replace=False)

            X_sample = np.concatenate([X[c_idx_batch], X[t_idx_batch]])
            t_sample = np.concatenate([t[c_idx_batch], t[t_idx_batch]])
            y_sample = np.concatenate([y[c_idx_batch], y[t_idx_batch]])

            self.est_all[id].fit(Y=y_sample, T=t_sample, X=X_sample)

        Parallel(n_jobs=1)(delayed(_balanced_fit)(i) for i in range(self.n_runs))

    def score(self, X, t, y):
        scores = [est.score(X, t, y) for est in self.est_all]
        return np.mean(scores)

    def cate(self, X, mode='mean'):
        preds = np.zeros((X.shape[0], self.n_runs))
        for i, est in enumerate(self.est_all):
            preds[:, i] = est.effect(X)

        # Could also return std dev/err.
        if mode == 'mean':
            return np.mean(preds, axis=1)
        elif mode == 'median':
            return np.median(preds, axis=1)

class OLearnerCV():
    def __init__(self, cate_estimator, cv=5, n_jobs=1):
        self.cate_estimator = cate_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        #self.params = {'n_runs': [1000], 't_frac': [0.8, 0.85, 0.9, 0.95, 1.0]}
        self.params = {'n_runs': [10, 100], 't_frac': [64, 128, 256, 512]}
        self.param_grid = list(ParameterGrid(self.params))
    
    def fit(self, X, t, y):
        grid_scores = []
        for param in self.param_grid:
            scores = []
            skf = StratifiedKFold(n_splits=self.cv)
            for train_idx, test_idx in skf.split(X, t):
                X_train, X_test = X[train_idx], X[test_idx]
                t_train, t_test = t[train_idx], t[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                est = OLearner(self.cate_estimator, n_runs=param['n_runs'], t_frac=param['t_frac'], n_jobs=self.n_jobs)
                est.fit(X_train, t_train, y_train)
                scores.append(est.score(X_test, t_test, y_test))

            grid_scores.append(np.mean(scores))

        # Re-fit the best params on all data
        best_idx = np.argmax(grid_scores)
        self.best_params_ = self.param_grid[best_idx]
        self.best_estimator_ = OLearner(self.cate_estimator, n_runs=self.best_params_['n_runs'], t_frac=self.best_params_['t_frac'], n_jobs=self.n_jobs)
        self.best_estimator_.fit(X, t, y)
        
    def cate(self, X, mode='mean'):
        return self.best_estimator_.cate(X, mode)