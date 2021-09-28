import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from os.path import join, isfile
from os import listdir

all_params = {
    'random_forest': {'random_forest__max_depth': [40, 50, 100],
                      'random_forest__min_samples_split': np.arange(2, 8, 2),
                      'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
                      },
    'transformer__poly': {'transformer__poly__degree': [1, 2, 3],
                          'transformer__poly__interaction_only': [True, False],
                          'transformer__poly__include_bias': [True, False], },
    'transformer__mpg_pipe': {'transformer__mpg_pipe__discretize__n_bins': [6, 10],
                             'transformer__mpg_pipe__discretize__encode': ['onehot', 'ordinal'],
                             'transformer__mpg_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans'],
                             },
    'transformer__tax_pipe': {'transformer__tax_pipe__discretize__n_bins': [8, 9, 10],
                             'transformer__tax_pipe__discretize__encode': ['onehot', 'ordinal'],
                             'transformer__tax_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans'], },
    'transformer__engine_size_pipe': {'transformer__engine_size_pipe__discretize__n_bins': [2, 3, 4],
                                     'transformer__engine_size_pipe__discretize__encode': ['onehot', 'ordinal'],
                                     'transformer__engine_size_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans'], },
    'transformer__year_pipe': {'transformer__year_pipe__discretize__n_bins': [3, 10, 11],
                              'transformer__year_pipe__discretize__encode': ['onehot', 'ordinal'],
                              'transformer__year_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans']}}


def get_best_estimator(model, param_grid, X, y, scoring, verbose=False):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=verbose,
                        n_jobs=-1, pre_dispatch=1
                        )
    grid.fit(X, y)
    return grid.best_estimator_

def get_best_params(model, param_grid, X, y, scoring, verbose=False):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=verbose,
                        n_jobs=-1, 
                        #pre_dispatch=1
                        )
    grid.fit(X, y)
    print(grid.best_params_)
    return grid.best_params_

def optimize(model, X_train, y_train):
    best_params = {}
    for key, param in all_params.items():
        if model.get_params()[key]=='passthrough':
            print(f"{key} skipped: 'passthrough'")
        else:
            print(f'Optimizing {key}')
            # *** GridSearchCV ***#
            mse = make_scorer(mean_squared_error, greater_is_better=False)
            best_params['key']=get_best_params(
                model, param, X_train, y_train, scoring=mse, verbose=1)
    return best_params

if __name__ == "__main__":
    print('nothing')
    pass
