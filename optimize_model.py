import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from os.path import join, isfile
from os import listdir

all_params = {'randomForest': {'random_forest__max_depth': [40, 50, 100],
                               'random_forest__min_samples_split': np.arange(2, 8, 2),
                               'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
                               },
              'poly': {'transformer__poly__degree': [1, 2, 3, 4],
                       'transformer__poly__interaction_only': [True, False],
                       'transformer__poly__include_bias': [True, False], },
              'mpg_discretizer': {'transformer__mpg_discretizer__n_bins': [5, 6, 10, 13],
                                  'transformer__mpg_discretizer__encode': ['onehot', 'ordinal'],
                                  'transformer__mpg_discretizer__strategy': ['uniform', 'quantile', 'kmeans'],
                                  },
              'tax_discretizer': {'transformer__tax_discretizer__n_bins': [7, 8, 9, 10],
                                  'transformer__tax_discretizer__encode': ['onehot', 'ordinal'],
                                  'transformer__tax_discretizer__strategy': ['uniform', 'quantile', 'kmeans'], },
              'engine_size_discretizer': {'transformer__engine_size_discretizer__n_bins': [2, 3, 4, 6, 9],
                                          'transformer__engine_size_discretizer__encode': ['onehot', 'ordinal'],
                                          'transformer__engine_size_discretizer__strategy': ['uniform', 'quantile', 'kmeans'], },
              'year_pipe__discretize': {'transformer__year_pipe__discretize__n_bins': [3, 6, 9, 11],
                                        'transformer__year_pipe__discretize__encode': ['onehot', 'ordinal'],
                                        'transformer__year_pipe__discretize__strategy': ['uniform', 'quantile', 'kmeans']}}


def get_best_estimator(model, param_grid, X, y, scoring, verbose=False):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=verbose,
                        n_jobs=-1, pre_dispatch=1
                        )
    grid.fit(X, y)
    print(grid.best_params_)
    return grid.best_estimator_


def optimize(model, X_train, y_train):
    for key, param in all_params.items():
        print(f'Optimizing {key}')
        # *** GridSearchCV ***#
        mse = make_scorer(mean_squared_error, greater_is_better=False)
        model = get_best_estimator(
            model, param, X_train, y_train, scoring=mse)


if __name__ == "__main__":
    print('nothing')
    pass