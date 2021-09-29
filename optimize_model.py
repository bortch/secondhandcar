import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
#from os.path import join, isfile
#from os import listdir
from itertools import product, permutations

all_params = {
    # 'random_forest': {'random_forest__max_depth': [40, 50, 100],
    #                   'random_forest__min_samples_split': np.arange(2, 8, 2),
    #                   'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
    #                   },
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
                        # pre_dispatch=1
                        )
    grid.fit(X, y)
    if verbose:
        print(grid.best_params_)
    return grid.best_params_


# def optimize(model, X_train, y_train):
#     best_params = {}
#     for key, param in all_params.items():
#         if model.get_params()[key] == 'passthrough':
#             best_params[key] = 'passthrough'
#             #print(f"{key} skipped: 'passthrough'")
#         else:
#             #print(f'Optimizing {key}')
#             # *** GridSearchCV ***#
#             mse = make_scorer(mean_squared_error, greater_is_better=False)
#             best_params[key] = get_best_params(
#                 model, param, X_train, y_train, scoring=mse, verbose=False)
#     print(best_params)
#     return best_params


def evaluate_combination(model, params, X_train, y_train, verbose=False):
    if verbose:
        print("\nEvaluate params combination")
    best_params = {}
    model_ = model
    best_score = np.Inf
    for key, param in params.items():
        if verbose:
            print(f"\nSearch best param for '{key}'")
        # \nParams: {param}")
        best_params[key] = 'passthrough'
        if param == 'passthrough':
            best_params[key] = 'passthrough'
        else:
            mse = make_scorer(mean_squared_error, greater_is_better=False)
            grid = GridSearchCV(model_, param_grid=param,
                                cv=5, scoring=mse,
                                verbose=verbose,
                                n_jobs=-1,
                                # pre_dispatch=1
                                )
            grid.fit(X_train, y_train)
            # evaluate model
            #print(f"current score {best_score}, best score {grid.best_score_}")
            if best_score >= grid.best_score_:
                # if better score
                model_ = grid.best_estimator_
                best_params[key] = grid.best_params_
                best_score = grid.best_score_
        #print('Best Score',best_score,'\n',best_params[key])
    return model_, best_params, best_score


def get_combinations_of_params():
    nb_params = len(all_params)
    masks = product(range(2), repeat=nb_params)

    for mask in masks:
        empty_combination = {}
        filled_combination = {}
        # build a combination
        for index, key in enumerate(all_params):
            if mask[index] < 1:
                empty_combination[key] = 'passthrough'
            else:
                filled_combination[key] = all_params[key]
        # produce permutations
        if sum(list(mask)) == 0:
            # no permutation if only "passthrough"
            yield empty_combination
        else:
            for permutation in permutations(filled_combination):
                perm = {}
                for key in permutation:
                    perm[key] = filled_combination[key]
                combination = {**perm, **empty_combination}
                yield combination


if __name__ == "__main__":
    for c in get_combinations_of_params():
        print(c, "\n")
