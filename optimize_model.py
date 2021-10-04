import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
#from os.path import join, isfile
#from os import listdir
from itertools import combinations, product, permutations
from sklearn.model_selection import StratifiedKFold

all_params = {
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

estimator_params = {
    'random_forest': {'random_forest__max_depth': [40, 50, 100],
                      'random_forest__min_samples_split': np.arange(2, 10, 2),
                      'random_forest__max_features': ['auto', 'sqrt', 'log2', None],
                      }
}

def get_best_estimator(model, param_grid, X, y, scoring, verbose=False):
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=5, scoring=scoring,
                        verbose=verbose,
                        n_jobs=-1
                        )
    grid.fit(X, y)
    return grid.best_estimator_

def get_grid_search_results(model, param_grid, X, y, scoring, verbose=False):
    skf = StratifiedKFold(n_splits=5,random_state=1, shuffle=True)
    grid = GridSearchCV(model, param_grid=param_grid,
                        cv=skf, scoring=scoring,
                        verbose=verbose,
                        n_jobs=-1
                        )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

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
            grid = GridSearchCV(model, param_grid=param,
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


def evaluate_combinations(model, params, X_train, y_train, verbose=False):
    if verbose:
        print("\nEvaluate params combination")
    best_params = {}
    best_model = model
    best_score = np.Inf
    to_skip_previous_param_key = []
    # get a permutation of params combination
    for permutation in permutations(params.keys()):
        if verbose:
            print(f'\nNew permutation\n')
            print(f"{permutation}\n")
        # build permutated combination
        combination = {}
        for key in permutation:
            combination[key] = params[key]
        # init score to challenge
        combination_score = best_score
        combination_model = best_model
        combination_params = best_params
        # print(f'Combination to evaluate {combination}')
        # evaluate params one at times:
        for key, param in combination.items():
            # skip the evaluation:
            # if params belong to current solution
            if key in best_params.keys():
                continue
            # if it still the same param as in the previous permutation
            if key in to_skip_previous_param_key:
                # print("try to evaluate last bad param")
                break
            to_skip_previous_param_key.clear()
            if verbose:
                print(f"\nSearch best param for '{key}'")
            # build the current grid_param to be optimized
            param_grid={}
            if param == 'passthrough':
                param_grid[key]= [param]
            else:
                for p in param:
                    param_grid[p]=param[p]

            for key, param in combination_params.items():
                if isinstance(param, list):
                    param_grid[key]=param
                else:
                    param_grid[key]= [param]
            if verbose:
                print('Current Param Grid to evaluate:',param_grid)
            mse = make_scorer(mean_squared_error, greater_is_better=False)
            grid = GridSearchCV(model, param_grid=param_grid,
                                cv=5, scoring=mse,
                                verbose=verbose,
                                n_jobs=-1,
                                # pre_dispatch=1
                                )
            grid.fit(X_train, y_train)
            # if it is a better score
            # keep track of the optimized parameters
            if combination_score >= grid.best_score_:
                if verbose:
                    print(f'current score {combination_score}, better solution found:{grid.best_score_}')
                combination_model = grid.best_estimator_
                combination_params = {**combination_params,**grid.best_params_}
                combination_score = grid.best_score_
                #to_skip_previous_param_key.pop(key)
            else:
                # else itisn't a good start
                if verbose:
                    print(f"Add {key} to be skipped with param: \n{param}\n")
                to_skip_previous_param_key.append(key)
                break
        #print('Best Score',best_score,'\n',best_params[key])
        if combination_score <= best_score:
            best_model = combination_model
            best_score = combination_score
            best_params = combination_params
    return best_model, best_params, best_score


# def get_permutation_of_combinations_of_params():
#     nb_params = len(all_params)
#     masks = product(range(2), repeat=nb_params)

#     for mask in masks:
#         empty_combination = {}
#         filled_combination = {}
#         # build a combination
#         for index, key in enumerate(all_params):
#             if mask[index] < 1:
#                 empty_combination[key] = 'passthrough'
#             else:
#                 filled_combination[key] = all_params[key]
#         # produce permutations
#         if sum(list(mask)) == 0:
#             # no permutation if only "passthrough"
#             yield empty_combination
#         else:
#             for permutation in permutations(filled_combination):
#                 perm = {}
#                 for key in permutation:
#                     perm[key] = filled_combination[key]
#                 combination = {**perm, **empty_combination}
#                 yield combination

def get_combinations_of_params():
    nb_params = len(all_params)
    comb = {}
    mask = product(range(2), repeat=nb_params)
    for m in mask:
        for index, key in enumerate(all_params):
            if m[index] < 1:
                comb[key] = 'passthrough'
            else:
                comb[key] = all_params[key]
        yield comb


if __name__ == "__main__":
    for c in get_combinations_of_params():
        print(c, "\n")
