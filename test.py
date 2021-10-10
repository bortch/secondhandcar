from itertools import combinations, product, permutations

masks = product(range(2), repeat=3)

D = {"key_a": "AAA", "key_b": "BBB", "key_c": "CCC"}
print(masks)


def gen(masks, D):
    for mask in masks:
        empty_combination = {}
        filled_combination = {}
        # print('\nmask', mask)
        for index, key in enumerate(D):
            if mask[index] < 1:
                empty_combination[key] = '_'
            else:
                filled_combination[key] = D[key]

        if sum(list(mask)) == 0:
            # no permutation
            combination = {**filled_combination, **empty_combination}
            yield empty_combination
        else:
            # print('filled_combination', filled_combination)
            for p in permutations(filled_combination):
                c = {}
                # print('permutation', p)
                for i in p:
                    # print(i, filled_combination[i])
                    c[i] = filled_combination[i]
                combination = {**c, **empty_combination}
                yield combination


# for x in gen(masks, D):
#     print(x)


params = {'a': "A",'e':'E', 'b': "B", 't': "T", 'i':'I'}
challenge = "BAT"
best_solution = ''
for p in permutations(params.keys()):
  print(f'testing permutation:{p}')
  permutation_solution = ''
  combination={}
  for key in p:
    combination[key]=params[key]
  for key, param in combination.items():
      #print(f"key: {key}, params: {param}")
      solution = permutation_solution+param
      print(f"\ntrying {solution}")
      if solution == challenge:
        permutation_solution = solution
        break
      elif solution in challenge[:len(solution)]:
          permutation_solution = solution
      else:
          permutation_solution=''
  if permutation_solution == challenge:
    break
print(best_solution)




# opel - Best score 678.730236472058
# opel - Best params {'transformer__poly': 'passthrough', 'transformer__mpg_pipe': 'passthrough', 'transformer__tax_pipe': {'transformer__tax_pipe__discretize__encode': 'onehot', 'transformer__tax_pipe__discretize__n_bins': 10, 'transformer__tax_pipe__discretize__strategy': 'uniform'}, 'transformer__engine_size_pipe': 'passthrough', 'transformer__year_pipe': {'transformer__year_pipe__discretize__encode': 'ordinal', 'transformer__year_pipe__discretize__n_bins': 11, 'transformer__year_pipe__discretize__strategy': 'quantile'}}
# Best Model saved @ ./model/opel_optimize_rmse_679.joblib

# toyota - Best score 918.9770768190461
# toyota - Best params {'transformer__poly': 'passthrough', 'transformer__mpg_pipe': 'passthrough', 'transformer__tax_pipe': 'passthrough', 'transformer__engine_size_pipe': 'passthrough', 'transformer__year_pipe': 'passthrough'}
# Best Model saved @ ./model/toyota_optimize_rmse_919.joblib

# audi - Best score 1841.8677762649354
# audi - Best params {'transformer__poly': {'transformer__poly__degree': 1, 'transformer__poly__include_bias': False, 'transformer__poly__interaction_only': False}, 'transformer__mpg_pipe': 'passthrough', 'transformer__tax_pipe': 'passthrough', 'transformer__engine_size_pipe': 'passthrough', 'transformer__year_pipe': 'passthrough'}
# Best Model saved @ ./model/audi_optimize_rmse_1842.joblib

# bmw - Best score 1634.6243531686475
# bmw - Best params {}
# Best Model saved @ ./model/bmw_optimize_rmse_1635.joblib

# hyundai - Best score 732.686608913161
# hyundai - Best params {}
# Best Model saved @ ./model/hyundai_optimize_rmse_733.joblib

# skoda - Best score 1427.7879745356845
# skoda - Best params {'transformer__poly': 'passthrough', 'transformer__mpg_pipe': {'transformer__mpg_pipe__discretize__encode': 'ordinal', 'transformer__mpg_pipe__discretize__n_bins': 10, 'transformer__mpg_pipe__discretize__strategy': 'quantile'}, 'transformer__tax_pipe': 'passthrough', 'transformer__engine_size_pipe': 'passthrough', 'transformer__year_pipe': 'passthrough'}
# Best Model saved @ ./model/skoda_optimize_rmse_1428.joblib

# ford - Best score 801.2504473522818
# ford - Best params {'transformer__poly': 'passthrough', 'transformer__mpg_pipe': 'passthrough', 'transformer__tax_pipe': 'passthrough', 'transformer__engine_size_pipe': 'passthrough', 'transformer__year_pipe': 'passthrough'}
# Best Model saved @ ./model/ford_optimize_rmse_801.joblib

# mercedes - Best score 2427.3463228889445
# mercedes - Best params {'transformer__poly': {'transformer__poly__degree': 2, 'transformer__poly__include_bias': True, 'transformer__poly__interaction_only': False}, 'transformer__mpg_pipe': 'passthrough', 'transformer__tax_pipe': {'transformer__tax_pipe__discretize__encode': 'onehot', 'transformer__tax_pipe__discretize__n_bins': 8, 'transformer__tax_pipe__discretize__strategy': 'quantile'}, 'transformer__engine_size_pipe': 'passthrough', 'transformer__year_pipe': 'passthrough'}
# Best Model saved @ ./model/mercedes_optimize_rmse_2427.joblib

# vw - Best score 1161.6773564640544
# vw - Best params {}
# Best Model saved @ ./model/vw_optimize_rmse_1162.joblib

