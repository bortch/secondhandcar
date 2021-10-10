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