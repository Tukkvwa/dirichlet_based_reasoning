import random
import math

class CocktailSortRunTimeModel:
    # Constants
    baseline = 19.59
    time_per_comparison = 0.19
    time_per_move = 0.31
    sigma = 87.5
    relative_sigma2 = 0.2121

    def nr_comparisons(lst):
        swaps_from = []
        swaps_to = []
        nr_swaps = 0
        nr_comparisons = 0
        has_swapped = True

        while has_swapped:
            has_swapped = False
            # forward pass
            for i in range(len(lst) - 1):
                if lst[i] > lst[i + 1]:
                    swaps_from.append(i)
                    swaps_to.append(i + 1)
                    lst[i], lst[i + 1] = lst[i + 1], lst[i]
                    nr_swaps += 1
                    has_swapped = True
                nr_comparisons += 1
            # backward pass
            for i in range(len(lst) - 1, 0, -1):
                if lst[i] < lst[i - 1]:
                    swaps_from.append(i)
                    swaps_to.append(i - 1)
                    lst[i], lst[i - 1] = lst[i - 1], lst[i]
                    nr_swaps += 1
                    has_swapped = True
                nr_comparisons += 1
        return nr_comparisons, nr_swaps

    def predict_rt(lst):
        nr_comparisons, nr_moves = CocktailSortRunTimeModel.nr_comparisons(lst)
        RT = (CocktailSortRunTimeModel.baseline +
              CocktailSortRunTimeModel.time_per_comparison * nr_comparisons +
              CocktailSortRunTimeModel.time_per_move * nr_moves)
        return RT

    def simulate_rt(lst):
        expected_rt = CocktailSortRunTimeModel.predict_rt(lst)
        RT = expected_rt + math.sqrt(expected_rt ** 2 * CocktailSortRunTimeModel.relative_sigma2) * random.gauss(0, 1)
        return RT
