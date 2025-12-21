import numpy as np
import math

class MergeSortRunTimeModel:
    baseline = 13.9805
    time_per_comparison = 1.0980
    time_per_move = 0.5195
    sigma = 69.8
    relative_sigma2 = 0.1453

    def create_virtual_slots(lst):
        nr_levels = math.ceil(math.log2(len(lst))) + 1
        virtual_slots = [None] * nr_levels
        nr_elements = len(lst)

        sublist = [-1] * nr_elements
        virtual_slots[-1] = [sublist]

        for l in range(nr_levels - 1, 0, -1):
            level = []
            for s in range(len(virtual_slots[l])):
                length_sublist1 = math.ceil(len(virtual_slots[l][s]) / 2)
                length_sublist2 = math.floor(len(virtual_slots[l][s]) / 2)

                sublist1 = [-1] * length_sublist1
                sublist2 = [-1] * length_sublist2

                if len(sublist1) > 0:
                    level.append(sublist1)
                if len(sublist2) > 0:
                    level.append(sublist2)
            virtual_slots[l - 1] = level

        # Fill the bottom level with the actual list values
        virtual_slots[0] = [[item] for item in lst]
        return virtual_slots

    def nr_merge_sort_moves(lst):
        virtual_slots = MergeSortRunTimeModel.create_virtual_slots(lst)
        nr_levels = len(virtual_slots)
        move_ind = 0
        nr_comparisons = 0

        for l in range(1, nr_levels):
            nr_sublists = len(virtual_slots[l])
            nr_sublists_used = 0
            for s in range(nr_sublists):
                start_ind = nr_sublists_used
                nr_elements = len(virtual_slots[l][s])
                pointer1 = [l - 1, start_ind, 0]
                length_subsublist1 = len(virtual_slots[l - 1][start_ind])

                if nr_elements > len(virtual_slots[l - 1][start_ind]):
                    pointer2_idx = min(len(virtual_slots[l - 1]) - 1, start_ind + 1)
                    pointer2 = [l - 1, pointer2_idx, 0]
                    length_subsublist2 = len(virtual_slots[l - 1][pointer2_idx])
                else:
                    pointer2 = pointer1.copy()
                    length_subsublist2 = length_subsublist1

                used_up_sublist1 = False
                used_up_sublist2 = False

                virtual_slots[l][s] = [-1] * nr_elements
                for e in range(nr_elements):
                    value1 = virtual_slots[pointer1[0]][pointer1[1]][pointer1[2]]

                    if nr_elements > 1 and not used_up_sublist1 and not used_up_sublist2:
                        nr_comparisons += 1

                    if nr_elements > 1 and not used_up_sublist2:
                        value2 = virtual_slots[pointer2[0]][pointer2[1]][pointer2[2]]

                        if (not used_up_sublist1 and (value1 <= value2 or used_up_sublist2)):
                            virtual_slots[l][s][e] = value1
                            if pointer1[2] < len(virtual_slots[l - 1][start_ind]) - 1:
                                pointer1[2] += 1
                            else:
                                used_up_sublist1 = True
                                nr_sublists_used += 1
                        else:
                            virtual_slots[l][s][e] = value2
                            if pointer2[2] < length_subsublist2 - 1:
                                pointer2[2] += 1
                            else:
                                used_up_sublist2 = True
                                nr_sublists_used += 1
                    else:
                        virtual_slots[l][s][e] = value1
                        if nr_elements == 1:
                            used_up_sublist1 = True
                            used_up_sublist2 = True
                            nr_sublists_used += 1
                        else:
                            if pointer1[2] < length_subsublist1 - 1:
                                pointer1[2] += 1
                            else:
                                used_up_sublist1 = True
                                nr_sublists_used += 1

                    move_ind += 1

        nr_moves = move_ind
        return nr_comparisons, nr_moves

    def predict_rt(lst):
        nr_comparisons, nr_moves = MergeSortRunTimeModel.nr_merge_sort_moves(lst)
        RT = (MergeSortRunTimeModel.baseline +
              MergeSortRunTimeModel.time_per_comparison * nr_comparisons +
              MergeSortRunTimeModel.time_per_move * nr_moves)
        return RT

    def simulate_rt(lst):
        expected_RT = MergeSortRunTimeModel.predict_rt(lst)
        RT = expected_RT + np.sqrt(expected_RT ** 2 * MergeSortRunTimeModel.relative_sigma2) * np.random.randn()
        return RT
