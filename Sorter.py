from AlgorithmExecuter import AlgorithmExecuter

# Standard Python implementations for sorting algorithms

def cocktail_sort(arr):
    """
    Perform a cocktail sort on the given array.
    
    Cocktail sort is a variation of bubble sort that sorts in both directions on each pass through the list.
    
    :param arr: List of elements to be sorted
    :return: Sorted list

    Time Complexity: O(n^2) in the worst case, O(n) in the best case
    Space Complexity: O(1) since it sorts in place
    """
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        # reset the swapped flag
        swapped = False
        
        # Forward pass (bubble sort)
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        # if no swap occurs, then the array is sorted.
        if not swapped:
            break
        
        # Decrease the end point
        end -= 1
        
        swapped = False

        # Backward pass moving from end to start
        for i in range(end, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        # Increase the start point
        start += 1

    return arr


def merge_sort(arr):
    """
    Perform a merge sort on the given array.
    
    Merge sort is a divide-and-conquer algorithm that sorts an array by recursively splitting it into halves,
    sorting each half, and then merging the sorted halves back together.
    
    :param arr: List of elements to be sorted
    :return: Sorted list

    Time Complexity: O(n log n) in all cases
    Space Complexity: O(n) due to the temporary arrays used for merging
    """
    def merge(left, right):
        merged = []
        i, j = 0, 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        # Append any remaining elements from either half
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged
    
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    return merge(left_half, right_half)


def quick_sort(data):
    arr = list(data)
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)
    
"""
def bogo_sort(data):
    arr = list(data)
    while not all(arr[i] <= arr[i+1] for i in range(len(arr)-1)):
        random.shuffle(arr)
    return arr

def insertion_sort(data):
    arr = list(data)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def selection_sort(data):
    arr = list(data)
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def bubble_sort(data):
    arr = list(data)
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def shell_sort(data):
    arr = list(data)
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(data):
    arr = list(data)
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr
"""

class Sorter(AlgorithmExecuter):
    """
    Sorting algorithm executer.
    """
    def __init__(self, algorithms=None):
        if algorithms is None:
            """
            algorithms = [
                bogo_sort,
                insertion_sort,
                selection_sort,
                bubble_sort,
                shell_sort,
                heap_sort,
                merge_sort,
                quick_sort
            ]
            """
            algorithms = [cocktail_sort, merge_sort]
        super().__init__(algorithms) 