from typing import List
def lst_permute(l: List, indices: List[int]):
    if (not len(l) == len(indices)
        or not all((i >= 0 and i < len(l)) for i in indices)
    ):
        raise ValueError(f"list length is {len(l)}, but indices are {indices}")
    return [l[i] for i in indices]


