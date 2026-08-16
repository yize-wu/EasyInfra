from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .chunk import ChunkBlock
from easyinfra.utils.stats import show_rank_print

from typing import List
def ids_of_chunk_list(chunk_list: List[ChunkBlock]):
    return [chunk.chunk_id for chunk in chunk_list]

def pop_elements_of_list(lst: List, indices: List[int] = None):
    '''
        If indices is None, pop all elements.
    '''
    
    length = len(lst)
    if length == 0:
        return
    
    index_max = length - 1 # >= 0
    if indices is None:
        real_indices = [_ for _ in range(len(lst)-1, -1, -1)]
    else:
        if any(i > index_max for i in indices):
            raise ValueError(f"lst length is {index_max+1}, index out of range {indices}")
        real_indices = sorted(indices, reverse=True)
    
    # show_rank_print(f"lst length is {index_max+1}, indices: {real_indices}")
    for i in real_indices:
        lst.pop(i)
    return lst