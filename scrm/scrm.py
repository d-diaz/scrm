import os
import numpy as np
import heapq
from skimage.future import graph
from skimage.future.graph.graph_merge import (_revalidate_node_edges,
                                              _invalidate_edge,
                                              _rename_node)


def scrm(img, labels, dms, mas, mmu):
    """Applies Size-Constrained Region Merging.

    Parameters
    ----------
    img : arr
      image being segmented
    labels : arr
      initial (oversegmented) regions
    dms : int
      desired mean size of merged regions, in pixels
    mas : int
      maximum allowed size of merged regions, in pixels
    mmu : int
      minimum mappable unit, in pixels

    Returns
    -------
    regions : arr
      array of same shape as image, with each distinct region indicated by
      increasing integer values
    """
    rag = graph.rag_mean_color(img, labels, connectivity=1)
    regions = merge_size_constrained(labels, rag,
                                     dms, mas, mmu,
                                     rag_copy=False, in_place_merge=True,
                                     merge_func=merge_scrm,
                                     weight_func=weight_scrm,
                                     ).astype(np.int16)

    return regions


def merge_size_constrained(labels, rag, dms, mas, mmu,
                           rag_copy, in_place_merge,
                           merge_func, weight_func):
    """Perform Size-Constrained Region Merging on a RAG.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    dms : int
        Desired Mean Size of regions, in pixels.
    mas : int
        Maximum Allowed Size of regions, in pixels. Note: Not a hard cap.
    mmu : int
        Minimum Mappable Unit, minimum size of regions, in pixels.
    rag_copy : bool
        If set, the RAG copied before modifying.
    in_place_merge : bool
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge..
    merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.
    Returns
    -------
    out : ndarray
        The new labeled array.
    """
    if rag_copy:
        rag = rag.copy()

    edge_heap = []

    # a couple attributes we'll track to enforce a partial stopping criterion
    rag.graph.update({
        'num_ge_mmu': 0,  # number of regions >= mmu size
        'area_lt_mmu': 0,  # total area in regions smaller than mmu size
     })

    total_area = 0  # total area in regions/image

    for n in rag:
        area = rag.nodes[n]['pixel count']
        total_area += area
        if area < mmu:
            rag.graph['area_lt_mmu'] += area
        else:
            rag.graph['num_ge_mmu'] += 1

    exp_final_num = total_area // dms  # expected number of regions

    for n1, n2, data in rag.edges(data=True):
        # Push a valid edge in the heap
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)

        # Reference to the heap item in the graph
        data['heap item'] = heap_item

    partial_stop = False
    while len(edge_heap) > 0:
        _, n1, n2, valid = heapq.heappop(edge_heap)

        num_ge_mmu = rag.graph['num_ge_mmu']
        area_lt_mmu = rag.graph['area_lt_mmu']
        if ((num_ge_mmu + (area_lt_mmu/dms)) < exp_final_num) and \
                not partial_stop:
            partial_stop = True

        # if the best fitting pair consists of two regions both exceeding MAS,
        # then it is not allowed to merge

        # The merging continues this way until the sum of (a) the number of
        # regions currently larger than the minimum allowed size MMU, plus (b)
        # the expected number of final regions that may result from the area
        # currently occupied by regions smaller than MMU, is less than the
        # expected number of final regions (i.e., the image area divided by
        # DMS).

        # Thereafter, the candidate list is restricted only to those pairs
        # where at least one of both regions is smaller than MMU.

        if valid:
            n1_area = rag.nodes[n1]['pixel count']
            n2_area = rag.nodes[n2]['pixel count']
            if n1_area > mas and n2_area > mas:
                valid = False
            if n1_area > mas and n2_area > mmu:
                valid = False
            if n1_area > mmu and n2_area > mas:
                valid = False
            if partial_stop:
                if n1_area >= mmu and n2_area >= mmu:
                    valid = False

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            # Invalidate all neigbors of `src` before its deleted
            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)

            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)

            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                src, dst = n1, next_id
            else:
                src, dst = n1, n2

            merge_func(rag, src, dst, mmu)
            new_id = rag.merge_nodes(src, dst, weight_func)
            _revalidate_node_edges(rag, new_id, edge_heap)

    label_map = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix

    return label_map[labels]


def weight_scrm(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)

    return {'weight': diff}


def merge_scrm(graph, src, dst, mmu):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    src_area = graph.nodes[src]['pixel count']
    dst_area = graph.nodes[dst]['pixel count']

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = graph.nodes[dst]['total color'] /\
        graph.nodes[dst]['pixel count']

    new_area = graph.nodes[dst]['pixel count']

    d_num_ge_mmu = (new_area >= mmu) - (src_area >= mmu) - (dst_area >= mmu)

    d_area_lt_mmu = (new_area < mmu)*new_area - \
                    (src_area < mmu)*src_area - \
                    (dst_area < mmu)*dst_area

    graph.graph['num_ge_mmu'] += d_num_ge_mmu
    graph.graph['area_lt_mmu'] += d_area_lt_mmu
