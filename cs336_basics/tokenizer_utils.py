import heapq


class Node:
    def __init__(self, data: bytes | None):
        self.data: bytes | None = data
        self.prev: Node | None = None
        self.next: Node | None = None


class DLL:  # Double Linked List
    def __init__(self):
        self.head: Node | None = None
        self.tail: Node | None = None

    def append(self, node: Node):
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node

    def delete(self, node: Node):
        prev = node.prev
        next = node.next

        if prev:
            prev.next = next
        else:
            self.head = next

        if next:
            next.prev = prev
        else:
            self.tail = prev


def pretok_to_dll(bts: bytes) -> DLL:
    if len(bts) < 1:
        raise ValueError('Cannot convert an empty bytes to a DLL')

    dll = DLL()
    for i in bts:
        node = Node(bytes([i]))
        dll.append(node)
    return dll


def merge_pretok(bts: bytes, merges_ranking: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    """
    merge bts according to merges_ranking
    """
    # build the DLL, the pairs pointer dict, the heap for the "min" merge
    dll = pretok_to_dll(bts)
    pairs_pointers = dict()
    heap = []

    if len(bts) <= 1:
        return [bts]

    node = dll.head
    while node.next is not None:
        pair_now = (node.data, node.next.data)
        if pair_now not in pairs_pointers:
            rank = merges_ranking.get(pair_now, None)
            if rank is not None:
                heapq.heappush(heap, (rank, pair_now))
            pairs_pointers[pair_now] = {node}
        else:
            pairs_pointers[pair_now].add(node)
        node = node.next

    # when heap is not empty, pop the min element, merge the bytes pair and on the fly modify the other components
    # we only push into the heap
    while heap:
        _, pair = heapq.heappop(heap)
        while pairs_pointers[pair]:
            # prev - node - next - next2 -> prev - (node + next) - next2
            # delete next
            node = pairs_pointers[pair].pop()
            prev = node.prev
            next = node.next
            next2 = next.next

            # new bytes data
            new_bytes = pair[0] + pair[1]

            # affected pairs
            if prev:
                pairs_pointers[(prev.data, node.data)].remove(prev)
                # new stuff
                pair_now = (prev.data, new_bytes)
                if pair_now not in pairs_pointers:
                    rank = merges_ranking.get(pair_now, None)
                    if rank is not None:
                        heapq.heappush(heap, (rank, pair_now))
                    pairs_pointers[pair_now] = {prev}
                else:
                    pairs_pointers[pair_now].add(prev)
            if next2:
                pairs_pointers[(next.data, next2.data)].remove(next)
                # new stuff
                pair_now = (new_bytes, next2.data)
                if pair_now not in pairs_pointers:
                    rank = merges_ranking.get(pair_now, None)
                    if rank is not None:
                        heapq.heappush(heap, (rank, pair_now))
                    pairs_pointers[pair_now] = {node}
                else:
                    pairs_pointers[pair_now].add(node)

            # finally change node.data and delete next
            node.data = new_bytes
            dll.delete(next)

    # at this point dll is the final result
    res = []
    node = dll.head
    while node:
        res.append(node.data)
        node = node.next

    return res


# merges_ranking = {(b'b', b's'): 1, (b'bs', b's'): 2}
# res = merge_pretok(b'bsss', merges_ranking)
# print(res)