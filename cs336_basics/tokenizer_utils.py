class Node:
    def __init__(self, data: bytes | None):
        self.data: bytes | None = data
        self.prev: Node | None = None
        self.next: Node | None = None


class DLL:  # Double Linked List
    def __init__(self):
        self.head: Node | None = None
        self.tail: Node | None = None


def pretok_to_dll(bts: bytes) -> DLL:
    if len(bts) < 1:
        raise ValueError('Cannot change an empty bytes to a DLL')

    if len(bts) == 1:
        node = Node(bts)
        dll = DLL()
        dll.head = node
        dll.tail = node
        return dll

    # len(bts) > 1
    prev = None
    next = None


