from priority_queue import PriorityQueue

pq = PriorityQueue()

pq.insert("a", 1)

assert len(pq) == 1

pq.insert("c", 3)

pq.insert("b", 2)

pq.insert("d", 3.5)

assert len(pq) == 4

pq._remove_item("b")

assert len(pq) == 3

assert pq.test("a")

assert pq.pop() == "a"

assert not pq.test("a")

assert len(pq) == 2

assert pq.pop() == "c"

assert len(pq) == 1

assert pq.pop() == "d"

assert len(pq) == 0

print("all tests passed")
