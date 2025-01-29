import numpy as np
from maze import Maze, Maze2D, Maze4D
from priority_queue import PriorityQueue


class Node():
    
    def __init__(self,
                 id,
                 state,
                 parent,
                 ):
        
        self.id = id
        self.state = state
        self.parent = parent
        
    def compute_cost(self, start_state, goal_state):
        to_come = np.linalg.norm(np.asarray(self.state)-start_state)
        to_go = np.linalg.norm(goal_state-np.asarray(self.state))
        return to_come + to_go
    
    def get_path(self):
        path = [self.state]
        node = self.parent
        while node.parent is not None:
            node = node.parent
            path.append(node.state)
        return np.array(path)
    
    def __str__(self):
        return "Node" + str(self.id)
    

def solve_a_star(m: Maze2D, max_expansion=10000) -> np.array:
    
    # Init open and closed node lists
    pq_open = PriorityQueue()
    pq_closed = PriorityQueue()
    
    start = np.asarray(m.state_from_index(m.get_start()))
    goal = np.asarray(m.state_from_index(m.get_goal()))
    print("Start:", start, " | Goal:", goal)
    
    # Compute start node heuristic value
    node = Node(m.get_start(),
                m.state_from_index(m.get_start()),
                parent=None,
                )
    cost = node.compute_cost(start, goal)
    
    # Load start node in open list
    pq_open.insert(node, cost)
        
    # Planning loop
    count = 0
    while count < max_expansion:
        
        # Pop top node from open list, put in closed list
        node = pq_open.pop()
        pq_closed.insert(node, 0)
        print("\tExpanding", node)
        
        # If node is goal, create path from parents & return
        if node.id == m.get_goal():
            return node.get_path()
        
        # Explore node neighbors (that are not in closed list)
        for id in m.get_neighbors(node.id):
            neighbor = Node(id,
                            m.state_from_index(id),
                            parent=node,
                            )
            if pq_closed.test(neighbor):
                # Don't consider closed nodes
                print("\tNODE ALREADY SEARCHED")
                continue
            else:
                count += 1
                # Compute neighbors heuristic values
                cost = neighbor.compute_cost(start, goal)
                # Add neighbor to open list
                pq_open.insert(neighbor,cost)
        
    # Return planning timeout
    print("Timeout reached, returning incomplete path")
    return node.get_path()



if __name__ == "__main__":
    max_expansion = 10000
    m = Maze2D.from_pgm('maze1.pgm')
    
    path = solve_a_star(m, max_expansion)
    
    print("Solved path:", path)
    
    m.plot_path(path, 'Maze2D')