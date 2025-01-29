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
        self.cost = 0
        
    def compute_cost(self, goal_state):
        if self.parent == None:
            return
        cost_to_come = self.parent.cost + np.linalg.norm(np.asarray(self.state)-np.asarray(self.parent.state))
        cost_to_go = self._euclidean_heuristic(goal_state)
        self.cost = cost_to_come + cost_to_go

    def _euclidean_heuristic(self, goal_state):
        return np.linalg.norm(goal_state-np.asarray(self.state))
    
    def get_path(self):
        path = [self.state]
        node = self.parent
        while node.parent is not None:
            node = node.parent
            path.append(node.state)
        return np.array(path)
    
    def __str__(self):
        return "Node"+str(self.id)
    
    def __repr__(self):
        return "Node"+str(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)
    

def solve_a_star(m: Maze, max_expansion=10000) -> np.array:
    
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
    node.compute_cost(goal)
    
    # Load start node in open list
    pq_open.insert(node, node.cost)
        
    # Planning loop
    count = 0
    while count < max_expansion:
        
        # Pop top node from open list, put in closed list
        node = pq_open.pop()
        pq_closed.insert(node, 0)
        
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
                continue
            else:
                count += 1
                # Compute neighbor heuristic values
                neighbor.compute_cost(goal)
                # Add neighbor to open list
                pq_open.insert(neighbor, neighbor.cost)
        
    # Return planning timeout
    print("Timeout reached, returning incomplete path")
    return node.get_path(), node.cost



if __name__ == "__main__":
    max_expansion = 10000
    maze_id = 1

    m = Maze2D.from_pgm(f'maze{maze_id}.pgm')
    
    path = solve_a_star(m, max_expansion)

    print(path)
    
    m.plot_path(path, f'Maze{maze_id} 2D Euclidean')