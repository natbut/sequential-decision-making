import time

import numpy as np
from maze import Maze, Maze2D, Maze4D
from priority_queue import PriorityQueue


class Node():
    
    def __init__(self,
                 id,
                 state,
                 parent,
                 epsilon=1.0
                 ):
        
        self.id = id
        self.state = state
        self.parent = parent
        self.epsilon=epsilon
        self.cost_to_come = 0
        self.cost_to_go = 0
        self.cost = 0
        
    def compute_cost(self, goal_state):
        if self.parent == None:
            return
        self.cost_to_come = self.parent.cost_to_come + np.linalg.norm(np.asarray(self.state)-np.asarray(self.parent.state))
        
        self.cost_to_go = self._euclidean_heuristic(goal_state)
        self.cost = self.cost_to_come + self.cost_to_go

    def _euclidean_heuristic(self, goal_state):
        return self.epsilon*np.linalg.norm(goal_state-np.asarray(self.state))
    
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
    

def a_star_experiments(m: Maze,
                       max_expansion=10000,
                       timeout=1.0,
                       epsilon=10,
                       ) -> np.array:
    
    start_time = time.time()
    running_time = 0.0
    completion_log = [] #epsilon, node_count, path_length
    last_cycle = False
    while running_time < timeout and not last_cycle:
        if epsilon == 1.0: last_cycle = True
        
        # === A* SEARCH LOOP ===
        node_ct, path_len, path = solve_a_star(m, max_expansion, epsilon)
        if path_len:
            completion_log.append([epsilon, node_ct, path_len])
        
        # === EPSILON DECAY AND OTHER BOOKEEPING ===
        epsilon -= 0.5*(epsilon-1)
        if epsilon < 1.001: epsilon = 1.0
        running_time = time.time() - start_time
        
    return completion_log

def solve_a_star(m: Maze,
                 max_expansion=10000,
                 epsilon=10,
                 ):
    
    # === A* SEARCH LOOP ===
    
    # Init open and closed node lists
    pq_open = PriorityQueue()
    pq_closed = PriorityQueue()
    
    start = np.asarray(m.state_from_index(m.get_start()))
    goal = np.asarray(m.state_from_index(m.get_goal()))
    
    # Compute start node heuristic value
    node = Node(m.get_start(),
                m.state_from_index(m.get_start()),
                parent=None,
                epsilon=epsilon
                )
    node.compute_cost(goal)
    
    # Load start node in open list
    pq_open.insert(node, node.cost)
        
    # Planning loop
    count = 0
    # path_done = False
    while count < max_expansion: # and not path_done:
        
        # Pop top node from open list, put in closed list
        node = pq_open.pop()
        pq_closed.insert(node, 0)
        
        # If node is goal, create path from parents & return
        if node.id == m.get_goal():
            # path_done = True
            path_length = node.cost_to_come
            return count, path_length, node.get_path()
        
        # Explore node neighbors (that are not in closed list)
        for id in m.get_neighbors(node.id):

            neighbor = Node(id,
                        m.state_from_index(id),
                        parent=node,
                        epsilon=epsilon
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
    print("Timeout reached, returning Fail")
    return count, False, False


if __name__ == "__main__":
    max_expansion = 10000
    epsilon=10
    timeout=1.0
    maze_id = 2

    m = Maze2D.from_pgm(f'maze{maze_id}.pgm')
    
    data = a_star_experiments(m, max_expansion, timeout, epsilon)

    print(data)
    
    _,_,path = solve_a_star(m, max_expansion, 1)
    m.plot_path(path, f'Maze{maze_id} 2D Euclidean')