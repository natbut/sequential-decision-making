import time

import numpy as np
from maze import Maze, Maze4D
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
        self.cost_to_come = 0 # previous steps
        self.cost_to_go = 0 # heuristic
        self.cost = 0 # f = g + h
        
    def compute_cost(self, goal_state, max_vel=2):
        """
        Minimize time
        """
        if self.parent == None:
            return
        self.cost_to_come = self.parent.cost_to_come + 1
        
        self.cost_to_go = self._time_heuristic(goal_state, m.max_vel)
        self.cost = self.cost_to_come + self.cost_to_go

    def _time_heuristic(self, goal_state, max_vel):
        """Minimum time to goal from current location; assumes max vel
        for entire straight line path."""
        return self.epsilon*np.linalg.norm(goal_state-np.asarray(self.state))/max_vel
    
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
        node_ct, path_time, path = solve_a_star(m, max_expansion, epsilon)
        if path_time:
            completion_log.append([epsilon, node_ct, path_time])
        
        # === EPSILON DECAY AND OTHER BOOKEEPING ===
        epsilon -= 0.5*(epsilon-1)
        if epsilon < 1.001: epsilon = 1.0
        running_time = time.time() - start_time
        
    return completion_log

def solve_a_star(m: Maze4D,
                 max_expansion=10000,
                 epsilon=10,
                 ):
    
    # === A* SEARCH LOOP ===
    
    # Init open and closed node lists
    pq_open = PriorityQueue()
    pq_closed = PriorityQueue()
    
    goal = np.asarray(m.state_from_index(m.get_goal()))
    
    # Compute start node heuristic value
    node = Node(m.get_start(),
                m.state_from_index(m.get_start()),
                parent=None,
                epsilon=epsilon
                )
    node.compute_cost(goal, m.max_vel)
    
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
            path_time = node.cost_to_come
            return count, path_time, node.get_path()
        
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
                # Compute neighbor cost + heuristic values
                neighbor.compute_cost(goal, m.max_vel)
                # Add neighbor to open list
                pq_open.insert(neighbor, neighbor.cost)
        
    # Return planning timeout
    print("Timeout reached, returning Fail")
    return count, False, False


if __name__ == "__main__":
    max_expansion = 10000
    epsilon=10
    timeout=0.05
    maze_id = 2

    m = Maze4D.from_pgm(f'maze{maze_id}.pgm')
    
    data = a_star_experiments(m, max_expansion, timeout, epsilon)

    print(data)
    
    _,_,path = solve_a_star(m, max_expansion, 1)
    m.plot_path(path, f'Maze{maze_id} 4D')