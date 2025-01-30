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
    

def solve_rrt_2D(m: Maze,
                 max_iters=1000,
                 ):
    
    
    start = np.asarray(m.state_from_index(m.get_start()))
    goal = np.asarray(m.state_from_index(m.get_goal()))
    
    # Compute start node heuristic value
    node = Node(m.get_start(),
                m.state_from_index(m.get_start()),
                parent=None,
                )
    node.compute_cost(goal)
    
    # pseudocode from https://theclassytim.medium.com/robotic-path-planning-rrt-and-rrt-212319121378
    # Qgoal //region that identifies success
    # Counter = 0 //keeps track of iterations
    # lim = n //number of iterations algorithm should run for
    # G(V,E) //Graph containing edges and vertices, initialized as empty
        
    # Planning loop
    count = 0
    # path_done = False
    while count < max_iters: # and not path_done:
        # pseudocode from https://theclassytim.medium.com/robotic-path-planning-rrt-and-rrt-212319121378
        
        # Xnew  = RandomPosition()
        # if IsInObstacle(Xnew) == True:
        #     continue
        # Xnearest = Nearest(G(V,E),Xnew) //find nearest vertex
        # Link = Chain(Xnew,Xnearest)
        # G.append(Link)
        # if Xnew in Qgoal:
        #     Return G

        
    # Return planning timeout
    print("Timeout reached, returning Fail")
    return count, False, False


if __name__ == "__main__":
    max_iters = 10000
    maze_id = 1

    m = Maze2D.from_pgm(f'maze{maze_id}.pgm')
    
    _,_,path = solve_rrt_2D(m, max_iters)
    
    m.plot_path(path, f'Maze{maze_id} 2D Euclidean')