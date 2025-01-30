import time

import numpy as np
from maze import Maze, Maze2D, Maze4D


class Node():
    
    def __init__(self,
                 id,
                 state,
                 parent=None
                 ):
        
        self.id = id
        self.state = state
        self.parent = parent
        
    
    def get_path(self):
        
        return []
    
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
    
def sample_maze(m: Maze, goal_prob) -> tuple:
    if np.random.rand() < goal_prob:
        # sample goal
        return m.state_from_index(m.get_goal())
    else:
        # do random sample
        x_samp = np.random.randint(m.cols)
        y_samp = np.random.randint(m.rows)
        return (x_samp, y_samp)
    
def get_nearest(sample: Node, node_list: list[Node])-> Node:
    nearest = node_list[0]
    np_sample = np.asarray(sample)
    min_dist = np.linalg.norm(np_sample-np.asarray(nearest.state))
    for node in node_list[1:]:
        dist = np.linalg.norm(np_sample-np.asarray(node.state))
        if dist < min_dist:
            nearest = node
            min_dist = dist
            
    return nearest
    
def step_toward_sample(node: Node, sample_state: tuple, step_size) -> tuple:
    if node.state == sample_state:
        # return false if same point
        return False
    
    np_sample = np.asarray(sample_state)
    np_orig = np.asarray(node.state)
    vec = np_sample-np_orig
    unit_vec = np.ceil((vec)/np.linalg.norm(vec))
    np_step_state = np.asarray(node.state) + step_size*unit_vec
    # print(f"Vec from {node.state} to {sample_state}: {vec}. Unit vec: {unit_vec}. New state: {tuple(np_step_state)}")
    
    return tuple(np_step_state)

def solve_rrt_2D(m: Maze,
                 max_iters=1000,
                 goal_prob=0.05,
                 step_size=1.0
                 ):
    
    start_time = time.time()
    
    
    # === RRT ALGO START ===
    # Initialize graph
    node_list = [Node(m.get_start(),
                m.state_from_index(m.get_start()),
                parent=None,
                )]
        
    # Planning loop
    count = 0
    while count < max_iters:
        count += 1
        
        # Sample random state
        sample_state = sample_maze(m, goal_prob)
        
        # Check if in obstacle
        if m.check_occupancy(sample_state):
            continue
        
        # Find nearest vertex
        nearest_node = get_nearest(sample_state, node_list)
        
        # Step toward sampled point
        new_state = step_toward_sample(nearest_node, sample_state, step_size)
        
        # Attempt to connect
        if new_state and not m.check_hit(nearest_node.state,
                           np.asarray(new_state)-np.asarray(nearest_node.state)):
            node_list.append(Node(m.index_from_state(new_state),
                                  new_state,
                                  nearest_node,
                                  ))
    # === RRT END ===
    
    run_time = time.time() - start_time
    # Extract plan
    for node in node_list:
        if node.id == m.get_goal():
            # Goal node found, attempt to backtrack path
            path = [node.state]
            path_cost = 0
            while node.parent:
                path_cost += np.linalg.norm(np.asarray(node.state)- \
                                        np.asarray(node.parent.state))
                node = node.parent
                path.append(node.state)
            if node.id == m.get_start():
                # Complete path found
                print("Path complete!")
                return path, path_cost, run_time
            else:
                # Incomplete path
                print("Incomplete path")
                return path, path_cost, run_time
            
    # Goal never reached
    print("Goal state never sampled")
    return False, False, False


if __name__ == "__main__":
    max_iters = 1000
    goal_prob = 0.05
    step=1.0
    maze_id = 1

    m = Maze2D.from_pgm(f'maze{maze_id}.pgm')
    
    path, cost, run_time = solve_rrt_2D(m, max_iters, goal_prob, step)
    
    if path:
        print(f"Completed in {run_time}s | Cost: {cost}")
        m.plot_path(path, f'Maze{maze_id} 2D')
    else:
        print("No path found!")