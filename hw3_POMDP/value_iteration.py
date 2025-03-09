import time
import numpy as np
from maze import Maze


# Define movement directions (North, East, South, West)
DIRECTIONS = {0: (0, -1), # North
              1: (1, 0),  # East
              2: (0, 1),  # South
              3: (-1, 0)} # West

def value_iteration(maze: Maze, gamma=0.9, noise=0.1, theta=1e-3):
    """
    Solve maze navigation MDP with value iteration

    Parameters:
        maze  - The maze
        gamma - Discount factor
        noise - Probability of unintended movement
        theta - Convergence threshold

    Returns:
        V      - Optimal value function
        policy - Optimal policy mapping states to actions
    """
    num_rows, num_cols = maze.x_max_range, maze.y_max_range
    
    V = np.zeros((num_rows, num_cols))  # Initialize value function
    policy = np.zeros((num_rows, num_cols), dtype=int)  # Initialize policy

    # Iterate until convergence within theta
    while True:
        delta = 0
        new_V = np.copy(V)

        for x in range(num_rows):
            for y in range(num_cols):
                if not maze.is_move_valid(0, (x, y)) and not maze.is_move_valid(1, (x, y)) and \
                   not maze.is_move_valid(2, (x, y)) and not maze.is_move_valid(3, (x, y)):
                    continue  # Skip walls
    
                best_value = float('-inf')
                best_action = None
                # Evaluate each action from state
                for action, (dx, dy) in DIRECTIONS.items():
                    new_x, new_y = x + dx, y + dy  # Intended move
                    if not maze.is_move_valid(action, (x, y)):
                        new_x, new_y = x, y  # Stay in place if blocked

                    # Compute expected value with noisy transitions (T(s'|s,a)*[R(s') + gamma V(a')])
                    # Intended action first (with prob 1-noise)
                    expected_value = (1 - noise) * (maze.get_reward((new_x, new_y)) + gamma * V[new_x, new_y])
                    # Then noisy moves (with noise prob distributed evenly)
                    for adj_action, (ax, ay) in DIRECTIONS.items():
                        if adj_action != action:
                            adj_x, adj_y = x + ax, y + ay
                            if not maze.is_move_valid(adj_action, (x,y)):
                                adj_x, adj_y = x, y  # Stay in place if invalid
                            expected_value += (noise / 3) * (maze.get_reward((adj_x, adj_y)) + gamma * V[adj_x, adj_y])

                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action
                
                new_V[x, y] = best_value
                policy[x, y] = best_action
                delta = max(delta, abs(V[x, y] - new_V[x, y]))

        V = new_V
        if delta < theta:
            break

    return V, policy

def policy_to_arrows(policy):
    """Print the learned policy as directional arrows."""
    symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    full = []
    for row in policy.T:
        full.append([symbols[a] for a in row])
    return np.array(full)


if __name__ == "__main__":
    noise = 0.2
    discount = 0.9

    maze = Maze(noise=noise, discount=discount)
    maze.load_maze("mazes/maze0.txt")

    V, policy = value_iteration(maze)
    print(V)
    print(policy)
    arrows_pol = policy_to_arrows(policy)
    print(arrows_pol)


    maze.draw_maze(policy, title="Noise 0.2 Policy")
    maze.export_image("step1_policy_noise02.png")

