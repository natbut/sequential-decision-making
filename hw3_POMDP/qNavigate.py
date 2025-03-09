import time
import numpy as np

from value_iteration import value_iteration, DIRECTIONS
from maze import Maze

def qNavigate(maze: Maze,
              policy,
              reward,
              max_steps=100,
              visualize=False,
              ):
    """
    Executes the optimal policy in the maze environment.
    
    Parameters:
        maze     - Grid world.
        policy   - Optimal policy from value iteration.
        start    - (x, y) start position.
        max_steps - Number of steps to run.
    
    Returns:
        total_reward - Total reward accumulated during navigation.
    """
    # cur_discount = discount
    for _ in range(max_steps):
        # choose a random direction to move in
        move = policy[maze.position]
        # try to move in the direction
        maze.step(move)
        # draw the maze - If you want to draw specific values in the squares
        # You can set the optional values array with the text or values you want to appear
        # In this example, we put random values
        maze.draw_maze(values=policy)

        # Update the reward
        reward += maze.get_reward()

        if visualize:
            # Print out location of the agent
            print(maze.position)
            # Sleep for a bit so that the user can see the animation
            # Can remove this if running headless
            time.sleep(0.2)

    return reward

if __name__ == "__main__":
    reward = 0
    noise = 0.2
    discount = 0.9
    cur_discount = discount

    maze = Maze(noise=noise, discount=discount)
    maze.load_maze("mazes/maze0.txt")

    # Get policy
    V, policy = value_iteration(maze)
    print("Solved policy:\n", policy)

    # Run navigation with the policy
    reward = qNavigate(maze,
                       policy,
                       reward,
                       max_steps=100,
                       visualize=True,
                       )

    # Report final reward
    print("Final Reward:", reward)
    print("Reward tracked by maze class:", maze.reward_current)

