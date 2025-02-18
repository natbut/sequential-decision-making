__author__ = 'Nathan Butler'

import time
import gzip
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from OneStepNavigator import OneStepNavigator
from IPPNavigator import IPPNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork


def run_sim(game: Game, robot: Robot, timeout):
    start_time = time.time()
    for i in range(0, 500):
        found_goal = game.tick()
        if i % 20 == 0:
            print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
        if found_goal:
            print(f"Found goal at time step: {game.getIteration()}!")
            break

        duration = time.time()-start_time
        if duration > timeout:
            break

    print(f"Final Score: {game.score}, Duration: {duration}")

    return game.score, duration

def run_tests(num_tests, timeout, nav_func, render=False):
    # Create a Map Class Object
    map = Map()

    # Create the world estimating network
    uNet = WorldEstimatingNetwork()
    # Create the digit classification network
    classNet = DigitClassificationNetwork()

    # Create a Robot that starts at (0,0)
    # The Robot Class stores the current position of the robot
    # and provides ways to move the robot 
    robot = Robot(0, 0)

    # Create navigator object
    navigator = nav_func(uNet, classNet)

    rewards = []
    runtimes = []
    for i in range(num_tests):
        map.getNewMap()
        # Get the current map from the Map Class
        data = map.map
        print(f"\nITER. {i} MAP DIGIT: {map.number}")
        robot.resetRobot()
        navigator.resetNav()

        # Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
        game = Game(data, map.number, navigator, robot)

        # This loop runs the game for 1000 ticks, stopping if a goal is found.
        reward, runtime = run_sim(game, robot, timeout)
        runtimes.append(runtime)
        rewards.append(reward)

        if render:
            # Show how much of the world has been explored
            Image.fromarray(np.uint8(game.exploredMap)).show(title=f"Iter {i} Explored Map")
            mask = np.zeros((28, 28))
            mask = np.where(game.exploredMap == 128, 0, 1)

            # Creates an estimate of what the world looks like
            image_est = uNet.runNetwork(game.exploredMap, mask)

            # Show the image of the estimated world
            Image.fromarray(image_est).show(title=f"Iter {i} Est. World")

    return np.average(rewards), np.average(runtimes)


if __name__ == "__main__":
    num_tests = 5
    timeout = 15*60
    nav_func = IPPNavigator #  OneStepNavigator # 
    render = True

    avg_rew, avg_time = run_tests(num_tests=num_tests,
                                    timeout=timeout,
                                    nav_func=nav_func,
                                    render=render
                                    )

    print(f"\nTests complete! Average reward: {avg_rew}, Average time: {avg_time}")