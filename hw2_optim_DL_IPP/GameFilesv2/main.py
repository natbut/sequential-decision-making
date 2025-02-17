__author__ = 'Nathan Butler'

import gzip
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from OneStepNavigator import OneStepNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork


if __name__ == "__main__":
    # Create a Map Class Object
    map = Map()

    # Get the current map from the Map Class
    data = map.map

    # Print the number of the current map
    print(map.number)

    # Create a Robot that starts at (0,0)
    # The Robot Class stores the current position of the robot
    # and provides ways to move the robot 
    robot = Robot(0, 0)

    # Create the world estimating network
    uNet = WorldEstimatingNetwork()

    # Create the digit classification network
    classNet = DigitClassificationNetwork()

    # Create navigator object
    navigator = OneStepNavigator(uNet, classNet)

    # Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
    game = Game(data, map.number, navigator, robot)


    # This loop runs the game for 1000 ticks, stopping if a goal is found.
    for x in range(0, 200):
        found_goal = game.tick()
        print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
        if found_goal:
            print(f"Found goal at time step: {game.getIteration()}!")
            break
    print(f"Final Score: {game.score}")

    # Show how much of the world has been explored
    im = Image.fromarray(np.uint8(game.exploredMap)).show()

    mask = np.zeros((28, 28))
    mask = np.where(game.exploredMap == 128, 0, 1)

    # Creates an estimate of what the world looks like
    image = uNet.runNetwork(game.exploredMap, mask)

    # Show the image of the estimated world
    Image.fromarray(image).show()

    # Use the classification network on the estimated image
    # to get a guess of what "world" we are in (e.g., what the MNIST digit of the world)
    char = classNet.runNetwork(image)
    # get the most likely digit 
    print(char.argmax())

    # Get the next map to test your algorithm on.
    map.getNewMap()
