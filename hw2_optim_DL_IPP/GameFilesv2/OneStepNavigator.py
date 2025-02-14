__author__ = 'Nathan Butler'
import numpy as np
from random import randint
from RobotClass import Robot
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork
from PIL import Image

class OneStepNavigator:
    def __init__(self,
                  estimator_net: WorldEstimatingNetwork,
                  classifier_net: DigitClassificationNetwork,
                  ) -> None:
        """
        Initializes the OneStepNavigator class. This class implements a greedy solver in which the
        robot looks one step ahead and moves to the location that gains the maximum information.
        """
        self.estimator_net = estimator_net
        self.classifier_net = classifier_net
        self.prev_loc = None

    def getAction(self,
                  robot: Robot,
                  map: Map,
                  ) -> str:
        """ 
        @param robot: A Robot object
        @param map: A 28x28 array representing the current explored map.

        @ return: A string representing one of four cardinal directions
        that the robot should move
        """
        explored_mask = np.zeros((28, 28))
        explored_mask = np.where(map == 128, 0, 1)

        est_map = self.estimator_net.runNetwork(map, explored_mask)
        # print(f"Estimated digit: {est_map}")
        # print(f"Classification: {self.classifier_net.runNetwork(est_map)}")
        robot_loc = robot.getLoc()

        # Get the possible moves
        possible_moves = [robot_loc + np.array([-1, 0]),
                            robot_loc + np.array([1, 0]),
                            robot_loc + np.array([0, 1]),
                            robot_loc + np.array([0, -1])]
        
        if self.prev_loc:
            print(f"Possible moves {possible_moves}, prev loc {self.prev_loc}")
            print(np.all(possible_moves == self.prev_loc, axis=0))
            prev_idx = np.where(np.all(possible_moves == self.prev_loc, axis=0))[0][0]

        # Get the value of the estimated map at the possible moves
        move_values = [est_map[move[1], move[0]] for move in possible_moves]
        if self.prev_loc:
            move_values[prev_idx] = -np.inf

        # print(f"Map row 0: {est_map[0]}")
        # print(f"Map row 1: {est_map[1]}")
        print(f"Previous location: {self.prev_loc}")
        # Show the estimated map with robot
        # est_map[robot_loc[0], robot_loc[1]] = 255
        # Image.fromarray(est_map).show()
        self.prev_loc = np.array(robot_loc)

        direction = None
        while direction is None:

            # Select min value (max entropy) cell to move to
            move_idx = np.argmax(move_values)
            # print(f"\n Possible moves: {possible_moves} \n Move values: {move_values}\n Selected idx: {move_idx}")

            if move_idx == 0:
                direction = 'left'
            if move_idx == 1:
                direction = 'right'
            if move_idx == 2:
                direction = 'down'
            if move_idx == 3:
                direction = 'up'
            
            # If it is not a valid move, try next-best
            if not robot.checkValidMove(direction):
                direction = None
                move_values[move_idx] = -np.inf
        
        print("Moving: ", direction)
        return direction
