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

        self.stored_locs = []

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
        # Initialize explored environment
        explored_mask = np.zeros((28, 28))
        explored_mask = np.where(map == 128, 0, 1)
        est_map = self.estimator_net.runNetwork(map, explored_mask)

        robot_loc = robot.getLoc()
        self.stored_locs.append(robot_loc)

        # Get the possible moves
        possible_moves = {"left": tuple(robot_loc + np.array([-1, 0])),
                            "right": tuple(robot_loc + np.array([1, 0])),
                            "up": tuple(robot_loc + np.array([0, -1])),
                            "down": tuple(robot_loc + np.array([0, 1])),
                            }

        # Get the value of the estimated map at the possible moves
        move_values = []
        for direction in possible_moves:
            if not robot.checkValidMove(direction) or possible_moves[direction] in self.stored_locs:
                move_values.append(-np.inf)
            else:
                move_values.append(est_map[possible_moves[direction][1], 
                                           possible_moves[direction][0]])

        # move_values = [est_map[move[1], move[0]] for move in possible_moves.values()]

        direction = None
        while direction is None and np.max(move_values) != -np.inf:

            # Select min value (max entropy) cell to move to
            move_idx = np.argmax(move_values)
            # print(f"\n Possible moves: {possible_moves} \n Move values: {move_values}\n Selected idx: {move_idx}")

            if move_idx == 0:
                direction = 'left'
            if move_idx == 1:
                direction = 'right'
            if move_idx == 2:
                direction = 'up'
            if move_idx == 3:
                direction = 'down'
            
            # If it is not a valid move or is already explored, try next-best
            move_loc = tuple(possible_moves[direction])
            # print(f"Checking loc {move_loc} for move {direction}")
            if not robot.checkValidMove(direction) or move_loc in self.stored_locs:
                direction = None
                move_values[move_idx] = -np.inf
        
        if direction is None:
            # If no valid moves, randomly select a move
            direction = list(possible_moves.keys())[randint(0, 3)]

        return direction
