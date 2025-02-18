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

        Once a satisfactory amound of information is collected, the robot estimates the map digit and
        moves to the corresponding location
        """
        self.estimator_net = estimator_net
        self.classifier_net = classifier_net

        self.goal_locs = {"012": (0,27),
                          "345": (27,27),
                          "6789": (27,0)
                          }
        
        self.goal_selected = None
        self.stored_locs = []

    def resetNav(self):
        self.goal_selected = None
        self.stored_locs = []

    def getAction(self,
                  robot: Robot,
                  map: Map,
                  digit_id_thresh=0.05,
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

        # Check whether digit can be identified, update goal if so
        if self.goal_selected is None:
            digit_vals = self.classifier_net.runNetwork(est_map)
            best_digit = digit_vals.argmax()
            best_val = digit_vals[0][best_digit]
            if abs(best_val) < digit_id_thresh:
                for key in self.goal_locs:
                    if str(best_digit) in key:
                        self.goal_selected = self.goal_locs[key]
                        print(f"!!! PREDICT DIGIT IS {best_digit}, GO TO GOAL {self.goal_selected}")

        # Get the possible moves
        robot_loc = robot.getLoc()
        self.stored_locs.append(robot_loc)
        possible_moves = {"left": tuple(robot_loc + np.array([-1, 0])),
                            "right": tuple(robot_loc + np.array([1, 0])),
                            "up": tuple(robot_loc + np.array([0, -1])),
                            "down": tuple(robot_loc + np.array([0, 1])),
                            }

        # Check to break out of getting stuck at wrong goal
        if robot_loc == self.goal_selected:
            self.goal_selected = None

        if self.goal_selected is not None:
            return self.get_goal_action(robot, possible_moves, est_map)
        else:
            return self.get_exploratory_action(robot, possible_moves, est_map)
        


    def get_exploratory_action(self, robot: Robot, possible_moves: dict, est_map: Map):
        """
        Get action that explores environment
        """
        # Get the value of the estimated map at the possible moves
        move_values = []
        for direction in possible_moves:
            if not robot.checkValidMove(direction) or possible_moves[direction] in self.stored_locs:
                move_values.append(-np.inf)
            else:
                move_values.append(est_map[possible_moves[direction][1], 
                                           possible_moves[direction][0]])

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

    def get_goal_action(self, robot: Robot, possible_moves: dict, est_map):
        """
        If goal has been determined, then take steps toward goal
        """
        move_dists = {"left": 0.0,
                      "right": 0.0,
                      "up": 0.0,
                      "down": 0.0
                      }
        for direction in possible_moves:
            # Don't use invalid moves
            if not robot.checkValidMove(direction):
                move_dists[direction] = np.inf
                continue

            # Compute euclidean distance from move loc to goal
            move_dists[direction] = np.linalg.norm(np.array(self.goal_selected) - \
                                                   np.array(possible_moves[direction]))

        # print(f"Robot loc: {robot.getLoc()}, goal: {self.goal_selected}, move dists: {move_dists}")
        return min(move_dists, key=move_dists.get)