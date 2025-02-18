__author__ = 'Nathan Butler'
import numpy as np
from random import randint
from RobotClass import Robot
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork
import random

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.info_gain = 0.0  # Information
        self.path_length = 0  # Path length from the start

    def update_stats(self, info_gain, new_length):
        self.info_gain = info_gain
        self.path_length = new_length



class IPPNavigator:
    def __init__(self,
                  estimator_net: WorldEstimatingNetwork,
                  classifier_net: DigitClassificationNetwork,
                  map_x=28,
                  map_y=28,
                  ) -> None:
        """
        Initializes the IPPNavigator class. This class implements an informative path planner that
        considers medium and long-horizon trajectories for selecting robot actions.

        Once a satisfactory amount of information is collected, the robot estimates the map digit and
        moves to the corresponding location
        """
        self.estimator_net = estimator_net
        self.classifier_net = classifier_net
        self.map_x = map_x
        self.map_y = map_y

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
                  RIG_iter=500,
                  max_RIG_path=10,
                  epsilon=1.0,
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
        possible_moves = {"left": tuple(robot_loc + np.array([-1, 0])),
                            "right": tuple(robot_loc + np.array([1, 0])),
                            "up": tuple(robot_loc + np.array([0, -1])),
                            "down": tuple(robot_loc + np.array([0, 1])),
                            }

        # Check to break out of getting stuck at wrong goal
        if robot_loc == self.goal_selected:
            self.stored_locs.append(robot_loc)
            self.goal_selected = None

        if self.goal_selected is not None:
            return self.get_goal_action(robot, possible_moves)
        else:
            self.stored_locs.append(robot_loc)
            return self.get_exploratory_action(robot, possible_moves, est_map, RIG_iter, max_RIG_path, epsilon)
        
    def euclidean_distance(self, p1, p2):
        """Compute Euclidean distance"""
        return np.linalg.norm(np.array(p2)-np.array(p1))

    def get_info_value(self, p, est_map):
        """Get the information value at a given point"""
        return est_map[p[1], p[0]]

    def get_neighbors(self, p, robot: Robot):
        """
        Get the neighbors of a point (connected map cells)
        """
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_point = (p[0] + dx, p[1] + dy)
            if robot.checkValidMove(new_point):
                neighbors.append(new_point)
        return neighbors

    def extend_tree(self, tree: list[Node], est_map: Map, random_point, visited_points, epsilon, max_path_length):
        """
        Extend the tree towards a random point
        """
        # Find nearest node & dist in search tree to sampled point
        nearest_node = min(tree, key=lambda node: self.euclidean_distance(node.position, random_point))
        direction = (random_point[0] - nearest_node.position[0], random_point[1] - nearest_node.position[1])
        move_zero = np.argmin(abs(np.array(direction))) # direction to step in
        dist = self.euclidean_distance(nearest_node.position, random_point)

        if dist == 0:
            return None
        
        step = min(epsilon, dist)

        # Step size is constrained by epsilon & 4-way movements
        new_position = [int(np.rint(nearest_node.position[0] + direction[0] * step / dist)),
                        int(np.rint(nearest_node.position[1] + direction[1] * step / dist))]
        new_position[move_zero] = nearest_node.position[move_zero] # constrain to 4-way move
        new_position = tuple(new_position)
        if new_position in visited_points:
            return None

        # Create a new node at the new position
        new_node = Node(position=new_position, parent=nearest_node)
        new_node.update_stats(nearest_node.info_gain + self.get_info_value(new_position, est_map), nearest_node.path_length + step)
        
        # Only add the new node if the path length is within the budget constraint
        if new_node.path_length <= max_path_length:
            tree.append(new_node)
            return new_node
        return None

    def rewire_tree(self, tree: list[Node], est_map: Map, new_node):
        """
        Rewire the tree to optimize path length and information gain
        """
        for node in tree:
            # Skip if the node is the new node or its parent
            if node == new_node.parent or node == new_node:
                continue
            # Calculate potential rewiring info gain and length
            dist = self.euclidean_distance(node.position, new_node.position)
            potential_info_gain = new_node.info_gain + self.get_info_value(node.position, est_map)
            potential_length = new_node.path_length + dist
            
            # If the rewired path is better (more info and within length constraint), rewire
            if potential_info_gain > node.info_gain and potential_length <= node.path_length:
                
                node.update_stats(potential_info_gain, potential_length)
                node.parent = new_node
                return True
        return False

    def RIG_search(self,
                   est_map: Map,
                   visited_locs,
                   robot: Robot,
                   max_iter=1000,
                   max_path_length=7,
                   epsilon=1.0,
                   ):
        """
        Use RIG to find path that maximizes information gained along trajectory
        subject to a path cost budget (path length for us here)

        P* = argmax I(P) s.t. c(P) <= B
        """
        # initialize search tree
        start = robot.getLoc()
        # print("Starting loc", start)
        tree = [Node(position=start)]
        
        sampled_points = [start] + visited_locs # closed list
        for _ in range(max_iter):
            # Sample a random point
            random_point = np.random.randint(0, self.map_x-1), np.random.randint(0, self.map_y-1)
            if random_point in sampled_points:
                continue
            else:
                sampled_points.append(random_point)

            # Extend the tree towards the random point
            new_node = self.extend_tree(tree, est_map, random_point, sampled_points, epsilon, max_path_length)
            
            if new_node:
                # Rewire the tree
                self.rewire_tree(tree, est_map, new_node)
                sampled_points.append(new_node.position)

        return self.process_RIG_tree(tree)

    def process_RIG_tree(self, tree: list[Node]):
        """
        Process tree to get values for left, right, up, down actions
        """

        most_informed_node = max(tree, key=lambda node: node.info_gain)

        path = []
        node = most_informed_node
        while node is not None:
            path.append(node)
            node = node.parent 
        path.reverse()

        # If no path is found, return none
        if len(path) > 1:
            best_direction = path[1].position
        else:
            best_direction = None

        return best_direction

    def get_exploratory_action(self,
                               robot: Robot,
                               possible_moves: dict,
                               est_map: Map,
                               RIG_iter,
                               max_RIG_path,
                               epsilon,
                               ):
        """
        Get action that explores environment.

        Use RIG to build high-value exploration path.
        """
        # Do path planning to estimate value of possible moves
        next_pos = self.RIG_search(est_map, self.stored_locs, robot, RIG_iter, max_RIG_path, epsilon)

        # Move according to path if move valid
        direction = None
        if next_pos and next_pos not in self.goal_locs.values():
            for move in possible_moves:
                if next_pos == possible_moves[move]:
                    direction = move

        else:
            # Take random action if no path found or goal move selected
            rand_move = random.choice(list(possible_moves.keys()))
            while possible_moves[rand_move] in self.goal_locs.values():
                rand_move = random.choice(list(possible_moves.keys()))
            return rand_move
        
        return direction

    def get_goal_action(self, robot: Robot, possible_moves: dict):
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

        return min(move_dists, key=move_dists.get)