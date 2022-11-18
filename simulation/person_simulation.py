from math import radians
from math import sin, cos, floor, acos, asin, sqrt, pi
from typing import List

import configparser
import random
import numpy as np


class PersonSimulation:
    """
    This class simulates the path of a person with parameters, that can be defined in the init file.
    Another option is to give a path in the form of x and y coordinates, from which the way points
    with the given velocity of the person will be calculated.
    """

    path_cartesian = [[0, 4.], [0, 4.]]
    current_angle = 0  # the angle the person is currently walking at. 0 is straight forward. degrees
    goal_angle = 0  # the angle the person wants to reach during a turn. degrees
    turn_steps = 0  # steps taken since the last turn was initiated
    path_given_x = []  # contains points if the robot should follow a certain path
    path_given_y = []  # contains points if the robot should follow a certain path

    def __init__(self, robot_simulation, config: configparser.ConfigParser):
        """
        These parameters can be defined in the config file:

        person_velocity: in m/s
        turn_angle_time_step: angle the person can turn in each time step in degrees
        turn_probability: probability that the person walking will make a turn in this step
        min_wait_turn_steps: minimum amount of steps to wait after a complete turn
        min_turn_angle: the minimum angle the person can turn before walking forward again
        max_turn_angle: the maximum angle the person can turn before walking forward again
        fps: number of frames per second from the FMCW radar --> simulation step time

        :param robot_simulation: to get the position of the robot
        :param config: the config file
        """
        self.robot_simulation = robot_simulation

        self.person_velocity = int(config['person']['person_velocity'])
        self.turn_angle_time_step = int(config['person']['turn_angle_time_step'])
        self.turn_probability = int(config['person']['turn_probability'])
        self.min_wait_turn_steps = int(config['person']['min_wait_turn_steps'])
        self.min_turn_angle = int(config['person']['min_turn_angle'])
        self.max_turn_angle = int(config['person']['max_turn_angle'])
        self.fps = int(config['person']['fps'])
        self.change_angle = config.getboolean(section='person', option='change_angle')

        self._ratio_radius = float(config['CANN']['ratio_radius'])

    def simulate_person_movement(self) -> None:
        """
        Function simulates a moving person. Parameters like velocity, can be changed during the simulation.
        The next position is added to path_cartesian. The time difference between elements in the path_cartesian
        array is 1/fps seconds.
        """
        if self.change_angle and len(self.path_given_x) == 0:
            # check if a turn can be made, the minimum amount of steps was waited and the probability for it was meat.
            if self.turn_steps >= self.min_wait_turn_steps and random.randint(1, 100) <= self.turn_probability:
                # calculate the goal angle
                rand_angle = random.randint(self.min_turn_angle, self.max_turn_angle)
                if random.randint(0, 1) == 1:
                    self.goal_angle += rand_angle
                else:
                    self.goal_angle -= rand_angle
                self.turn_steps = 0

            # check if a turn is in progress and update the current angle accordingly
            if self.goal_angle != self.current_angle:
                if self.goal_angle > self.current_angle:
                    if self.current_angle + self.turn_angle_time_step > self.goal_angle:
                        self.current_angle = self.goal_angle
                    else:
                        self.current_angle += self.turn_angle_time_step
                else:
                    if self.current_angle - self.turn_angle_time_step < self.goal_angle:
                        self.current_angle = self.goal_angle
                    else:
                        self.current_angle -= self.turn_angle_time_step
            else:
                self.turn_steps += 1

        if len(self.path_given_x) == 0:
            # multiply the distance moved on x by the ratio of the total degrees and height
            rad = radians(-self.current_angle)
            new_x_pos_cartesian = self.path_cartesian[-1][0] + (self.person_velocity / self.fps * sin(rad))
            new_y_pos_cartesian = self.path_cartesian[-1][1] + (self.person_velocity / self.fps * cos(rad))
        else:
            if len(self.path_given_x) > 0:
                new_x_pos_cartesian = self.path_given_x[0]
                new_y_pos_cartesian = self.path_given_y[0]
                self.path_given_x = self.path_given_x[1::]
                self.path_given_y = self.path_given_y[1::]
            else:
                return

        self.path_cartesian.append([new_x_pos_cartesian, new_y_pos_cartesian])

    def get_cartesian_path_adapted(self) -> np.ndarray:
        """
        Function returns the last 100 position of the person from the current
        robot's perspective in cartesian coordinates.
        :return: the adapted last 100 positions in cartesian coordinates [[x_1, y_1], ..., [x_100, y_100]]
        """
        # adapt all the points, so that the make sense with the current robot orientation
        path_c_adapted = np.array(self.path_cartesian[-100:])
        return self.robot_simulation.get_rotated_points_cartesian(path_c_adapted)

    def get_polar_path_adapted(self) -> np.ndarray:
        """
        Function returns the polar coordinates from the adapted cartesian coordinates.
        Note for myself: the line looks weird because points that are closer to the robot and not straight in
        :return: the adapted last 100 positions in polar coordinates [[x_1, y_1], ..., [x_100, y_100]]
        """
        path_c_adapted = self.get_cartesian_path_adapted()
        # remove all points, that are behind the Radar, y coordinate is negative
        a = path_c_adapted[path_c_adapted[:, 1] >= 0].T
        # calculate polar coordinates from remaining points
        r = np.sqrt(a[0] ** 2 + a[1] ** 2)
        d = np.degrees(np.arctan2(a[0], a[1]))
        return np.array([d, r]).T

    def reset(self, x: float, y: float) -> None:
        """
        Resets the simulation to start values in case the person walks out of the radar area.
        Parameters for the path simulation of the person are not reset.

        :param x: start position of the person in cartesian coordinates on x axes
        :param y: start position of the person in cartesian coordinates on y axes
        """
        self.path_cartesian = [[x, y], [x, y]]
        self.turn_steps = 0

    def set_path(self, x_path: [], y_path: []) -> None:
        """
        Function calculates way points for the person for given x and y coordinates.
        The Points will be set in the class and the simulation will automatically stop,
        when the person finished walking the path
        :param x_path: cartesian x coordinates for the path in meters
        :param y_path: corresponding cartesian y coordinates for the path in meters
        """

        if len(x_path) != len(y_path) and len(x_path) == 0:
            return
        self.path_given_x = np.append(self.path_given_x, x_path[0])
        self.path_given_y = np.append(self.path_given_y, y_path[0])
        x_path = x_path[1::]
        y_path = y_path[1::]

        while True:
            if len(x_path) == 0:
                break
            diff_x = x_path[0] - self.path_given_x[-1]
            diff_y = y_path[0] - self.path_given_y[-1]

            theta = self._get_angle(np.array([0, 2]), np.array([diff_x, diff_y]), x_path, y_path)

            p_diff_x = self.person_velocity / self.fps * cos(theta)
            p_diff_y = self.person_velocity / self.fps * sin(theta)
            n_points = sqrt(diff_x ** 2 + diff_y ** 2) / self.person_velocity / self.fps
            rest = n_points - floor(n_points)
            n_points = floor(n_points)

            self.path_given_x = np.append(self.path_given_x, np.linspace(self.path_given_x[-1] + p_diff_x,
                                                                         self.path_given_x[-1] + n_points * p_diff_x,
                                                                         n_points))
            self.path_given_y = np.append(self.path_given_y, np.linspace(self.path_given_y[-1] + p_diff_y,
                                                                         self.path_given_y[-1] + n_points * p_diff_y,
                                                                         n_points))
            if rest != 0:
                if len(x_path) > 1:
                    a = np.array([diff_x, diff_y])
                    b = np.array([x_path[0] - x_path[1], y_path[0] - y_path[1]])
                    theta = acos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    e = x_path[0] - self.path_given_x[-1]
                    f = y_path[0] - self.path_given_y[-1]
                    beta = asin((sin(theta) * np.linalg.norm([e, f])) / (self.person_velocity / self.fps))
                    alpha = pi - theta - beta
                    a = (self.person_velocity / self.fps * sin(alpha)) / sin(theta)

                    angle = self._get_angle(np.array([0, 2]),
                                            np.array([x_path[1] - x_path[0], y_path[1] - y_path[0]]),
                                            x_path,
                                            y_path)
                    self.path_given_x = np.append(self.path_given_x, x_path[0] + a * cos(angle))
                    self.path_given_y = np.append(self.path_given_y, y_path[0] + a * sin(angle))
                else:
                    self.path_given_x = np.append(self.path_given_x, x_path[0])
                    self.path_given_y = np.append(self.path_given_y, y_path[0])

            x_path = x_path[1::]
            y_path = y_path[1::]

    def _get_angle(self, a: np.ndarray, b: np.ndarray, x_path: List, y_path: List) -> float:
        """
        Function returns the angle between two vectors in radians
        :param a: 1d numpy array for the first vector
        :param b: 1d numpy array for the second vector
        :param x_path: to check the orientation of the vectors
        :param y_path: to check the orientation of the vectors
        """
        if a.dot(b) == 0:
            theta = 0
        else:
            theta = asin(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if x_path[0] < self.path_given_x[-1]:
            if y_path[0] < self.path_given_y[-1]:
                theta = pi - theta
            else:
                theta = pi + theta
        else:
            if y_path[0] < self.path_given_y[-1]:
                theta = pi * 2 + theta
        return theta
