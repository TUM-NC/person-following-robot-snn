import math
import numpy as np
import configparser
from typing import Tuple


class RobotSimulation:

    current_angle = 0  # the angle the robot is currently driving at. 0 is in parallel to the y-axis. radians

    def __init__(self, config: configparser.ConfigParser, fps: int = 10):
        """
        :param config: the config file with parameters for the simulation
        :param fps: the wheel speed will be given in m/s to calculate the correct distance
        """
        self.path = [(0, 0), (0, 0)]
        self.fps = fps
        self.vehicle_width = float(config['robot']['width'])
        self.max_angle_velocity = float(config['robot']['max_angle_velocity'])
        self.v_left = 0
        self.v_right = 0

    def _bound_speed(self, v_left, v_right) -> Tuple[float, float]:
        """
        Function takes the newly predicted speeds sets them to a range
        at which the maximum angle velocity is not surpassed.
        :param v_left: the newly predicted velocity change for the left wheel
        :param v_right: the newly predicted velocity change for the right wheel
        :return: tuple with the bounded velocities
        """
        v_left = v_left + self.v_left
        v_right = v_right + self.v_right
        max_change = self.vehicle_width * self.max_angle_velocity
        diff = abs(v_left - v_right)
        if diff > max_change:
            if v_left < v_right:
                adapted_diff = -(diff - max_change) / 2
            else:
                adapted_diff = (diff - max_change) / 2
            v_left = v_left - adapted_diff
            v_right = v_right + adapted_diff
        return max(0, min(2, v_left)), max(0, min(2, v_right))

    def move(self, v_left, v_right) -> None:
        """
        The algorihtm is adapted from:
        http://ais.informatik.uni-freiburg.de/teaching/ss17/robotics/exercises/solutions/03/sheet03sol.pdf
        Function calculates the next position of the robot depending on the newly predicted
        left and right wheel velocity changes.
        :param v_left: the predicted velocity change for the left wheel in m/s
        :param v_right: the predicted velocity change for the right wheel in m/s
        """
        y = -self.path[-1][0]
        x = self.path[-1][1]
        self.v_left, self.v_right = self._bound_speed(v_left, v_right)

        if self.v_left == self.v_right:
            y_n = x + self.v_left * 0.1 * np.cos(self.current_angle)
            x_n = -(y + self.v_left * 0.1 * np.sin(self.current_angle))
        # circular motion
        else:
            r = self.vehicle_width / 2.0 * ((self.v_left + self.v_right) / (self.v_right - self.v_left))
            # computing center of curvature
            ICC_x = x - r * np.sin(self.current_angle)
            ICC_y = y + r * np.cos(self.current_angle)
            # compute the angular velocity
            omega = (self.v_right - self.v_left) / self.vehicle_width
            # computing angle change
            d_theta = omega * 0.1
            # forward kinematics for differential drive
            y_n = np.cos(d_theta) * (x - ICC_x) - np.sin(d_theta) * (y - ICC_y) + ICC_x
            x_n = -np.sin(d_theta) * (x - ICC_x) - np.cos(d_theta) * (y - ICC_y) - ICC_y
            self.current_angle = self.current_angle + d_theta

        self.path.append((x_n, y_n))

    def get_rotated_points_cartesian(self, a) -> np.ndarray:
        """
        Function rotates a set of points around the current driving angle of the robot.
        :param a: the 2D array with points that should be rotated
        :return: 2D array with rotated points
        """
        c = math.cos(-self.current_angle)
        s = math.sin(-self.current_angle)
        a = (np.array([[c, -s], [s, c]]).dot(a.T)).T - np.array([[c, -s], [s, c]]).dot(np.array(self.path[-1]))
        return a

    def reset(self):
        """
        Resets the simulation to start values in case the person walks out of the radar area.
        """
        self.path = [(0, 0), (0, 0)]
        self.current_angle = 0
        self.v_left = 0
        self.v_right = 0
