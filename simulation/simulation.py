import math
import numpy as np
import configparser

from .person_simulation import PersonSimulation
from .robot_simulation import RobotSimulation
from .cann_simulation import CANNSimulation
from .visualization import Visualization
from typing import List


class Simulation:
    """
    Class combines the person simulation, robot simulation, cann simulation and visualization.
    It also provides the needed methods for the SNN and training.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        :param config: the config file with parameters for the simulation
        """

        # get all the parameters from the init file
        self._number_neurons_y = int(config['CANN']['number_neurons_y'])

        self._reward_scaler_cann_x = float(config['reward.cann']['scaler_x'])
        self._reward_scaler_cann_y = float(config['reward.cann']['scaler_y'])
        self._reward_scaler_distance = float(config['reward.distance']['scaler'])

        self._distance_to_person_x = float(config['simulation']['desired_distance_person_x'])
        self._distance_to_person_y = float(config['simulation']['desired_distance_person_y'])

        self._starting_pos_person_radius = float(config['setup']['starting_pos_person_radius'])
        self._starting_pos_person_angle = float(config['setup']['starting_pos_person_angle'])
        self._ratio_radius = float(config['CANN']['ratio_radius'])

        self._show_simulation_visualization = config.getboolean(section='setup', option='show_simulation_visualization')

        self.previous_reward_left = 0
        self.previous_reward_right = 0

        self.robot_simulation = RobotSimulation(config=config)
        self.person_simulation = PersonSimulation(self.robot_simulation, config=config)
        self.cann_simulation = CANNSimulation(config=config)
        if self._show_simulation_visualization:
            self.visualization = Visualization(config=config)

    def animate_next_step(self, step: int) -> bool:
        """
        Function animates the next simulation step.
        :param step: how often the simulation was run
        """
        self.person_simulation.simulate_person_movement()
        if self._show_simulation_visualization:
            self.visualization.update(step=step,
                                      cartesian_path_person_adapted=self.person_simulation.get_cartesian_path_adapted(),
                                      polar_path_person=self.person_simulation.get_polar_path_adapted(),
                                      cartesian_path_robot=self.robot_simulation.path,
                                      cartesian_path_person=self.person_simulation.path_cartesian,
                                      cann_spikes=self.cann_simulation.previous_spikes_probability,
                                      angle_person=self.person_simulation.current_angle,
                                      angle_robot=self.robot_simulation.current_angle,
                                      v_robot=(self.robot_simulation.v_left + self.robot_simulation.v_right) / 2,
                                      v_person=self.person_simulation.person_velocity)

        if self._stop_animation():
            return True
        else:
            return False

    def cann_reward(self) -> (float, float):
        """
        Function returns the reward for the synapses connecting the CANN to the left and right
        wheel neurons.
        Scalars can be modified in the config.ini
        :return: left and right reward
        """
        # subtract the distance that should be kept to the person
        difference_edited = self.person_path_cartesian[-1][1] - self.robot_path_cartesian[-1][1]
        difference_edited -= self._distance_to_person_y
        r, d = self.distance

        l_left = np.sign(difference_edited) * self._reward_scaler_cann_y * r + d * self._reward_scaler_cann_x
        l_right = np.sign(difference_edited) * self._reward_scaler_cann_y * r - d * self._reward_scaler_cann_x

        if abs(self.previous_reward_left) > abs(l_left) and abs(self.previous_reward_right) > abs(l_right):
            self.previous_reward_left = l_left
            self.previous_reward_right = l_right
            l_left, l_right = 0, 0
        else:
            self.previous_reward_left = l_left
            self.previous_reward_right = l_right

        return l_left, l_right

    def distance_reward(self) -> (float, float):
        """
        Function returns the reward for the distance network.
        :return: reward, reward; two rewards need to be set, synapses connecting to the left and right
                 wheel neurons
        """

        dist = np.array(self.person_path_cartesian[-1]) - np.array(self.robot_path_cartesian[-1])
        dist = dist - np.array([self._distance_to_person_x, self._distance_to_person_y])
        v_diff = self.robot_velocity - self.person_velocity
        reward = self._reward_scaler_distance * abs(v_diff) / max(0.1, np.sqrt(dist[0] ** 2.0 + dist[1] ** 2.0))
        # robot is too close to the person
        if dist[1] < 0 and v_diff <= 0:
            return reward, reward
        elif dist[1] > 0 and v_diff > 0:
            return -reward, -reward
        else:
            return 0, 0

    def get_simulated_cann_spikes(self,
                                  step: int,
                                  noise_index: List = None,
                                  noise_time: List = None) -> (np.array, np.array):
        """
        Function returns the spike indices and times for the simulated CANN,
        that can be fed into the SNN.
        :param step: the current simulation step, how often it was repeated
        :param noise_index: optional list with index for neurons that send spikes as noise
        :param noise_time: optional list with times for neurons that send spikes as noise. Has to correspond with the
                           list for noise indices
        """
        last = self.person_simulation.get_polar_path_adapted()[-1]
        angle = last[0]
        radius = last[1]
        return self.cann_simulation.simulate_spikes(step=step,
                                                    angle=angle,
                                                    radius=radius,
                                                    noise_index=noise_index,
                                                    noise_time=noise_time)

    def set_path(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Function sets a path the person has to follow
        :param x: cartesian x coordinates for the path in meters
        :param y: corresponding cartesian y coordinates for the path in meters
        """
        self.person_simulation.turn_angle_time_step = 0
        self.person_simulation.turn_probability = 0
        self.person_simulation.min_wait_turn_steps = 0
        self.person_simulation.min_turn_angle = 0
        self.person_simulation.max_turn_angle = 0
        self.person_simulation.set_path(x, y)

    def reset(self, radius_random: bool = False, angle_random: bool = False, start_angle_random: bool = False):
        """
        Function resets the animations and visualization in case the person walked out of the radar area.
        :param radius_random: if true, the person will be placed in a random distance from the robot
        :param angle_random: if true, the person will be placed at a random angle
        :param start_angle_random: if true, the person will start walking at a random angle
        """
        radius = self._starting_pos_person_radius
        angle = self._starting_pos_person_angle

        if radius_random:
            radius = self._ratio_radius * self._number_neurons_y * np.random.random_sample()

        if angle_random:
            angle = 180 * np.random.random_sample() - 90

        if start_angle_random:
            angle = 360 * np.random.random_sample()
            self.person_simulation.goal_angle = self.person_simulation.current_angle
        else:

            self.person_simulation.goal_angle = 0
            self.person_simulation.current_angle = 0

        x = radius * np.sin(math.radians(angle))
        y = radius * np.cos(math.radians(angle))

        self.robot_simulation.reset()
        self.cann_simulation.simulate_spikes(step=0, angle=angle, radius=radius)
        self.person_simulation.reset(x=x, y=y)

        if not self._show_simulation_visualization:
            return
        self.visualization.update(step=0,
                                  cartesian_path_person_adapted=self.person_simulation.get_cartesian_path_adapted(),
                                  polar_path_person=self.person_simulation.get_polar_path_adapted(),
                                  cartesian_path_robot=self.robot_simulation.path,
                                  cartesian_path_person=self.person_simulation.path_cartesian,
                                  cann_spikes=self.cann_simulation.previous_spikes_probability,
                                  angle_person=self.person_simulation.current_angle,
                                  angle_robot=self.robot_simulation.current_angle,
                                  v_robot=(self.robot_simulation.v_left + self.robot_simulation.v_right) / 2,
                                  v_person=self.person_simulation.person_velocity)

    # Some properties to access often needed values more easily
    @property
    def person_path_cartesian(self) -> List:
        return self.person_simulation.path_cartesian

    @property
    def robot_path_cartesian(self) -> List:
        return self.robot_simulation.path

    @property
    def robot_velocity(self) -> float:
        return (self.robot_simulation.v_left + self.robot_simulation.v_right) / 2

    @property
    def person_velocity(self) -> float:
        return self.person_simulation.person_velocity

    @property
    def robot_angle(self) -> float:
        return self.robot_simulation.current_angle

    @property
    def person_angle(self) -> float:
        return self.person_simulation.current_angle

    @property
    def distance(self) -> (float, float):
        """
        property contains the angular and radius distance to the optimal position of the person
        """
        p_person = self.person_path_cartesian[-1]
        p_robot = self.robot_path_cartesian[-1]
        d = round(p_person[0] - p_robot[0], 10), round(p_person[1] - p_robot[1], 10)

        # subtract the distance that should be kept to the person
        difference_edited = d[0] - self._distance_to_person_x, d[1] - self._distance_to_person_y
        r = np.sqrt(difference_edited[0] ** 2 + difference_edited[1] ** 2)
        d = np.degrees(np.arctan2(difference_edited[0], abs(difference_edited[1])))
        return r, d

    def _stop_animation(self) -> bool:
        """
        Function checks if the animation should be stopped, because the person left the visible area
        :return: True if the person left the visible area
        """
        p = np.array(self.person_simulation.path_cartesian[-1])
        p_person = self.robot_simulation.get_rotated_points_cartesian(p)
        r = np.sqrt(p_person[0] ** 2 + p_person[1] ** 2)
        d = np.degrees(np.arctan2(p_person[0], p_person[1]))
        if r <= 0 or r > self._number_neurons_y * self._ratio_radius or d >= 90.0 or d <= -90.0:
            print("position person:", p_person, "d:", d, "r:", r)
            return True
        return False

    def _difference(self) -> (float, float):
        """
        Calculates the difference in cartesian coordinates between the robot and the person.
        :return x and y absolute difference
        """
        p_person = self.person_simulation.path_cartesian[-1]
        p_robot = self.robot_simulation.path[-1]
        return round(abs(p_person[0] - p_robot[0]), 10), round(abs(p_person[1] - p_robot[1]), 10)
