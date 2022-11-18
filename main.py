import os
import random

import numpy as np
import configparser
import matplotlib.pyplot as plt

from neural_network.visualization import i_off_snn
from simulation.simulation import Simulation
from simulation.visualization import i_off_simulation
from neural_network.snn import SNN
from evaluate import Evaluate


def train(weights_file_name_cann: str = None, weights_file_name_dist: str = None):
    """
    Trains the neural network by calling the corresponding methods from the snn and simulation.
    Order:
        1. 0-1500 steps the person walks in a straight line and the robot should follow the person
        2. 1501-4000 The Person gets placed at a random distance in front of
           the robot at a 0° angle. If the robot reaches the goal, the simulation is reset.
        3. 4001-20000 The person gets placed randomly in the radar area and every
           1000 steps the turn probability 4.7.1 for the random walk is
           increased by 1.2%. If the robot reaches the goal, the simulation is reset.
    :param weights_file_name_cann: The name for the weights for the CANN synapses.
                                   If not specified, the default weights are used.
    :param weights_file_name_dist: The name for the weights for the distance synapses.
                                   If not specified, the default weights are used.
    """
    global reward_left
    global reward_right
    global current_iteration_steps

    if weights_file_name_cann and weights_file_name_dist:
        snn.set_weights(path_weights_cann="../weights/" + weights_file_name_cann,
                        path_weights_dist="../weights/" + weights_file_name_dist)

    for step in range(20000):
        """
        1. simulate the next animation step and check if the animation needs to be stopped, 
        because the person left the radar area
        """
        if simulation.animate_next_step(step):
            print("Animation was reset")
            simulation.reset()
            snn.reset()

        """ 2. get the output spikes from the simulated CANN """
        indices_sim, times_sim = simulation.get_simulated_cann_spikes(step)

        """ 3. set the output spikes from the CANN in the SNN """
        snn.set_input_spikes(indices_sim, times_sim)

        """ 4. Run the SNN and get output velocities"""
        snn.run_simulation(step=step)
        v_left, v_right = snn.get_predicted_velocity(step)

        """ 5. Simulate the vehicle movement with the calculated speed """
        simulation.robot_simulation.move(v_left, v_right)

        """ 6. Calculate the reward and save it for later """
        l_v_left, l_v_right = simulation.cann_reward()
        l_d_left, l_d_right = simulation.distance_reward()
        reward_left[step] = l_v_left
        reward_right[step] = l_v_right

        v_veh_l = simulation.robot_simulation.v_left
        v_veh_r = simulation.robot_simulation.v_right

        """ 7. Set reward spikes, if the reward is unequal to 0 in the SNN network """
        if step > 0:
            reward_smaller = l_v_left < reward_left[-2] and l_v_right < reward_right[-2]
        else:
            reward_smaller = False

        # train cann synapses
        if reward_smaller:
            if step % 2 == 0:
                snn.set_reward_spikes_cann(l_v_left, l_v_right, step)
            else:  # train distance synapses
                snn.set_reward_spikes_distance(l_d_left, l_d_right, step)

        if 1500 < step <= 4000:
            reach_goal_reset(step)
        elif 4000 < step:
            if step % 1000 == 0:
                simulation.person_simulation.turn_probability += 1.2
            reach_goal_reset(step)

        print("step:", step, "v_l:", v_veh_l, "v_r:", v_veh_r, "reward_v_l:", round(l_v_left, 10),
              "reward_v_r:", round(l_v_right, 10), "reward_d_l:", round(l_d_left, 10),
              "reward_d_r:", round(l_d_right, 10))


def reach_goal_reset(step: int):
    """
    Function resets the network if the robot manges to catch up with the person,
    or if the robot took too many steps trying to catch up.
    The robot is placed in a random location on the grid and with a random angle
    """
    global current_iteration_steps
    r, d = simulation.distance
    con_a = -0.3 < r < 0.3
    con_b = -2 < d < 2
    con_c = abs(simulation.robot_angle - simulation.person_angle) < 1
    if (con_a and con_b and con_c) or current_iteration_steps == max_current_iteration_steps:
        print("Animation was reset, because the robot reached its goal.")
        if 1500 < step <= 4000:
            simulation.reset(radius_random=True)
        elif 4000 < step:
            simulation.reset(radius_random=True, angle_random=True)
        snn.reset()
        current_iteration_steps = 0

    current_iteration_steps += 1


def test(weights_file_name_cann: str, weights_file_name_dist: str, i_steps: int):
    snn.set_weights(path_weights_cann="weights/" + weights_file_name_cann,
                    path_weights_dist="weights/" + weights_file_name_dist)

    simulation.person_simulation.turn_probability = 20
    simulation.person_simulation.change_angle = True
    simulation.person_simulation.path_cartesian = [[0, 5], [0, 5]]
    simulation.person_simulation.min_turn_angle = 20
    simulation.person_simulation.max_turn_angle = 40
    for step in range(i_steps):
        """
        1. simulate the next animation step and check if the animation needs to be stopped, 
        because the person left the radar area
        """
        if simulation.animate_next_step(step):
            print("Animation was reset")
            simulation.reset()
            snn.reset()
            break

        """ 2. get the output spikes from the simulated CANN """
        indices_sim, times_sim = simulation.get_simulated_cann_spikes(step)

        """ 3. set the output spikes from the CANN in the SNN """
        snn.set_input_spikes(indices_sim, times_sim)

        """ 4. Run the SNN and get output velocities"""
        snn.run_simulation(step=step)
        v_left, v_right = snn.get_predicted_velocity(step)
        # v_left, v_right = _bound_speed(v_left, v_right)

        """ 5. Simulate the vehicle movement with the calculated speed """
        simulation.robot_simulation.move(v_left, v_right)

        """ 6. Calculate the reward and save it for later """
        l_left, l_right = simulation.cann_reward()
        reward_left[step] = l_left
        reward_right[step] = l_right

        v_vehicle_l = simulation.robot_simulation.v_left
        v_vehicle_r = simulation.robot_simulation.v_right

        print("step:", step, "v_left:", v_vehicle_l, "v_right:", v_vehicle_r, v_left, v_right)

        change = random.randint(0, 2)
        p = simulation.person_simulation.person_velocity
        if change == 0 and p > 0:
            simulation.person_simulation.person_velocity -= 0.05
        elif change == 2 and p < 2:
            simulation.person_simulation.person_velocity += 0.05


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    train_network = config.getboolean(section='setup', option='train')
    test_network = config.getboolean(section='setup', option='test')
    evaluate_network = config.getboolean(section='setup', option='evaluate')

    if evaluate_network:
        e = Evaluate()
    elif train_network or test_network:
        simulation_step_time = int(config['simulation']['simulation_step_time'])
        distance_to_person_x = float(config['simulation']['desired_distance_person_x'])
        distance_to_person_y = float(config['simulation']['desired_distance_person_y'])

        simulation = Simulation(config)
        snn = SNN(config)

        current_iteration_steps = 0
        max_current_iteration_steps = 500
        reward_left, reward_right = [], []
        if test_network:
            iteration_steps = int(config['setup']['test_steps'])
            reward_left, reward_right = list(range(iteration_steps)), list(range(iteration_steps))
            test(weights_file_name_cann=config['setup']['name_cann_weights'],
                 weights_file_name_dist=config['setup']['name_dist_weights'],
                 i_steps=iteration_steps)
        elif train_network:
            reward_left, reward_right = list(range(20000)), list(range(20000))
            simulation.person_simulation.turn_probability = 0
            train(weights_file_name_cann=None, weights_file_name_dist=None)
            run_n = 1
            for file in os.listdir("weights"):
                run_n += 1
            np.savetxt('weights/weights_run_cann' + str(run_n / 2) + '.txt', snn.cann_weights, delimiter=',')
            np.savetxt('weights/weights_run_dist' + str(run_n / 2) + '.txt', snn.distance_weights, delimiter=',')
        i_off_snn()
        i_off_simulation()

        # Plot the reward
        x = np.arange(0, len(reward_left), 1)
        plt.plot(x, reward_left, label="reward left")
        plt.plot(x, reward_right, label="reward right")
        plt.show()
