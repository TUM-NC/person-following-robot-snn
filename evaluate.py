"""
This class is intended to simulate different scenarios and evaluate them

Every scenario
    repeat 10 times
    repeat every scenario with random speed

Scenarios
    1. Follow the person for 10 minutes, with random path
    2. Person walks straight, then a circle, with varying radius, and straight again
    3. Person stops
    4. Person starts at random position in the visible area for the robot, robot must catch up and follow for 20s
    5. Person follows the robot and over time the amount of random noise is increased until the robot loses the person

Performance measures
    Time the robot managed to keep up
    Overall distance to goal distance between person and robot over the simulation time
    Finding scenarios with maximum errors in the goal distance difference
    monitor random noise in relation to the error
    Monitor the amount of spikes, to calculate the energy consumption
"""

from simulation.simulation import Simulation
import configparser
from neural_network.snn import SNN
import numpy as np
import random
from typing import List
import csv


class Evaluate:

    def __init__(self):
        # should contain tuples with angular error and radius error
        self.error = []
        self.pos_person = []
        self.pos_robot = []
        self.v_person = []
        self.v_robot = []
        # the amount of steps taken, to check how long the robot managed to keep up
        self.step = 0
        # the amount of spikes emitted in each step
        self.emitted_spikes = []

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        scenario = int(self.config['setup']['evaluate_scenario'])

        self.simulation = None
        self.snn = None

        if scenario == 1:
            self.scenario_1()
        elif scenario == 2:
            self.scenario_2()
        elif scenario == 3:
            self.scenario_3()
        elif scenario == 4:
            self.scenario_4()
        elif scenario == 5:
            self.scenario_5()

    def scenario_1(self):
        """
        Follow the person for 10 mins, with random path
        """

        for run in range(10):
            steps = 6000
            if self.simulation and self.snn:
                self.simulation.visualization.close_plot()
                self.snn.close_plot()
            self.simulation = Simulation(self.config)
            self.snn = SNN(self.config)
            self.set_weights()
            self.simulation.person_simulation.turn_probability = 20
            self.simulation.person_simulation.change_angle = True
            self.simulation.person_simulation.path_cartesian = [[0, 5], [0, 5]]
            self.simulation.person_simulation.min_turn_angle = 20
            self.simulation.person_simulation.max_turn_angle = 40
            for step in range(steps):
                a = random.randint(0, 2)
                p = self.simulation.person_simulation.person_velocity
                if a == 0 and p > 0.05:
                    self.simulation.person_simulation.person_velocity -= 0.001
                elif a == 2 and p < 0.2:
                    self.simulation.person_simulation.person_velocity += 0.001
                if not self.iteration(step):
                    self.collect_data()
                    break
                self.collect_data()

            self.store_data(scenario=1, run=run)
            self.simulation.reset()
            self.snn.reset()

    def scenario_2(self):
        """
        Person walks straight, then a circle, with varying radius, and straight again
        """

        for run in range(10):
            if self.simulation and self.snn:
                self.simulation.visualization.close_plot()
                self.snn.close_plot()
            self.simulation = Simulation(self.config)
            self.snn = SNN(self.config)
            self.set_weights()
            radius = (run + 1) * 2
            theta = np.linspace(0, 2 * np.pi, 200)
            x = radius * np.cos(theta)
            y = (radius * np.sin(theta) + radius + 10)

            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 4)
            x = np.append(x, [radius])
            y = np.append(y, [100])

            self.simulation.set_path(x, y)
            step = 0
            while True:
                p = self.simulation.person_path_cartesian[-1]
                a = (not self.iteration(step) or (p[0] == radius and p[1] == 100))
                b = len(self.simulation.person_simulation.path_given_x) == 0
                if a or b:
                    self.collect_data()
                    break
                self.collect_data()
                step += 1

            self.store_data(scenario=2, run=run)
            self.simulation.reset()
            self.snn.reset()

    def scenario_3(self):
        """
        Person stops
        """

        for run in range(10):
            if self.simulation and self.snn:
                self.simulation.visualization.close_plot()
                self.snn.close_plot()
            self.simulation = Simulation(self.config)
            self.snn = SNN(self.config)
            self.set_weights()
            self.simulation.person_simulation.turn_probability = 0
            self.simulation.person_simulation.change_angle = False
            self.simulation.person_simulation.path_cartesian = [[0, 5], [0, 5]]
            self.simulation.person_simulation.min_turn_angle = 0
            self.simulation.person_simulation.max_turn_angle = 0
            steps_before_stopping = random.randint(10, 100)
            wait = random.randint(20, 100) + 10
            steps_after_stopping = 100
            step = 0
            while True:
                if not self.iteration(step) or steps_after_stopping + wait + steps_after_stopping < step:
                    self.collect_data()
                    break
                self.collect_data()

                a = steps_before_stopping < step < steps_before_stopping + wait
                if a and self.simulation.person_simulation.person_velocity > 0.01:
                    self.simulation.person_simulation.person_velocity -= 0.01
                elif step > steps_before_stopping + wait and self.simulation.person_simulation.person_velocity < 0.1:
                    self.simulation.person_simulation.person_velocity += 0.01
                step += 1

            self.store_data(scenario=3, run=run)
            self.simulation.reset()
            self.snn.reset()

    def scenario_4(self):
        """
        Person starts at random position in the visible area for the robot, robot must catch up and follow for 10s
        """
        for run in range(10, 60):
            if self.simulation and self.snn:
                self.simulation.visualization.close_plot()
                self.snn.close_plot()
            self.simulation = Simulation(self.config)
            self.snn = SNN(self.config)
            self.set_weights()
            self.simulation.reset(radius_random=True, angle_random=True)
            self.simulation.person_simulation.turn_probability = 0
            self.simulation.person_simulation.change_angle = False
            self.simulation.person_simulation.min_turn_angle = 0
            self.simulation.person_simulation.max_turn_angle = 0

            step = 0
            follow_count = 0
            reached = -1
            while follow_count < 100:
                if not self.iteration(step) or step > 800:
                    self.collect_data()
                    break
                self.collect_data()

                distance = self.simulation.distance
                if abs(distance[0]) < 1 and abs(distance[1]) < 10 and follow_count == 0:
                    print("reached")
                    reached = step
                    follow_count += 1
                if follow_count > 0:
                    follow_count += 1

                step += 1

            self.store_data(scenario=6, run=run, reached=reached)
            self.simulation.reset()
            self.snn.reset()

    def scenario_5(self):
        """
        Person follows the robot and over time the amount of random noise is increased until the robot loses the person
        """
        for run in range(10, 60):
            if self.simulation and self.snn:
                self.simulation.visualization.close_plot()
                self.snn.close_plot()
            self.simulation = Simulation(self.config)
            self.snn = SNN(self.config)
            self.set_weights()
            self.simulation.person_simulation.turn_probability = 0
            self.simulation.person_simulation.change_angle = True
            self.simulation.person_simulation.min_turn_angle = 0
            self.simulation.person_simulation.max_turn_angle = 0
            self.simulation.person_simulation.path_cartesian = [[0, 5], [0, 5]]
            step = 0
            noise = 0
            while True:
                if step % 20 == 0:
                    noise += 2
                noise_index = random.sample(range(0, 1600), noise)
                noise_time = np.array(random.sample(range(1, 100), noise)) + step * 100
                if not self.iteration(step, noise_index, noise_time):
                    self.collect_data(noise_spikes=noise)
                    break
                self.collect_data()
                step += 1

            self.store_data(scenario=5, run=run, reached=noise)
            self.simulation.reset()
            self.snn.reset()

    def iteration(self, step, noise_index: List = None, noise_time: List = None):
        """
        function simulates one iteration step
        """
        if self.simulation.animate_next_step(step):
            print("Animation was reset")
            return False

        """ 2. get the output spikes from the simulated CANN """
        indices_sim, times_sim = self.simulation.get_simulated_cann_spikes(step,
                                                                           noise_index=noise_index,
                                                                           noise_time=noise_time)

        """ 3. set the output spikes from the CANN in the SNN """
        self.snn.set_input_spikes(indices_sim, times_sim)

        """ 4. Run the SNN """
        self.snn.run_simulation(step=step)

        """ 5. Get the output spikes from the SNN and calculate the resulting speed """
        v_left, v_right = self.snn.get_predicted_velocity(step)

        """ 6. Simulate the robot movement with the calculated speed """
        self.simulation.robot_simulation.move(v_left, v_right)

        v_robot_l = self.simulation.robot_simulation.v_left
        v_robot_r = self.simulation.robot_simulation.v_right

        """ 8. Update the plotted data in the in the SNN plot """
        print("step:", step, "v_left:", v_robot_l, "v_right:", v_robot_r, v_left, v_right)
        return True

    def collect_data(self):
        p = np.array(self.simulation.person_simulation.path_cartesian[-1:])
        e = self.simulation.person_simulation.get_polar_path_adapted()
        self.error.append((e[0], e[1] - 4))

        self.pos_person.append(self.simulation.person_path_cartesian[-1])
        self.pos_robot.append(self.simulation.robot_path_cartesian[-1])

        self.v_person.append(self.simulation.person_velocity)
        self.v_robot.append((self.simulation.robot_simulation.v_left, self.simulation.robot_simulation.v_left))

        self.step += 1

    def store_data(self, scenario, run, reached: int = -1):
        with open('scenario_' + str(scenario) + "_run_" + str(run) + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Error", "pos_person", "pos_robot", "v_person", "v_robot"])
            for i in range(len(self.error)):
                writer.writerow([i,
                                 self.error[i],
                                 self.pos_person[i],
                                 self.pos_robot[i],
                                 self.v_person[i],
                                 self.v_robot[i]])

            if reached != -1:
                writer.writerow([i, reached, -1, -1, -1, -1, -1])
        self.error = []
        self.pos_person = []
        self.pos_robot = []
        self.v_person = []
        self.v_robot = []

    def set_weights(self):
        self.snn.set_weights(path_weights_cann="weights/weights_run_cann.txt",
                             path_weights_dist="weights/weights_run_dist.txt")

    @staticmethod
    def load_data(path) -> List:
        data = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                if len(row) != 4:
                    continue
                new_row = []
                for i in range(len(row)):
                    row[i] = row[i].replace("\"", "").replace("[", "").replace("]", "").replace("(", "")
                    row[i] = row[i].replace(")", "")
                    row[i] = row[i].split(",")
                new_row.append(int(row[0][0]))
                new_row.append(float(row[0][1]))
                new_row.append((float(row[0][1]), float(row[1][0])))
                new_row.append((float(row[1][1]), float(row[2][0])))
                new_row.append(float(row[2][1]))
                new_row.append((float(row[2][2]), float(row[3][0])))
                new_row.append(float(row[3][1]))
                data.append(new_row)
        return data
