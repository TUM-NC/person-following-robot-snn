import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.markers import MarkerStyle


def i_off_simulation():
    plt.ioff()
    plt.tight_layout()


class Visualization:
    """
    Class visualizes the output of the simulation.
     - Person path from robot's perspective in cartesian and polar
     - global path of robot and person
     - cann output
     - velocity and angle robot/person
    """
    def __init__(self, config):
        """
        initiates all the figures
        :param config: the config file with parameters for the simulation
        """
        self._cann_neurons_x = int(config['CANN']['number_neurons_x'])
        self._cann_neurons_y = int(config['CANN']['number_neurons_y'])
        self._ratio_radius = float(config['CANN']['ratio_radius'])
        self._update_rate = float(config['setup']['simulation_visualization_update_rate'])

        self._figure = plt.figure(figsize=(7, 6), facecolor='#FFFFFF')
        plt.rcParams.update({'font.size': 7})

        # subplot walking simulation cartesian
        ax1 = plt.subplot(231)
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(0, self._cann_neurons_y * self._ratio_radius)
        ax1.set_xlabel("X in m")
        ax1.set_ylabel("Distance from Radar in m")
        ax1.set_title("Walking Person Simulation Cartesian Coordinates")
        self._path_walking_line_cartesian, = ax1.plot([])

        # subplot walking simulation polar
        ax2 = plt.subplot(232)
        ax2.set_xlim(-90, 90)
        ax2.set_ylim(0, self._cann_neurons_y * self._ratio_radius)
        ax2.set_xlabel("Radar Angle")
        ax2.set_ylabel("Distance from Radar in m")
        ax2.set_title("Walking Person Simulation Polar Coordinates")
        self._path_walking_line_polar, = ax2.plot([])

        # subplot CANN simulation
        ax3 = plt.subplot(233)
        ax3.set_xlim(-0.5, self._cann_neurons_x - 0.5)
        ax3.set_ylim(-0.5, self._cann_neurons_y - 0.5)
        ax3.set_xlabel("Radar Angle")
        ax3.set_ylabel("Distance from Radar in m")
        ax3.set_title("CANN output simulation")
        # labels for the CANN simulation, so that they match the labels in the master thesis for the CANN
        x_labels = ['-75', '-50', '-25', '0', '25', '50', '75']
        step_w = self._cann_neurons_x / 180
        x_ticks = np.around(np.array([15, 40, 65, 90, 115, 140, 165]) * step_w - 0.5, 2)
        end = self._ratio_radius * self._cann_neurons_y
        y_labels = list(map(str, np.append(np.arange(0, end, end / 5), [end])))
        y_ticks = np.around(np.arange(-0.5, self._cann_neurons_y + 0.5, self._cann_neurons_y / 5), 2)
        ax3.set(yticks=y_ticks, yticklabels=y_labels, xticks=x_ticks, xticklabels=x_labels)

        output_spikes = np.linspace(0, 1, num=self._cann_neurons_x * self._cann_neurons_y)
        output_spikes = output_spikes.reshape(self._cann_neurons_x, self._cann_neurons_y)
        self._spiking_simulation = ax3.imshow(output_spikes, cmap='hot', interpolation='nearest')

        self._ax_global_plot = plt.subplot(234)
        self._ax_global_plot.set_xlabel("X in m")
        self._ax_global_plot.set_ylabel("Y in m")
        self._ax_global_plot.set_title("Global coordinates - robot and Person")
        self._global_plot_person = self._ax_global_plot.scatter([0], [4], color="y", marker=(3, 0, 90), label="Person")
        self._global_plot_person_path, = self._ax_global_plot.plot([], color="y")
        self._global_plot_robot = self._ax_global_plot.scatter([0], [0], color="b", marker=(3, 0, -90), label="robot")
        self._global_plot_robot_path, = self._ax_global_plot.plot([], color="b")
        self._ax_global_plot.legend(loc="upper right")
        self._ax_global_plot.grid()

        self._ax_information = plt.subplot(235)
        self._ax_information.set_xlabel("")
        self._ax_information.set_ylabel("")
        self._ax_information.set_title("Status Information")
        self._ax_information_text = self._ax_information.text(0.1, 0.4, "")

        plt.ion()
        plt.tight_layout()
        plt.show()

    def update(self,
               step: int,
               cartesian_path_person_adapted: np.ndarray,
               polar_path_person: np.ndarray,
               cartesian_path_robot: np.ndarray,
               cartesian_path_person: np.ndarray,
               cann_spikes: np.ndarray,
               angle_person,
               angle_robot,
               v_robot,
               v_person):
        """
        Function updates all the figures with new information
        :param step: the current step of the simulation, to check if the plot should be updated
        :param cartesian_path_person_adapted: last 100 steps of the person from the robots
                                              current perspective in cartesian
        :param polar_path_person: last 100 steps of the person from the robots current perspective in polar
        :param cartesian_path_robot: the total path of the robot in cartesian
        :param cartesian_path_person: the total path of the person in cartesian
        :param cann_spikes: the probability of the simulated cann neurons if they spike
        :param angle_person: the current angle of the person
        :param angle_robot: the current angle of the robot
        :param v_robot: the current velocity of the robot
        :param v_person: the current velocity of the person
        """
        pos_person = cartesian_path_person[-1]
        pos_robot = cartesian_path_robot[-1]
        if step % self._update_rate == 0:
            self._update_plot_cartesian(cartesian_path_person_adapted)
            self._update_plot_polar(polar_path_person)
            self._update_plot_cann(cann_spikes)
            self._update_plot_global(pos_person,
                                     pos_robot,
                                     angle_person,
                                     angle_robot,
                                     cartesian_path_person,
                                     cartesian_path_robot)
            self._update_plot_information(v_robot, v_person, angle_robot, angle_person)
            self._figure.canvas.draw()
            self._figure.canvas.flush_events()

    def _update_plot_cartesian(self, cartesian_path):
        self._path_walking_line_cartesian.set_data(cartesian_path.T)

    def _update_plot_polar(self, polar_path):
        self._path_walking_line_polar.set_data(polar_path.T)

    def _update_plot_cann(self, cann_spikes: np.ndarray):
        self._spiking_simulation.set_data(cann_spikes)

    def _update_plot_global(self,
                            pos_person,
                            pos_robot,
                            angle_person,
                            angle_robot,
                            cartesian_path_person,
                            cartesian_path_robot):
        self._ax_global_plot.set_xlim(min(pos_robot[0], pos_person[0]) - 5, max(pos_robot[0], pos_person[0]) + 5)
        self._ax_global_plot.set_ylim(min(pos_robot[1], pos_person[1]) - 5, max(pos_robot[1], pos_person[1]) + 5)

        new_m = MarkerStyle("^")
        # noinspection PyProtectedMember
        new_m._transform.rotate_deg(math.degrees(angle_robot))
        self._global_plot_robot.set_paths([new_m.get_path().transformed(new_m.get_transform())])

        new_m = MarkerStyle("^")
        # noinspection PyProtectedMember
        new_m._transform.rotate_deg(angle_person)
        self._global_plot_person.set_paths([new_m.get_path().transformed(new_m.get_transform())])

        self._global_plot_robot.set_offsets([pos_robot])
        self._global_plot_person.set_offsets([pos_person])

        self._global_plot_person_path.set_data(np.array(cartesian_path_person).T)
        self._global_plot_robot_path.set_data(np.array(cartesian_path_robot).T)

    def _update_plot_information(self, v_robot, v_person, angle_robot, angle_person):
        s = "\nAngle robot: " + str(math.degrees(angle_robot))
        s += "\nVelocity robot: " + str(v_robot) + "\n"
        s += "Angle Person: " + str(angle_person) + "\n"
        s += "Velocity Person: " + str(v_person) + "\n"
        self._ax_information_text.set_text(s)
