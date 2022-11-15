import numpy as np
from scipy.stats import multivariate_normal
import configparser


class CANNSimulation:
    """
    This class simulates the output spike indices and corresponding times for a CANN.
    """

    # indicates how wide the spike circle ist
    covariance_matrix = np.array(([20, 0], [0, 20]))

    previous_spikes_probability = None

    def __init__(self, config: configparser.ConfigParser):
        """
        :param config: the config file with parameters for the simulation
        """
        self._cann_neurons_x = int(config['CANN']['number_neurons_x'])
        self._cann_neurons_y = int(config['CANN']['number_neurons_y'])
        self._ratio_radius = float(config['CANN']['ratio_radius'])
        self._simulation_step_time = int(config['simulation']['simulation_step_time'])

        # needed for the visualization
        self.previous_spikes_probability = np.linspace(0, 1, num=self._cann_neurons_x * self._cann_neurons_y)
        self.previous_spikes_probability = self.previous_spikes_probability.reshape(self._cann_neurons_x,
                                                                                    self._cann_neurons_y)

    def simulate_spikes(self,
                        step: int,
                        angle: float,
                        radius: float,
                        noise_index: np.ndarray = None,
                        noise_time: np.ndarray = None) -> (np.array, np.array):
        """
        Function simulates the output around a given position for a CANN with the help of a 2D gaussian distribution
        :param step: the current simulation step, how often it was repeated
        :param angle: the angle position of the person, center of the spike circle of the CANN
        :param radius: the radius position of the person, center of the spike circle of the CANN
        :param noise_index: optional list with index for neurons that send spikes as noise
        :param noise_time: optional list with times for neurons that send spikes as noise. Has to correspond with the
                           list for noise indices
        :return: the times and corresponding indices of spiking neurons for the given position and step
        """
        if angle > 90 or angle < -90 or radius < 0 or radius > self._ratio_radius * self._cann_neurons_x:
            return self.previous_spikes_probability
        # calculate center of spike circle
        gaussian_mean = np.array([round((angle + 90) * self._cann_neurons_x / 180), round(radius / self._ratio_radius)])

        x, y = np.meshgrid(np.arange(0, self._cann_neurons_x, 0.1), np.arange(0, self._cann_neurons_y, 0.1))

        pos = np.dstack((x, y))
        rv = multivariate_normal(gaussian_mean, self.covariance_matrix)
        pdf = rv.pdf(pos)
        pdf = np.add.reduceat(pdf, np.arange(0, len(pdf[0]), 10), axis=1)
        pdf = np.add.reduceat(pdf, np.arange(0, len(pdf), 10), axis=0)

        pdf = np.sinh(np.exp(pdf))

        min_pdf = np.min(pdf)
        max_pdf = np.max(pdf)

        # scale to a range between 0 and 1 spikes in the frame
        pdf = (pdf - min_pdf) / (max_pdf - min_pdf)

        # adjust spike rates to be more plausible with the real input
        pdf = np.where(pdf < 0.4286, 0, pdf)
        pdf = np.where(pdf > 0.619, pdf ** 2, pdf * 0.8)

        t_end = self._simulation_step_time * (step + 1) - 0.1
        t_total = self._simulation_step_time - 0.2
        times = np.array([])
        indices = np.array([])

        for index, x in enumerate(pdf.flatten()):
            if x == 0:
                continue
            times = np.append(times, t_end - x * t_total)
            indices = np.append(indices, index)

        if noise_index and len(noise_index) > 0:
            indices = np.append(indices, noise_index)
            times = np.append(times, noise_time)
        indices = np.array(indices).flatten()
        times = np.array(times).flatten()

        self.previous_spikes_probability = pdf

        return indices, times
