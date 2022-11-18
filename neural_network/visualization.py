from typing import Tuple, List
from brian2 import ms

import matplotlib.pyplot as plt
import numpy as np
import configparser


def i_off_snn():
    plt.ioff()
    plt.tight_layout()
    plt.show()


class Visualization:
    """
    Class plots information about the SNN during training and simulation.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        :param config: read config file
        """

        self.figure = plt.figure(figsize=(7, 6), facecolor='#FFFFFF')
        plt.rcParams.update({'font.size': 7})

        self._update_rate = int(config['setup']['snn_visualization_update_rate'])
        self._simulation_duration = float(config['simulation']['simulation_step_time'])
        self._cann_neurons_x = int(config['CANN']['number_neurons_x'])
        self._cann_neurons_y = int(config['CANN']['number_neurons_y'])

        n = self._cann_neurons_x * self._cann_neurons_y

        # input spikes
        # noinspection PyTypeChecker
        self._input_spike_ax = plt.subplot(4, 4, (1, 4))
        self._input_spike_ax.set_xlabel("Spike time in ms")
        self._input_spike_ax.set_xlim([0, self._simulation_duration * ms])
        self._input_spike_ax.set_ylabel("Neuron Index")
        self._input_spike_ax.title.set_text("Input Data Spikes")
        self._input_spike_ax.set_ylim([-0.1, n + 0.1])
        self._input_spike_ax.set_yticks(np.arange(0, n, n / 10))
        self._input_spike_plot, = self._input_spike_ax.plot([], [], 'ob')

        # output spikes
        # noinspection PyTypeChecker
        self._output_spike_ax = plt.subplot(4, 4, (5, 8))
        self._output_spike_ax.set_xlabel("Spike time in ms")
        self._output_spike_ax.set_xlim([0, self._simulation_duration * ms])
        self._output_spike_ax.set_ylabel("Neuron Index")
        self._output_spike_ax.title.set_text("Output Neuron Spikes")
        self._output_spike_ax.set_ylim([-0.1, 1.1])
        self._output_spike_ax.set_yticks([0, 1])
        self._output_spike_plot, = self._output_spike_ax.plot([], [], 'ob')

        w_min = float(config['velocity.RSTDP']['w_min']) / 1000
        w_max = float(config['velocity.RSTDP']['w_max']) / 1000
        c_min = float(config['velocity.RSTDP']['c_min']) / 1000
        c_max = float(config['velocity.RSTDP']['c_max']) / 1000
        b_values = np.linspace(0, 1, num=self._cann_neurons_x * self._cann_neurons_y)
        b_values = b_values.reshape(self._cann_neurons_x, self._cann_neurons_y)

        # weights left wheel R-STDP
        ax3 = plt.subplot(4, 4, 9)
        ax3.set_xticks(np.round(np.linspace(0, self._cann_neurons_x, num=6), 0))
        ax3.set_yticks(np.round(np.linspace(0, self._cann_neurons_y, num=6), 0))
        self._weights_left_wheel = ax3.imshow(b_values, cmap='hot', interpolation='nearest', vmin=w_min, vmax=w_max)
        plt.colorbar(self._weights_left_wheel, ax=ax3)

        # weights left wheel R-STDP
        ax4 = plt.subplot(4, 4, 10)
        ax4.set_xticks(np.round(np.linspace(0, self._cann_neurons_x, num=6), 0))
        ax4.set_yticks(np.round(np.linspace(0, self._cann_neurons_y, num=6), 0))
        self._weights_right_wheel = ax4.imshow(b_values, cmap='hot', interpolation='nearest', vmin=w_min, vmax=w_max)
        plt.colorbar(self._weights_right_wheel, ax=ax4)

        # eligibility trace left wheel R-STDP
        ax5 = plt.subplot(4, 4, 11)
        ax5.set_xticks(np.round(np.linspace(0, self._cann_neurons_x, num=6), 0))
        ax5.set_yticks(np.round(np.linspace(0, self._cann_neurons_y, num=6), 0))
        self._eli_trace_left_wheel = ax5.imshow(b_values, cmap='hot', interpolation='nearest', vmin=c_min, vmax=c_max)
        plt.colorbar(self._eli_trace_left_wheel, ax=ax5)

        # eligibility trace right wheel R-STDP
        ax6 = plt.subplot(4, 4, 12)
        ax6.set_xticks(np.round(np.linspace(0, self._cann_neurons_x, num=6), 0))
        ax6.set_yticks(np.round(np.linspace(0, self._cann_neurons_y, num=6), 0))
        self._eli_trace_right_wheel = ax6.imshow(b_values, cmap='hot', interpolation='nearest', vmin=c_min, vmax=c_max)
        plt.colorbar(self._eli_trace_right_wheel, ax=ax6)

        w_min = float(config['distance.RSTDP']['w_min']) / 1000
        w_max = float(config['distance.RSTDP']['w_max']) / 1000
        c_min = float(config['distance.RSTDP']['c_min']) / 1000
        c_max = float(config['distance.RSTDP']['c_max']) / 1000
        a = [[0], [1]]
        # weights distance further
        ax7 = plt.subplot(4, 4, 13)
        ax7.set_xticks([0])
        ax7.set_yticks([0, 1])
        self._weights_distance_further = ax7.imshow(a, cmap='hot', interpolation='nearest', vmin=w_min, vmax=w_max)
        plt.colorbar(self._weights_distance_further, ax=ax7)

        # weights distance closer
        ax8 = plt.subplot(4, 4, 14)
        ax8.set_xticks([0])
        ax8.set_yticks([0, 1])
        self._weights_distance_closer = ax8.imshow(a, cmap='hot', interpolation='nearest', vmin=w_min, vmax=w_max)
        plt.colorbar(self._weights_distance_closer, ax=ax8)

        # eligibility trace distance further
        ax9 = plt.subplot(4, 4, 15)
        ax9.set_xticks([0])
        ax9.set_yticks([0, 1])
        self._eli_trace_distance_further = ax9.imshow(a, cmap='hot', interpolation='nearest', vmin=c_min, vmax=c_max)
        plt.colorbar(self._eli_trace_distance_further, ax=ax9)

        # eligibility trace distance closer
        ax10 = plt.subplot(4, 4, 16)
        ax10.set_xticks([0])
        ax10.set_yticks([0, 1])
        self._eli_trace_distance_closer = ax10.imshow(a, cmap='hot', interpolation='nearest', vmin=c_min, vmax=c_max)
        plt.colorbar(self._eli_trace_distance_closer, ax=ax10)

        plt.ion()
        plt.tight_layout()

    def update(self,
               step: int,
               input_spikes: Tuple[List, List],
               output_spikes: Tuple[np.ndarray, np.ndarray],
               weights_left_wheel: np.ndarray,
               weights_right_wheel: np.ndarray,
               eli_trace_left_wheel: np.ndarray,
               eli_trace_right_wheel: np.ndarray,
               weights_distance_further: np.ndarray,
               weights_distance_closer: np.ndarray,
               eli_trace_distance_further: np.ndarray,
               eli_trace_distance_closer: np.ndarray):
        """
        Function updates the plots visualizing the current SNN status.
        :param step: the current simulation step
        :param input_spikes: the input spikes for the network as a tuple (spike indices, spike times)
        :param output_spikes: the output velocity spikes as a tuple (spike indices, spike times)
        :param weights_left_wheel: the synapses weights connecting the CANN to the left wheel neuron in mV without unit
        :param weights_right_wheel: the synapses weights connecting the CANN to the right wheel neuron in
                                    mV without unit
        :param eli_trace_left_wheel: the eligibility trace for the synapses weights connecting the CANN
                                     to the left wheel neuron in mV without unit
        :param eli_trace_right_wheel: the eligibility trace for the synapses weights connecting the CANN
                                      to the left wheel neuron in mV without unit
        :param weights_distance_further: the synapses weights connecting the further distance neuron to the
                                         velocity neurons in mV without unit
        :param weights_distance_closer: the synapses weights connecting the closer distance neuron to the
                                        velocity neurons in mV without unit
        :param eli_trace_distance_further: the eligibility trace for the synapses connecting the further distance
                                           neuron to the velocity neurons in mV without unit
        :param eli_trace_distance_closer: the eligibility trace for the synapses connecting the closer distance
                                          neuron to the velocity neurons in mV without unit
        """
        if step % self._update_rate != 0:
            return
        self._update_input_spike_plot(step=step, input_spikes=input_spikes)
        self._update_output_spike_plot(step=step, output_spikes=output_spikes)
        self._update_weights_left_wheel(weights_left_wheel=weights_left_wheel)
        self._update_weights_right_wheel(weights_right_wheel=weights_right_wheel)
        self._update_eli_trace_left_wheel(eli_trace_left_wheel=eli_trace_left_wheel)
        self._update_eli_trace_right_wheel(eli_trace_right_wheel=eli_trace_right_wheel)
        self._update_weights_distance_further(weights_distance_further=weights_distance_further)
        self._update_weights_distance_closer(weights_distance_closer=weights_distance_closer)
        self._update_eli_trace_distance_further(eli_trace_distance_further=eli_trace_distance_further)
        self._update_eli_trace_distance_closer(eli_trace_distance_closer=eli_trace_distance_closer)

    def _update_input_spike_plot(self, step: int, input_spikes: Tuple[List, List]):
        self._input_spike_ax.set_xlim([self._simulation_duration * step,
                                       self._simulation_duration * (step + 1)])
        self._input_spike_plot.set_data(input_spikes[1], input_spikes[0])

    def _update_output_spike_plot(self, step: int, output_spikes: Tuple[np.ndarray, np.ndarray]):
        self._output_spike_ax.set_xlim([self._simulation_duration * step * ms,
                                        self._simulation_duration * (step + 1) * ms])
        self._output_spike_plot.set_data(output_spikes[1], output_spikes[0])

    def _update_weights_left_wheel(self, weights_left_wheel: np.ndarray):
        weights_left_wheel = np.reshape(weights_left_wheel, (self._cann_neurons_x, self._cann_neurons_x)) / 1000
        self._weights_left_wheel.set_data(weights_left_wheel)

    def _update_weights_right_wheel(self, weights_right_wheel: np.ndarray):
        weights_right_wheel = np.reshape(weights_right_wheel, (self._cann_neurons_x, self._cann_neurons_x)) / 1000
        self._weights_right_wheel.set_data(weights_right_wheel)

    def _update_eli_trace_left_wheel(self, eli_trace_left_wheel: np.ndarray):
        eli_trace_left_wheel = np.reshape(eli_trace_left_wheel, (self._cann_neurons_x, self._cann_neurons_x)) / 1000
        self._eli_trace_left_wheel.set_data(eli_trace_left_wheel)

    def _update_eli_trace_right_wheel(self, eli_trace_right_wheel: np.ndarray):
        eli_trace_right_wheel = np.reshape(eli_trace_right_wheel, (self._cann_neurons_x, self._cann_neurons_x)) / 1000
        self._eli_trace_right_wheel.set_data(eli_trace_right_wheel)

    def _update_weights_distance_further(self, weights_distance_further: np.ndarray):
        self._weights_distance_further.set_data(np.reshape(weights_distance_further, (2, 1)) / 1000)

    def _update_weights_distance_closer(self, weights_distance_closer: np.ndarray):
        self._weights_distance_closer.set_data(np.reshape(weights_distance_closer, (2, 1)) / 1000)

    def _update_eli_trace_distance_further(self, eli_trace_distance_further: np.ndarray):
        self._eli_trace_distance_further.set_data(np.reshape(eli_trace_distance_further, (2, 1)) / 1000)

    def _update_eli_trace_distance_closer(self, eli_trace_distance_closer: np.ndarray):
        self._eli_trace_distance_closer.set_data(np.reshape(eli_trace_distance_closer, (2, 1)) / 1000)

    def close_plot(self):
        plt.close(self.figure)
