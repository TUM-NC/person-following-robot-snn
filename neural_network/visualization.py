from typing import Tuple, List

import matplotlib.pyplot as plt
from brian2 import Synapses
import numpy as np
from brian2 import ms, mV
import configparser


def i_off_snn():
    plt.ioff()
    plt.tight_layout()
    plt.show()


class Visualization:
    """
    Class plots information about the SNN during the training and simulation. Data plotted depends on the network
    configuration and the data given to this class.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        :param config: read config file
        """
        self.neuron_monitors = []
        self.synapses_monitors = []
        self.input_monitors = []
        self.figure = plt.figure(figsize=(7, 6), facecolor='#FFFFFF')
        plt.rcParams.update({'font.size': 7})
        self.columns = 0

        self.simulation_duration = float(config['simulation']['simulation_step_time'])
        self.number_neurons_x = int(config['CANN']['number_neurons_x'])
        self.number_neurons_y = int(config['CANN']['number_neurons_y'])

        self.synapses_w_min = float(config['STDP']['w_min']) * mV
        self.synapses_w_max = float(config['STDP']['w_max']) * mV

        self.plots_created = False
        self.neuron_monitors_subplots = []
        self.synapses_monitors_subplots = []
        self.input_monitors_subplots = []

        self.start = 0
        self.end = self.simulation_duration

    def plot_data(self,
                  neuron_monitors: List[Tuple[Tuple[np.ndarray, np.ndarray], str]],
                  syn_monitors: List[Tuple[Synapses, str, List[str]]],
                  in_monitors: List[Tuple[Tuple[np.ndarray, np.ndarray], str]],
                  sim: int) -> None:
        """
        Function should be called from the outside and automatically handles plot creation or updating of the data.
        :param neuron_monitors: list of tuples containing a tuples with the spike indices and times and
                                the title for the plot
        :param syn_monitors: list of tuples containing the synapses, the title for the plot and the list with
                             values that should be plotted
        :param in_monitors: list of tuples containing a tuple that contains the spike times and indices and the
                            plot title
        :param sim: the simulation step
        """
        self.start = sim * self.simulation_duration * ms
        self.end = (sim + 1) * self.simulation_duration * ms

        if self.plots_created:
            self._update_data(neuron_monitors, syn_monitors, in_monitors)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        else:
            a = max([len(x[2]) * 2 for x in syn_monitors])
            self.columns = max([a, len(neuron_monitors), len(syn_monitors), len(in_monitors)])
            self._plot_neuron_monitors(neuron_monitors)
            self._plot_input_spikes(in_monitors)
            self._plot_synapses_monitors(syn_monitors)
            self.plots_created = True
            plt.ion()
            plt.tight_layout()
            plt.show()
        plt.pause(0.0001)

    def _plot_neuron_monitors(self, neuron_spikes: List[Tuple[Tuple[np.ndarray, np.ndarray], str]]) -> None:
        """
        :param neuron_spikes: list of tuples containing a tuples with the spike indices and times and
                                the title for the plot
        """
        for index, spikes_title in enumerate(neuron_spikes):
            spikes, plot_title = spikes_title
            ind, times = spikes
            a = plt.subplot(4, int(self.columns / 4), 2)
            line = self._plot_data(a, plot_title, ind, times, True, lim=2)
            self.neuron_monitors_subplots.append((a, line))

    def _plot_input_spikes(self, input_spikes: List[Tuple[Tuple[np.ndarray, np.ndarray], str]]) -> None:
        """
        :param input_spikes: list of tuples containing a tuple that contains the spike times and indices and the
                             plot title
        """
        for index, spikes_title in enumerate(input_spikes):
            spikes, plot_title = spikes_title
            ind, times = spikes
            a = plt.subplot(4, int(self.columns / 4), 1)
            n = self.number_neurons_x * self.number_neurons_y
            if index == 0:
                line = self._plot_data(a, plot_title, np.array(ind[:]), np.array(times[:]), True, n)
            else:
                line = self._plot_data(a, plot_title, np.array(ind[:]), np.array(times[:]), True, n)
            self.input_monitors_subplots.append((a, line))

    def _plot_data(self,
                   subplot,
                   title: str,
                   ind: np.ndarray,
                   times: np.ndarray,
                   set_y_lim: bool = False,
                   lim: int = 2) -> plt.Line2D:
        """
        :param subplot: the subplot object in which data should be plotted
        :param title: The title for the plot
        :param ind: the indices for y-axis
        :param times: the times for the x-axis
        :param set_y_lim: true, if the range from the y-axis should be set with between 0.1 and lim -0.1
        :param lim: upper bound for y limit
        """
        line, = subplot.plot(times, ind, 'ob')
        subplot.set_xlim([0, self.simulation_duration * ms])
        if set_y_lim:
            subplot.set_ylim([-0.1, lim - 0.9])
            if lim > 20:
                subplot.set_yticks(np.round(np.arange(0, lim, lim / 10), 0))
            else:
                subplot.set_yticks(np.arange(0, lim, 1, dtype=int))
        subplot.title.set_text(title)
        subplot.set_ylabel("Neuron Index")
        subplot.set_xlabel("Spike time in s")
        return line

    def _update_data(self,
                     neuron_monitors: List[Tuple[Tuple[np.ndarray, np.ndarray], str]],
                     syn_monitors: List[Tuple[Synapses, str, List[str]]],
                     in_monitors: List[Tuple[Tuple[np.ndarray, np.ndarray], str]]) -> None:
        """
        :param neuron_monitors: list of tuples containing a tuples with the spike indices and times and
                                the title for the plot
        :param syn_monitors: list of tuples containing the synapses, the title for the plot and the list with
                             values that should be plotted
        :param in_monitors: list of tuples containing a tuple that contains the spike times and indices and the
                            plot title
        """
        a = [(neuron_monitors, self.neuron_monitors_subplots), (in_monitors, self.input_monitors_subplots)]
        for m in a:
            spikes_titles, sub_plt = m
            for index, spikes_title in enumerate(spikes_titles):
                p, line = sub_plt[index]
                p.set_xlim([self.start, self.end])

                spikes, plot_title = spikes_title
                ind, times = spikes
                line.set_data(times, ind)

        self._update_synapses_monitors(syn_monitors)

    def _update_synapses_monitors(self, syn_monitors: List[Tuple[Synapses, str, List[str]]]) -> None:
        """
        :param syn_monitors: list of tuples containing the synapses, the title for the plot and the list with
                             values that should be plotted
        """
        plot_index = 0
        for index, monitor_title_variables in enumerate(syn_monitors):
            monitor, plot_title, variables = monitor_title_variables
            for i, variable in enumerate(variables):
                # noinspection PyUnusedLocal
                p_1 = self.synapses_monitors_subplots[plot_index]
                # noinspection PyUnusedLocal
                p_2 = self.synapses_monitors_subplots[plot_index + 1]
                l_1 = "a = np.array(monitor." + variable + "[:])\n"
                l_1 += "syn_neuron_one = a[0::2]\n"
                if index == 0:
                    l_1 += "syn_neuron_one = np.reshape(syn_neuron_one, (self.number_neurons_x, " \
                            "self.number_neurons_y))\n"
                else:
                    l_1 += "syn_neuron_one = np.reshape(syn_neuron_one, (2, 1))\n"
                l_1 += "p_1.set_data(syn_neuron_one)\n"

                l_1 += "syn_neuron_two = a[1::2]\n"
                if index == 0:
                    l_1 += "syn_neuron_two = np.reshape(syn_neuron_two, (self.number_neurons_x, " \
                           "self.number_neurons_y))\n"
                else:
                    l_1 += "syn_neuron_two = np.reshape(syn_neuron_two, (2, 1))\n"
                l_1 += "p_2.set_data(syn_neuron_two)\n"
                plot_index += 2
                exec(l_1)

    def _plot_synapses_monitors(self, syn_monitors: List[Tuple[Synapses, str, List[str]]]) -> None:
        """
        :param syn_monitors: list of tuples containing the synapses, the title for the plot and the list with
                             values that should be plotted
        """
        for index, monitor_title_variables in enumerate(syn_monitors):
            monitor, plot_title, variables = monitor_title_variables
            for i, variable in enumerate(variables):
                plot_buffer = []
                l_1 = "a = np.array(monitor." + variable + "[:])\n"
                l_1 += "syn_neuron_one = a[0::2]\n"
                if index == 0:
                    l_1 += "syn_neuron_one = np.reshape(syn_neuron_one, (self.number_neurons_x, " \
                           "self.number_neurons_y))\n"
                else:
                    l_1 += "syn_neuron_one = np.reshape(syn_neuron_one, (2, 1))\n"
                l_1 += "sub_plt_one = plt.subplot(4, self.columns, 1 + ((2 + index) * self.columns) + 2 * i)\n"
                l_1 += "heatmap_one = sub_plt_one.imshow(syn_neuron_one, cmap='hot', vmin=self.synapses_w_min," \
                       "vmax=self.synapses_w_max)\n"
                l_1 += "sub_plt_one.title.set_text(variable + ' one')\n"
                if index == 0:
                    l_1 += "sub_plt_one.set_xticks(np.arange(0, self.number_neurons_x, self.number_neurons_x / 5))\n"
                    l_1 += "sub_plt_one.set_yticks(np.arange(0, self.number_neurons_y, self.number_neurons_y / 5))\n"

                l_1 += "syn_neuron_two = a[1::2]\n"
                if index == 0:
                    l_1 += "syn_neuron_two = np.reshape(syn_neuron_two, (self.number_neurons_x, " \
                           "self.number_neurons_y))\n"
                else:
                    l_1 += "syn_neuron_two = np.reshape(syn_neuron_two, (2, 1))\n"
                l_1 += "sub_plt_two = plt.subplot(4, self.columns, 1 + ((2 + index) * self.columns) + 2 * i + 1)\n"
                l_1 += "heatmap_two = sub_plt_two.imshow(syn_neuron_two, cmap='hot', vmin=self.synapses_w_min," \
                       "vmax=self.synapses_w_max)\n"
                l_1 += "sub_plt_two.title.set_text(plot_title)\n"
                l_1 += "sub_plt_two.title.set_text(variable + ' two')\n"
                if index == 0:
                    l_1 += "sub_plt_two.set_xticks(np.arange(0, self.number_neurons_x, self.number_neurons_x / 5))\n"
                    l_1 += "sub_plt_two.set_yticks(np.arange(0, self.number_neurons_y, self.number_neurons_y / 5))\n"

                l_1 += "plt.colorbar(heatmap_one, ax=sub_plt_one)\n"
                l_1 += "plt.colorbar(heatmap_two, ax=sub_plt_two)\n"
                l_1 += "plot_buffer.append(heatmap_one)\n"
                l_1 += "plot_buffer.append(heatmap_two)\n"

                exec(l_1)
                self.synapses_monitors_subplots += plot_buffer

    def close_plot(self):
        plt.close(self.figure)
