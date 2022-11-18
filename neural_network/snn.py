from brian2 import NeuronGroup, SpikeGeneratorGroup, Network
from brian2 import ms, mV, Synapses, SpikeMonitor, second, volt
from brian2.core.variables import VariableView
from .visualization import Visualization
import numpy as np
from typing import Tuple, List


class SNN:
    """
    Class constructs and simulates the SNN for the person following robot.
    Parameters for the network and the time for one simulation step can be set
    in the config.ini file.
    The network is constructed on init and the run_simulation method can be called
    to simulate one step. The input and reward spikes should be set beforehand.
    """
    # buffer for plotting, so that they do not needed to be recorded
    _input_spikes = ([], [])

    def __init__(self, config):
        """
        :param config: read config file
        """
        self._config = config

        self._simulation_step_time = int(self._config['simulation']['simulation_step_time'])
        self._number_neurons_x = int(self._config['CANN']['number_neurons_x'])
        self._number_neurons_y = int(self._config['CANN']['number_neurons_y'])
        self._show_snn_visualization = config.getboolean(section='setup', option='show_snn_visualization')

        self._epsilon_dopa_left_cann = 0.0
        self._epsilon_dopa_right_cann = 0.0

        self._epsilon_dopa_closer_dis = 0.0
        self._epsilon_dopa_further_dis = 0.0

        self._net = Network()

        """ Cann network generator group """
        n = self._number_neurons_x * self._number_neurons_y

        # noinspection PyTypeChecker
        self._spikes_input_cann = SpikeGeneratorGroup(n, indices=[0], times=[999999999] * second)
        
        """ Velocity Neurons """
        tau_velocity = float(self._config['velocity.network']['tau'])
        v_threshold_velocity = float(self._config['velocity.network']['v_threshold'])
        v_reset_velocity = float(self._config['velocity.network']['v_reset'])
        self._neurons_velocity = NeuronGroup(2,
                                             f'dv/dt = -v / ({tau_velocity} * ms) : volt',
                                             threshold=f'v > {v_threshold_velocity} * mV',
                                             reset=f'v = {v_reset_velocity} * mV',
                                             method='exact')

        """ Distance Neurons """
        # middle layer between the cann and distance output neurons
        # noinspection PyTypeChecker
        self._middle = NeuronGroup(2 * n,
                                   model='dv/dt = -v/(20 * ms) : volt',
                                   threshold='v>20*mV',
                                   reset='v=0*volt',
                                   method='exact')

        # output neurons for the distance - person further 0 spikes; person closer 1 spikes
        tau_distance = float(self._config['distance.network']['tau'])
        v_threshold_distance = float(self._config['distance.network']['v_threshold'])
        v_reset_distance = float(self._config['distance.network']['v_reset'])
        self._output_distance = NeuronGroup(2,
                                            model=f'dv/dt = -v / ({tau_distance} * ms) : volt',
                                            threshold=f'v > {v_threshold_distance} * mV',
                                            reset=f'v = {v_reset_distance} * mV',
                                            method='exact')

        """ Velocity Network """
        self._syn_cann_velocity = self._cann_velocity_synapses(neurons_in=self._spikes_input_cann,
                                                               neurons_out=self._neurons_velocity)

        "Distance Network"
        # excitatory synapses between the cann layer and the middle layer
        self._syn_excitatory = Synapses(self._spikes_input_cann, self._middle, model='w : volt',
                                        on_pre='v_post += w', method='linear')
        self._syn_excitatory.connect("2 * i == j")
        self._syn_excitatory.w = 21 * mV

        # inhibitory synapses within the middle layer to prevent the network from spiking after the first spike
        self._syn_inhibitory = Synapses(self._middle, self._middle, model='w : volt', on_pre='v_post += w',
                                        method='linear')
        self._syn_inhibitory.connect("(i % 2) == 0 and (j % 2) == 0 and i != j")
        self._syn_inhibitory.w = -100 * mV  # big negative weight to ensure nothing else spikes

        # excitatory synapses within to the mirrored neuron in the middle layer, which spikes in the next simulation
        # step to present the previous position of the person
        self._synapses_recurrent = Synapses(self._middle, self._middle, model='w : volt', on_pre='v_post += w',
                                            delay=99.9 * ms, method='linear')
        self._synapses_recurrent.connect("(i + 1) == j and (i % 2) == 0")
        self._synapses_recurrent.w = 21 * mV

        # excitatory synapses connecting the middle layer which contains the distance information to the
        # output layer which spikes depending on if the person came closer or not
        self._syn_excitatory_output = Synapses(self._middle, self._output_distance, model='w : volt',
                                               on_pre='v_post += w', method='linear')
        self._syn_excitatory_output.connect()
        a = np.linspace(10, 400, num=self._number_neurons_x).repeat(self._number_neurons_y)
        self._syn_excitatory_output.w = np.array([(x, -x, -x, x) for x in a]).flatten() * mV

        # synapses connecting the output if the person came closer or not to velocity neurons.
        # these will also be trained with R-STDP
        self._syn_output_dis = self._distance_velocity_synapses(neurons_in=self._output_distance,
                                                                neurons_out=self._neurons_velocity)

        """ Reward Velocity Network"""
        # noinspection PyTypeChecker
        self._spikes_reward_left_cann_velocity = SpikeGeneratorGroup(n, indices=[0], times=[999999999] * second)
        # noinspection PyTypeChecker
        self._spikes_reward_right_cann_velocity = SpikeGeneratorGroup(n, indices=[0], times=[999999999] * second)

        self._syn_reward_left_velocity = self._reward_syn_cann_velocity(
            neurons_in=self._spikes_reward_left_cann_velocity,
            neurons_out=self._syn_cann_velocity,
            left=True)

        self._syn_reward_right_velocity = self._reward_syn_cann_velocity(
            neurons_in=self._spikes_reward_right_cann_velocity,
            neurons_out=self._syn_cann_velocity,
            left=False)

        """ Reward Distance Network"""
        # noinspection PyTypeChecker
        self._spikes_reward_closer_dis = SpikeGeneratorGroup(2, indices=[0], times=[99999999999] * second)
        # noinspection PyTypeChecker
        self._spikes_reward_further_dis = SpikeGeneratorGroup(2, indices=[0], times=[99999999999] * second)

        self._syn_reward_closer_dis = self._reward_syn_distance_velocity(neurons_in=self._spikes_reward_closer_dis,
                                                                         neurons_out=self._syn_output_dis,
                                                                         left=True)

        self._syn_reward_further_dis = self._reward_syn_distance_velocity(neurons_in=self._spikes_reward_further_dis,
                                                                          neurons_out=self._syn_output_dis,
                                                                          left=False)
        if self._show_snn_visualization:
            self._visualization = Visualization(self._config)

        # noinspection PyTypeChecker
        self._neuron_output_mon = SpikeMonitor(self._neurons_velocity)

        # noinspection PyTypeChecker
        self._net.add([self._spikes_input_cann,
                       self._spikes_reward_left_cann_velocity,
                       self._spikes_reward_right_cann_velocity,
                       self._neurons_velocity,
                       self._syn_cann_velocity,
                       self._syn_reward_left_velocity,
                       self._syn_reward_right_velocity,
                       self._spikes_reward_closer_dis,
                       self._spikes_reward_further_dis,
                       self._syn_output_dis,
                       self._syn_reward_closer_dis,
                       self._syn_reward_further_dis,
                       self._neuron_output_mon,
                       self._middle,
                       self._output_distance,
                       self._syn_excitatory,
                       self._syn_inhibitory,
                       self._syn_excitatory_output,
                       self._synapses_recurrent,
                       self._neuron_output_mon
                       ])

    def run_simulation(self, step: int) -> None:
        """
        Function runs the simulation for simulation_step_time.
        Input spikes or reward/dopamine spikes need to be set beforehand.
        """
        # noinspection PyUnusedLocal
        epsilon_dopa_left_cann = self._epsilon_dopa_left_cann
        # noinspection PyUnusedLocal
        epsilon_dopa_right_cann = self._epsilon_dopa_right_cann

        # noinspection PyUnusedLocal
        epsilon_dopa_closer_dis = self._epsilon_dopa_closer_dis
        # noinspection PyUnusedLocal
        epsilon_dopa_further_dis = self._epsilon_dopa_further_dis

        self._net.run(self._simulation_step_time * ms)
        self._plot_data(step)

    def _cann_velocity_synapses(self, neurons_in: SpikeGeneratorGroup, neurons_out: NeuronGroup) -> Synapses:
        """
        Function creates and returns the synapses connecting the CANN to the velocity neurons
        and sets the parameters given in the config.ini file.
        :param neurons_in: the pre-synaptic neurons
        :param neurons_out: the post-synaptic neurons
                :return: synapses connecting the neurons
        """
        # STDP velocity
        tau_pre = float(self._config['velocity.RSTDP']['tau_pre'])
        tau_post = float(self._config['velocity.RSTDP']['tau_post'])
        w_min = float(self._config['velocity.RSTDP']['w_min'])
        w_max = float(self._config['velocity.RSTDP']['w_max'])
        dApre = float(self._config['velocity.RSTDP']['dApre'])
        dApost = -dApre * tau_pre / tau_post * 1.05
        dApost *= w_max
        dApre *= w_max

        # Dopamine signaling velocity
        tau_c = float(self._config['velocity.RSTDP']['tau_c'])
        tau_r = float(self._config['velocity.RSTDP']['tau_r'])
        c_min = float(self._config['velocity.RSTDP']['c_min'])
        c_max = float(self._config['velocity.RSTDP']['c_max'])

        model = f'''
                 dc/dt = -c / ({tau_c} * ms) : volt (clock-driven)
                 dr/dt = -r / ({tau_r} * ms) : 1 (clock-driven)
                 dw/dt = c * r / ms: volt (clock-driven)
                 dApre/dt = -Apre / ({tau_pre} * ms) : volt (event-driven)
                 dApost/dt = -Apost / ({tau_post} * ms) : volt (event-driven)
                 '''
        on_pre = f'''
                  v_post += w
                  Apre += {dApre} * mV
                  c = clip(c + Apost, {c_min} * mV, {c_max} * mV)
                  w = clip(w, {w_min} * mV, {w_max} * mV)
                  '''
        on_post = f'''
                   Apost += {dApost} * mV
                   c = clip(c + Apre, {c_min} * mV, {c_max} * mV)
                   w = clip(w, {w_min} * mV, {w_max} * mV)
                  '''
        s = Synapses(neurons_in, neurons_out, model=model, on_pre=on_pre, on_post=on_post, method='euler')
        s.connect()

        s.w = float(self._config['velocity.network']['w_start']) * mV
        s.c = 0 * mV
        s.r = 0
        return s

    def _distance_velocity_synapses(self, neurons_in: NeuronGroup, neurons_out: NeuronGroup) -> Synapses:
        """
        Function creates and returns the synapses connecting the distance to the velocity neurons and sets the
        parameters given in the config.ini file.
        :param neurons_in: the pre-synaptic neurons
        :param neurons_out: the post-synaptic neurons
        :return: synapses connecting the neurons
        """
        tau_pre_d = float(self._config['distance.RSTDP']['tau_pre'])
        tau_post_d = float(self._config['distance.RSTDP']['tau_post'])
        # noinspection PyUnusedLocal
        w_min_d = float(self._config['distance.RSTDP']['w_min'])
        w_max_d = float(self._config['distance.RSTDP']['w_max'])
        dApre_d = float(self._config['distance.RSTDP']['dApre'])
        dApost_d = -dApre_d * tau_pre_d / tau_post_d * 1.05
        dApost_d *= w_max_d
        dApre_d *= w_max_d

        # Dopamine signaling distance
        tau_c_d = float(self._config['distance.RSTDP']['tau_c'])
        tau_r_d = float(self._config['distance.RSTDP']['tau_r'])
        c_min_d = float(self._config['distance.RSTDP']['c_min'])
        c_max_d = float(self._config['distance.RSTDP']['c_max'])

        model = f'''
                 dc_d/dt = -c_d / ({tau_c_d} * ms) : volt (clock-driven)
                 dr_d/dt = -r_d / ({tau_r_d} * ms) : 1 (clock-driven)
                 dw_d/dt = c_d * r_d / ms: volt (clock-driven)
                 dApre_d/dt = -Apre_d / ({tau_pre_d} * ms) : volt (event-driven)
                 dApost_d/dt = -Apost_d / ({tau_post_d} * ms) : volt (event-driven)
                 '''
        on_pre = f'''
                  v_post += w_d
                  Apre_d += {dApre_d} * mV
                  c_d = clip(c_d + Apost_d, {c_min_d} * mV, {c_max_d} * mV)
                  w_d = clip(w_d, {w_min_d} * mV, {w_max_d} * mV)
                  '''
        on_post = f'''
                   Apost_d += {dApost_d} * mV
                   c_d = clip(c_d + Apre_d, {c_min_d} * mV, {c_max_d} * mV)
                   w_d = clip(w_d, {w_min_d} * mV, {w_max_d} * mV)
                   '''

        s = Synapses(neurons_in, neurons_out, model=model, on_pre=on_pre, on_post=on_post, method="euler")
        s.connect()
        s.w_d = float(self._config['distance.network']['w_start']) * mV
        s.c_d = 0 * mV
        s.r_d = 0

        return s

    @staticmethod
    def _reward_syn_cann_velocity(neurons_in: SpikeGeneratorGroup, neurons_out: Synapses, left: bool) -> Synapses:
        """
        Function creates the reward synapses for the synapses connecting the CANN neurons to the velocity neurons.
        :param neurons_in: the pre-synaptic neurons
        :param neurons_out: the pros-synaptic neurons
        :param left: True if the synapses for the left neuron are created. This is necessary
                     as they can have different reward values, so on pre has different epsilon_dopa values.
        """
        if left:
            on_pre = 'r_post += epsilon_dopa_left_cann'
            condition = "2 * i == j"
        else:
            on_pre = 'r_post += epsilon_dopa_right_cann'
            condition = "2 * i + 1 == j"
        r = Synapses(neurons_in, neurons_out, model='s : volt', on_pre=on_pre, on_post='', method='exact')
        r.connect(condition)
        r.s = 0 * mV
        return r

    @staticmethod
    def _reward_syn_distance_velocity(neurons_in: SpikeGeneratorGroup, neurons_out: Synapses, left: bool) -> Synapses:
        """
        Function creates the reward synapses for the synapses connecting the distance neurons to the velocity neurons.
        :param neurons_in: the pre-synaptic neurons
        :param neurons_out: the pros-synaptic neurons
        :param left: True if the synapses for the left neuron are created. This is necessary
                     as they can have different reward values, so on pre has different epsilon_dopa values.
        """
        if left:
            on_pre = 'r_d_post += epsilon_dopa_closer_dis'
            condition = "i == j"
        else:
            on_pre = 'r_d_post += epsilon_dopa_further_dis'
            condition = "i + 2 == j"

        r = Synapses(neurons_in, neurons_out, model='s : volt', on_pre=on_pre, on_post='', method='exact')
        r.connect(condition)
        r.s = 0 * mV
        return r

    def _plot_data(self, simulation_step: int) -> None:
        """
        Function prepares the data that should be plotted by the plotting class.
        The input spikes are needed as an input, because it would make sense to monitor them,
        as the times and indices are known.
        :param simulation_step: how often the simulation was run
        """
        if not self._show_snn_visualization:
            return

        self._visualization.update(step=simulation_step,
                                   input_spikes=self._input_spikes,
                                   output_spikes=self._neuron_output_mon.it,
                                   weights_left_wheel=self._syn_cann_velocity.w[0::2] / mV,
                                   weights_right_wheel=self._syn_cann_velocity.w[1::2] / mV,
                                   eli_trace_left_wheel=self._syn_cann_velocity.c[0::2] / mV,
                                   eli_trace_right_wheel=self._syn_cann_velocity.c[1::2] / mV,
                                   weights_distance_further=self._syn_output_dis.w_d[0::2] / mV,
                                   weights_distance_closer=self._syn_output_dis.w_d[1::2] / mV,
                                   eli_trace_distance_further=self._syn_output_dis.c_d[0::2] / mV,
                                   eli_trace_distance_closer=self._syn_output_dis.c_d[1::2] / mV)

    def set_input_spikes(self, spike_ind: List, spike_t: List) -> None:
        """
        Set the input spikes for the next simulation step. Times must be adapted to the total simulation time.
        :param spike_ind: array with indices of the neurons that send out spikes
        :param spike_t: the times for the input spikes. Position 0 corresponds to the time of the
                        index of the spike_ind array at position 0
        """
        self._input_spikes = (spike_ind, spike_t)
        # noinspection PyTypeChecker
        self._spikes_input_cann.set_spikes(indices=spike_ind, times=spike_t * ms)

    def set_reward_spikes_cann(self, left: float, right: float, simulation_step: int) -> None:
        """
        Takes as input the left and right reward and calculates the spike times, spikes indices and sets the
        epsilon dopamine values.
        :param left: the left reward
        :param right: the right reward
        :param simulation_step: how often the simulation was run to calculate the correct spike time
        """
        n = self._number_neurons_x * self._number_neurons_y
        spike_t = np.full(n, 1) * ms + self._simulation_step_time * (simulation_step + 1) * ms
        spike_ind = np.arange(n)

        # noinspection PyTypeChecker
        self._spikes_reward_left_cann_velocity.set_spikes(indices=spike_ind, times=spike_t)
        # noinspection PyTypeChecker
        self._spikes_reward_right_cann_velocity.set_spikes(indices=spike_ind, times=spike_t)

        self._epsilon_dopa_left_cann = left
        self._epsilon_dopa_right_cann = right

    def set_reward_spikes_distance(self, closer: float, further: float, simulation_step: int) -> None:
        """
        Takes as input the left and right reward and calculates the spike times, spikes indices and sets the
        epsilon dopamine values.
        :param closer: the left reward
        :param further: the right reward
        :param simulation_step: how often the simulation was run
        """
        spike_t = np.full(2, 1) * ms + self._simulation_step_time * (simulation_step + 1) * ms
        spike_ind = np.arange(2)

        # noinspection PyTypeChecker
        self._spikes_reward_further_dis.set_spikes(indices=spike_ind, times=spike_t)
        # noinspection PyTypeChecker
        self._spikes_reward_closer_dis.set_spikes(indices=spike_ind, times=spike_t)

        self._epsilon_dopa_further_dis = further
        self._epsilon_dopa_closer_dis = closer

    def get_spike_output(self) -> (VariableView, VariableView):
        """
        Returns the output from the left and right wheel neurons as a tuple. The first value corresponds
        to the indices (0, 1) and the second value to the times.
        :return: output of the wheel velocity neurons
        """
        return self._neuron_output_mon.it

    def set_weights(self, path_weights_cann: str, path_weights_dist: str) -> None:
        """
        Function sets weights for the synapses connecting the cann neurons to the velocity neurons
        and the distance neurons to the velocity neurons.
        Weights are loaded from a txt file, that is expected to be in the following format
        file:
            0.001
            0.001
            0.003
            ... and so on
        :param path_weights_cann: path to the weights for the cann to the velocity synapses
        :param path_weights_dist: path to the weights for the distance to the velocity synapses
        """
        self._syn_cann_velocity.w = np.loadtxt(path_weights_cann) * volt
        self._syn_output_dis.w_d = np.loadtxt(path_weights_dist) * volt

    def get_predicted_velocity(self, step) -> Tuple[float, float]:
        """
        Returns the speed difference for left and right wheel, which can be added to the current speeds.
        0s => -0.5 m/s
        50s => 0
        100s => 0.5 m/s
        :param step: how often the simulation was run
        :return: Tuple (left wheel velocity difference, right wheel velocity difference) in m/s
        """
        indices, times = self.get_spike_output()
        if not indices or not times:
            return -0.5, -0.5

        index = np.argmax(times >= (step * self._simulation_step_time) * ms)
        if index == 0 and times[-1] < (step + 1) * self._simulation_step_time * ms:
            return -0.5, -0.5

        v_left_diff = -1
        v_right_diff = -1
        for i, spike_neuron_index in enumerate(indices[index:]):
            if spike_neuron_index == 0 and v_left_diff == -1:
                a = (self._simulation_step_time / 2 + step * self._simulation_step_time)
                v_left_diff = -round((times[index + i] / ms - a) / 100, 4)
            if spike_neuron_index == 1 and v_right_diff == -1:
                a = (self._simulation_step_time / 2 + step * self._simulation_step_time)
                v_right_diff = -round((times[index + i] / ms - a) / 100, 4)

        if v_left_diff == -1:
            v_left_diff = -0.5

        if v_right_diff == -1:
            v_right_diff = -0.5

        return v_left_diff, v_right_diff

    @property
    def distance_weights(self):
        """
        Returns the weights of the synapses between distance output neurons and the wheel velocity neurons
        """
        return self._syn_output_dis.w_d

    @property
    def cann_weights(self):
        """
        Returns the weights of the synapses between CANN input and the wheel velocity neurons
        """
        return self._syn_cann_velocity.w

    def reset(self) -> None:
        """
        Function resets all values of the snn
        """
        self._syn_cann_velocity.c = -self._syn_cann_velocity.c

        self._syn_reward_left_velocity.s = 0 * mV
        self._syn_reward_left_velocity.r = 0
        self._syn_reward_left_velocity.r_post = 0

        self._syn_reward_right_velocity.s = 0 * mV
        self._syn_reward_right_velocity.r = 0
        self._syn_reward_right_velocity.r_post = 0

        self._epsilon_dopa_left_cann = 0
        self._epsilon_dopa_right_cann = 0

        self._epsilon_dopa_closer_dis = 0
        self._epsilon_dopa_further_dis = 0

    def close_plot(self) -> None:
        """
        Function closes the snn plot
        """
        self._visualization.close_plot()
