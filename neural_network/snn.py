from brian2 import NeuronGroup, SpikeGeneratorGroup, array, Network
from brian2 import ms, mV, Synapses, SpikeMonitor, second, volt
from brian2.core.variables import VariableView
from .visualization import Visualization
import numpy as np
from typing import Tuple


class SNN:

    def __init__(self, config):
        """
        :param config: read config file
        """
        self._config = config

        self._simulation_step_time = int(self._config['simulation']['simulation_step_time'])

        # CANN
        self._number_neurons_x = int(self._config['CANN']['number_neurons_x'])
        self._number_neurons_y = int(self._config['CANN']['number_neurons_y'])

        self._epsilon_dopa_left_cann = 0.0
        self._epsilon_dopa_right_cann = 0.0

        self._epsilon_dopa_closer_dis = 0.0
        self._epsilon_dopa_further_dis = 0.0

        self._net = Network()

        # Cann network
        n = self._number_neurons_x * self._number_neurons_y
        # noinspection PyTypeChecker
        self._spikes_input_cann = SpikeGeneratorGroup(n, indices=array([0]), times=array([999999999]) * second)
        # noinspection PyTypeChecker
        self._spikes_reward_left_cann = SpikeGeneratorGroup(n, indices=array([0]), times=array([999999999]) * second)
        # noinspection PyTypeChecker
        self._spikes_reward_right_cann = SpikeGeneratorGroup(n, indices=array([0]), times=array([999999999]) * second)

        self._neurons_output = self._create_neurons(distance=False)

        self._syn_input_cann = self._create_input_synapses(neurons_in=self._spikes_input_cann,
                                                           neurons_out=self._neurons_output,
                                                           distance=False,
                                                           method='euler')

        self._syn_reward_left_cann = self._create_reward_synapses(neurons_in=self._spikes_reward_left_cann,
                                                                  neurons_out=self._syn_input_cann,
                                                                  left=True,
                                                                  distance=False,
                                                                  method='exact')

        self._syn_reward_right_cann = self._create_reward_synapses(neurons_in=self._spikes_reward_right_cann,
                                                                   neurons_out=self._syn_input_cann,
                                                                   left=False,
                                                                   distance=False,
                                                                   method='exact')

        # distance measure network
        # noinspection PyTypeChecker
        self._middle = NeuronGroup(2 * n, model='dv/dt = -v/(20 * ms) : volt', threshold='v>20*mV', reset='v=0*volt',
                                   method='exact')

        # person further 0 spikes; person closer 1 spikes
        self._output_distance = NeuronGroup(2, model='dv/dt = -v/(20 * ms) : volt', threshold='v>5*mV',
                                            reset='v=0*volt', method='exact')

        self._syn_excitatory = Synapses(self._spikes_input_cann, self._middle, model='w : volt',
                                        on_pre='v_post += w', method='linear')
        self._syn_excitatory.connect("2 * i == j")
        self._syn_excitatory.w = 21 * mV

        self._syn_inhibitory = Synapses(self._middle, self._middle, model='w : volt', on_pre='v_post += w',
                                        method='linear')
        self._syn_inhibitory.connect("(i % 2) == 0 and (j % 2) == 0 and i != j")
        self._syn_inhibitory.w = -100 * mV

        self._synapses_recurrent = Synapses(self._middle, self._middle, model='w : volt', on_pre='v_post += w',
                                            delay=99.9 * ms, method='linear')
        self._synapses_recurrent.connect("(i + 1) == j and (i % 2) == 0")
        self._synapses_recurrent.w = 21 * mV

        self._syn_excitatory_output = Synapses(self._middle, self._output_distance, model='w : volt',
                                               on_pre='v_post += w', method='linear')
        self._syn_excitatory_output.connect()
        a = np.linspace(10, 400, num=self._number_neurons_x).repeat(self._number_neurons_y)
        self._syn_excitatory_output.w = np.array([(x, -x, -x, x) for x in a]).flatten() * mV

        # noinspection PyTypeChecker
        self._spikes_reward_closer_dis = SpikeGeneratorGroup(2, indices=array([0]), times=array([999999]) * second)
        # noinspection PyTypeChecker
        self._spikes_reward_further_dis = SpikeGeneratorGroup(2, indices=array([0]), times=array([99999]) * second)

        # noinspection PyTypeChecker
        self._syn_output_dis = self._create_input_synapses(neurons_in=self._output_distance,
                                                           neurons_out=self._neurons_output,
                                                           distance=True,
                                                           method='euler')

        self._syn_reward_closer_dis = self._create_reward_synapses(neurons_in=self._spikes_reward_closer_dis,
                                                                   neurons_out=self._syn_output_dis,
                                                                   left=True,
                                                                   distance=True,
                                                                   method='exact')

        self._syn_reward_further_dis = self._create_reward_synapses(neurons_in=self._spikes_reward_further_dis,
                                                                    neurons_out=self._syn_output_dis,
                                                                    left=False,
                                                                    distance=True,
                                                                    method='exact')

        self._visualization = Visualization(self._config)

        # noinspection PyTypeChecker
        self._neuron_output_mon = SpikeMonitor(self._neurons_output)

        # noinspection PyTypeChecker
        self._net.add([self._spikes_input_cann,
                       self._spikes_reward_left_cann,
                       self._spikes_reward_right_cann,
                       self._neurons_output,
                       self._syn_input_cann,
                       self._syn_reward_left_cann,
                       self._syn_reward_right_cann,
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

    def run_simulation(self) -> None:
        """
        Function runs the simulation for simulation_step_time.
        Input spikes or reward/dopamine spikes need to be set beforehand
        """
        # Neurons
        # noinspection PyUnusedLocal
        tau = float(self._config['neuron.groups']['tau']) * ms
        # noinspection PyUnusedLocal
        v_threshold = float(self._config['neuron.groups']['v_threshold']) * mV
        # noinspection PyUnusedLocal
        v_reset = float(self._config['neuron.groups']['v_reset']) * mV

        # R-STDP
        tau_pre = float(self._config['STDP']['tau_pre']) * ms
        tau_post = float(self._config['STDP']['tau_post']) * ms
        # noinspection PyUnusedLocal
        w_min = float(self._config['STDP']['w_min']) * mV
        w_max = float(self._config['STDP']['w_max']) * mV
        dApre = float(self._config['STDP']['dApre'])
        dApost = -dApre * tau_pre / tau_post * 1.05
        dApost *= w_max
        dApre *= w_max

        # Dopamine signaling
        # noinspection PyUnusedLocal
        tau_c = float(self._config['STDP']['tau_c']) * ms
        # noinspection PyUnusedLocal
        tau_r = float(self._config['STDP']['tau_r']) * ms
        # noinspection PyUnusedLocal
        tau_s = float(self._config['STDP']['tau_s']) * ms
        # noinspection PyUnusedLocal
        epsilon_dopa_left_cann = self._epsilon_dopa_left_cann
        # noinspection PyUnusedLocal
        epsilon_dopa_right_cann = self._epsilon_dopa_right_cann
        # noinspection PyUnusedLocal
        epsilon_dopa_closer_dis = self._epsilon_dopa_closer_dis
        # noinspection PyUnusedLocal
        epsilon_dopa_further_dis = self._epsilon_dopa_further_dis

        self._net.run(self._simulation_step_time * ms)

    def _create_neurons(self, distance: bool) -> NeuronGroup:
        """
        function creates the output neurons, to predict the wheel velocities.
        """
        num = int(self._config['neuron.groups']['number'])
        equation = 'dv/dt = -v / tau : volt'
        if distance:
            threshold = 'v > 0 * mV'
            reset = 'v = 0 * mV'
        else:
            threshold = 'v > v_threshold'
            reset = 'v = v_reset'
        method = 'exact'
        refactory = "0.5 * ms"
        return NeuronGroup(num, equation, threshold=threshold, reset=reset, method=method, refractory=refactory)

    def _create_input_synapses(self,
                               neurons_in: SpikeGeneratorGroup,
                               neurons_out: NeuronGroup,
                               distance: bool,
                               method: str) -> Synapses:
        """
        Function creates the input synapses that use R-STDP.
        :param neurons_in: the pre-synaptic neurons
        :param neurons_out: the pros-synaptic neurons
        :param method: the integration method
        """
        if distance:
            model = '''
                    dc_d/dt = -c_d / (105 * ms) : volt (clock-driven)
                    dr_d/dt = -r_d / (105 * ms) : 1 (clock-driven)
                    dw_d/dt = c_d * r_d / ms: volt (clock-driven)
                    dApre/dt = -Apre / tau_pre : volt (event-driven)
                    dApost/dt = -Apost / tau_post : volt (event-driven)
                    '''
            on_pre = '''
                     v_post += w_d
                     Apre += dApre
                     c_d = clip(c_d + Apost, 0 * volt, w_max - 100 * mV)
                     w_d = clip(w_d, w_min - 100 * mV, w_max + 100 * mV)
                     '''
            on_post = '''
                      Apost += dApost
                      c_d = clip(c_d + Apre, 0 * volt, w_max - 100 * mV)
                      w_d = clip(w_d, w_min - 100 * mV, w_max + 100 * mV)
                      '''
        else:
            model = '''
                    dc/dt = -c / tau_c : volt (clock-driven)
                    dr/dt = -r / tau_r : 1 (clock-driven)
                    dw/dt = c * r / ms: volt (clock-driven)
                    dApre/dt = -Apre / tau_pre : volt (event-driven)
                    dApost/dt = -Apost / tau_post : volt (event-driven)
                    '''
            on_pre = '''
                     v_post += w
                     Apre += dApre
                     c = clip(c + Apost, 0 * volt, w_max - 100 * mV)
                     w = clip(w, w_min, w_max)
                     '''
            on_post = '''
                      Apost += dApost
                      c = clip(c + Apre, 0 * volt, w_max - 100 * mV)
                      w = clip(w, w_min, w_max)
                      '''
        s = Synapses(neurons_in, neurons_out, model=model, on_pre=on_pre, on_post=on_post, method=method)
        s.connect()
        if distance:
            s.w_d = 12 * mV
            s.c_d = 0 * mV
            s.r_d = 0
        else:
            s.w = float(self._config['synapse']['w_start']) * mV
            s.c = 0 * mV
            s.r = 0
        return s

    def _create_reward_synapses(self,
                                neurons_in: SpikeGeneratorGroup,
                                neurons_out: Synapses,
                                left: bool,
                                distance: bool,
                                method: str) -> Synapses:
        """
        Function creates the reward synapses.
        :param neurons_in: the pre-synaptic neurons
        :param neurons_out: the pros-synaptic neurons
        :param left: True if the synapses for the left neuron are created. Differentiation is necessary,
                     as they can have different loss values, so on pre has different epsilon_dopa values.
        :param method: the integration method
        """
        model = 's : volt'
        if left:
            if distance:
                on_pre = 'r_d_post += epsilon_dopa_closer_dis'
                condition = "i == j"
            else:
                on_pre = 'r_post += epsilon_dopa_left_cann'
                condition = "2 * i == j"
        else:
            if distance:
                on_pre = 'r_d_post += epsilon_dopa_further_dis'
                condition = "i + 2 == j"
            else:
                on_pre = 'r_post += epsilon_dopa_right_cann'
                condition = "2 * i + 1 == j"
        on_post = ''
        r = Synapses(neurons_in, neurons_out, model=model, on_pre=on_pre, on_post=on_post, method=method)
        r.connect(condition)
        r.s = float(self._config['reward.synapse']['weight']) * mV
        return r

    def plot_data(self,
                  simulation_step: int,
                  input_spikes: Tuple[np.ndarray, np.ndarray],
                  plotting_interval: int = 1) -> None:
        """
        Function prepares the data that should be plotted by the plotting class.
        The input spikes are needed as an input, because it would make sense to monitor them,
        as the times and indices are known.
        :param simulation_step: how often the simulation was run
        :param input_spikes: the indices and times of the input spikes as tuples
        :param plotting_interval: how often the data should be plotted
                                  --> drastically improves performance if less often
        """

        if simulation_step % plotting_interval != 0:
            return

        output_spikes_indices, output_spikes_times = self._neuron_output_mon.it
        if not output_spikes_indices:
            output_spikes_indices, output_spikes_times = np.array([]), np.array([])
        neuron_monitors = [((output_spikes_indices, output_spikes_times), "Output Neurons Spikes")]

        synapses_monitors = [(self._syn_input_cann, "Input Synapses", ['w', 'c']),
                             (self._syn_output_dis, "Distance Synapses", ['w_d', 'c_d'])]

        spike_input_monitor = [(input_spikes, "Input Data Spikes")]

        self._visualization.plot_data(neuron_monitors=neuron_monitors,
                                      syn_monitors=synapses_monitors,
                                      in_monitors=spike_input_monitor,
                                      sim=simulation_step)

    def set_input_spikes(self, spike_ind: array, spike_t: array) -> None:
        """
        Set the input spikes for the next simulation step. Times must be adapted to the total simulation time.
        :param spike_ind: array with indices of the neurons that send out spikes
        :param spike_t: the times for the input spikes. Position 0 corresponds to the time of the
                        index of the spike_ind array at position 0
        """
        self._spikes_input_cann.set_spikes(indices=spike_ind, times=spike_t * ms)

    def set_reward_spikes_cann(self, left: float, right: float, simulation_step: int) -> None:
        """
        Takes as input the left and right loss and calculates the spike times, spikes indices and sets the
        epsilon dopamine values.
        :param left: the left loss
        :param right: the right loss
        :param simulation_step: how often the simulation was run to calculate the correct spike time
        """
        n = self._number_neurons_x * self._number_neurons_y
        spike_t = np.full(n, 1) * ms + self._simulation_step_time * (simulation_step + 1) * ms
        spike_ind = np.arange(n)

        # noinspection PyTypeChecker
        self._spikes_reward_left_cann.set_spikes(indices=spike_ind, times=spike_t)
        # noinspection PyTypeChecker
        self._spikes_reward_right_cann.set_spikes(indices=spike_ind, times=spike_t)

        self._epsilon_dopa_left_cann = left
        self._epsilon_dopa_right_cann = right

    def set_reward_spikes_distance(self, closer: float, further: float, simulation_step: int) -> None:
        """
        Takes as input the left and right loss and calculates the spike times, spikes indices and sets the
        epsilon dopamine values.
        :param closer: the left loss
        :param further: the right loss
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
        Returns the output from the left and right motor neurons as a tuple. The first value corresponds
        to the indices (0, 1) and the second value to the times.
        """
        return self._neuron_output_mon.it

    def set_weights(self, path_weights_cann: str, path_weights_dist: str):
        """
        Function sets weights for the input synapses of the cann and distance network output.
        Weights are loaded from a txt file , that is expected to be in the following format
        file:
            0.001
            0.001
            0.003
            ... and so on
        :param path_weights_cann: path to the weights for the input synapses of the cann
        :param path_weights_dist: path to the weights for distance network output
        """
        self._syn_input_cann.w = np.loadtxt(path_weights_cann) * volt
        self._syn_output_dis.w_d = np.loadtxt(path_weights_dist) * volt

    def get_predicted_velocity(self, step):
        """
        Returns the speed difference for left and right wheel, which can be added to the current speeds.
        0s => -0.5
        50s => 0
        100s => 0.5
        :param step: how often the simulation was run
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
                v_left_diff = -round((times[index + i] / ms - (50 + step * self._simulation_step_time)) / 100, 4)
            if spike_neuron_index == 1 and v_right_diff == -1:
                v_right_diff = -round((times[index + i] / ms - (50 + step * self._simulation_step_time)) / 100, 4)

        if v_left_diff == -1:
            v_left_diff = -0.5

        if v_right_diff == -1:
            v_right_diff = -0.5

        return v_left_diff, v_right_diff

    @property
    def distance_weights(self):
        """
        Returns the weights of the synapses between distance output neurons and the speed output neurons
        """
        return self._syn_output_dis.w_d

    @property
    def cann_weights(self):
        """
        Returns the weights of the synapses between CANN input and the speed output neurons
        """
        return self._syn_input_cann.w

    def reset(self):
        """
        Function resets all values of the snn
        """
        self._syn_input_cann.c = -self._syn_input_cann.c

        self._syn_reward_left_cann.s = 0 * mV
        self._syn_reward_left_cann.r = 0
        self._syn_reward_left_cann.r_post = 0

        self._syn_reward_right_cann.s = 0 * mV
        self._syn_reward_right_cann.r = 0
        self._syn_reward_right_cann.r_post = 0

        self._epsilon_dopa_left_cann = 0
        self._epsilon_dopa_right_cann = 0

        self._epsilon_dopa_closer_dis = 0
        self._epsilon_dopa_further_dis = 0

    def close_plot(self):
        self._visualization.close_plot()
