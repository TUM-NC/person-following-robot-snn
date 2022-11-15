from Simulation import Simulation, i_off_simulation
import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')

    s = Simulation(config, True)

    # total_simulation_steps = 300
    #
    # for step in range(total_simulation_steps):
    #     s.animate_next_step(step)
    #     if step % 100 == 0:
    #         print("animation was reset")
    #         s.reset()
    #     indices_sim, times_sim = s.get_output_spikes(step)

    i_off_simulation()
