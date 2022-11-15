from snn import SNN
import configparser


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')

    snn = SNN(config)
    snn.run_simulation()
