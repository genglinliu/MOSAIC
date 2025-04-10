import json
from simulation import Simulation
from utils import Utils
import logging

if __name__ == "__main__":
    with open('configs/experiment_config.json', 'r') as file:
        config = json.load(file)
        
    Utils.configure_logging(engine=config['engine'])
    logging.info(f"Starting simulation with {config['num_users']} users for {config['num_time_steps']} time steps using {config['engine']}...")

    sim = Simulation(config)
    sim.run(config['num_time_steps'])
    