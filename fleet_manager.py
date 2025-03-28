import numpy as np
import os
from model.rl import dgd_based_algo as dgda
from model.rl.dqn import DQN
from data import data_manager as dm
from data import map_manager as mm
from data import time_manager as tm
from model.rl import exp_simulator as exp_sim
from tqdm import tqdm
import pandas as pd
import math
from scipy.stats import wasserstein_distance as wd
import matplotlib.pyplot as plt
import random

"""
Author: Peibo Duan
Date: 2021.02.05
Fun: 
    get experimental results in the format of excel with the columns defined as follows:
    ['day', 'type', 'num_vehicles', 'completion', 'non-completion', 'completion_ratio', 'total_revenue', 'WD'], where
    type: the type for the experiment, including: 
        'r': random vehicle deployment with the given total number of vehicles
        'opt': optimal vehicle deployment with the given total number of vehicles
        'r_b_opt': random vehicle deployment based on total number of vehicles calculated based on optimal 
                   deployment strategy
        'rt_b_r': vehicle deployment based order distribution and random deployment strategy
        'rt_b_opt': vehicle deployment based on order distribution and optimal deployment strategy 
"""


class FleetManager:

    def __init__(self):
        self.state_size = int(mm.CITY_DIVISION ** 2 + 2)
        self.dqn = DQN(self.state_size)
        self.last_time_window = int(2 * 60 * 60 / tm.TIME_WINDOW_LEN - 1)
        self.test_working = ['20161121', '20161122', '20161123', '20161124', '20161125']
        self.test_weekend = ['20161126', '20161127']

        self.all_grid_ids = np.array([grid_id for grid_id in range(int(mm.CITY_DIVISION ** 2))], dtype=int)
        self.len_time_windows = int(2 * 60 * 60 / tm.TIME_WINDOW_LEN)

        # parameters for DGD based algorithm
        self.max_iterations = 12  # maximal iteration for GD algorithm
        self.p_lb = int(math.floor(dm.ph_min_vehicles / self.state_size))
        self.p_up = int(math.ceil(dm.ph_max_vehicles / self.state_size))
        self.np_lb = int(math.floor(dm.nph_min_vehicles / self.state_size))
        self.np_up = int(math.ceil(dm.nph_max_vehicles / self.state_size))

        # parameters for DQN training
        self.pool_size = 2048
        self.max_episodes = 5000

    def train_dqn_model(self, day_type_opt: str, time_opt: str):
        samples_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'simulated_data', 'samples',
                                      day_type_opt, time_opt)
        # load data
        samples = np.load(os.path.join(samples_r_path, 'samples.npz'))
        states = samples['states']
        next_states = samples['next_states']
        flags = samples['flags']
        rewards = samples['rewards']

        # training DQN model
        loss = list()
        len_states = len(states)
        save_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimental_results', day_type_opt,
                                   time_opt)
        print('training models...')
        for episode in range(self.max_episodes):
            print('episode: ', episode)
            sampling_indices = random.sample(range(0, len_states), self.pool_size)
            sampling_states = np.array([states[i] for i in sampling_indices])
            sampling_next_states = np.array([next_states[i] for i in sampling_indices])
            sampling_flags = np.array([flags[i] for i in sampling_indices])
            sampling_rewards = np.array([rewards[i] for i in sampling_indices])
            sampling_targets = self.dqn.predict(sampling_next_states)
            sampling_targets = sampling_targets.reshape(-1)
            sampling_targets = np.array(sampling_flags) * sampling_targets * self.dqn.gamma + np.array(sampling_rewards)
            loss_in_episode = self.dqn.train_model(sampling_states, sampling_targets, save_r_path, episode)
            loss.append(loss_in_episode)

        loss = np.array(loss)
        loss = loss.flatten()
        loss_path = os.path.join(save_r_path, 'loss.npy')
        np.save(loss_path, loss)
        plt.plot([i + 1 for i in range(len(loss))], loss)
        plt.savefig(os.path.join(save_r_path, 'loss.jpg'))

    def get_optimal_vehicle_deployment(self, day_type_opt: str, time_opt: str):
        # load model
        dqn_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimental_results', day_type_opt,
                                      time_opt, 'dqn_model.h5')
        print('model loading...')
        self.dqn.load(dqn_model_path)
        wd_nph = str(int(tm.wd_snph.split(':')[0])) + '-' + str(int(tm.wd_enph.split(':')[0]))
        wkd_nph = str(int(tm.wkd_snph.split(':')[0])) + '-' + str(int(tm.wkd_enph.split(':')[0]))
        if time_opt == wd_nph or time_opt == wkd_nph:
            optimal_D = dgda.dgd_based_algo(self.max_iterations, self.dqn, self.np_lb, self.np_up)
        else:
            optimal_D = dgda.dgd_based_algo(self.max_iterations, self.dqn, self.p_lb, self.p_up)
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimental_results', day_type_opt,
                                 time_opt, 'optimal_D.npy')
        np.save(save_path, optimal_D)

    def get_order_distribution(self, day: str, day_type_opt: str, time_opt: str) -> np:
        test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'simulated_data',
                                      'testing_data', day_type_opt, time_opt, day + '.csv')
        test_data = dm.load_data(test_data_path, True, False, list())
        total_orders = len(test_data)
        order_distribution = np.zeros(len(self.all_grid_ids))
        for index, row in test_data.iterrows():
            order_distribution[row['start_grid_id']] += 1
        order_distribution /= total_orders
        order_distribution = order_distribution.reshape(mm.CITY_DIVISION, mm.CITY_DIVISION)
        return test_data, order_distribution

    def random_deployment(self, num_vehicles: int) -> np:
        r_d = np.zeros(len(self.all_grid_ids), dtype=int)
        positions_idx = np.random.choice(len(self.all_grid_ids), size=num_vehicles, replace=True)
        for idx in positions_idx:
            grid_id = self.all_grid_ids[idx]
            r_d[grid_id] += 1
        return r_d.reshape(mm.CITY_DIVISION, mm.CITY_DIVISION)

    @staticmethod
    def assign_vehicles(vd: np) -> pd.DataFrame:
        col_names = ['vehicle_id', 'time_window', 'grid_id', 'state']
        initial_all_vehicles = pd.DataFrame(columns=col_names, index=None)
        vehicle_id = 0
        for i in range(mm.CITY_DIVISION):
            for j in range(mm.CITY_DIVISION):
                grid_id = i * mm.CITY_DIVISION + j
                for _ in range(vd[i][j]):
                    df = pd.DataFrame([[vehicle_id, 0, grid_id, 'cruising']], columns=col_names)
                    initial_all_vehicles = initial_all_vehicles.append(df, ignore_index=True)
                    vehicle_id += 1
        return initial_all_vehicles

    def get_exp_result(self, initial_all_vehicles: pd.DataFrame, orders: pd.DataFrame) -> np:

        sim = exp_sim.Simulator(orders, initial_all_vehicles)
        total_completed_orders = 0
        accumulated_reward = 0
        non_assigned_orders = list()
        result = list()
        for time_window in tqdm(range(self.len_time_windows)):
            non_assigned_orders, assigned_orders = sim.taxi_dispatching(time_window, non_assigned_orders)
            total_completed_orders += len(assigned_orders)
            if assigned_orders.keys():
                for order in assigned_orders.values():
                    accumulated_reward += order.reward
        result.append(total_completed_orders)
        result.append(len(orders) - total_completed_orders)
        result.append(total_completed_orders / len(orders))
        result.append(accumulated_reward)

        return np.array(result)

    @staticmethod
    def get_optimal_D(day_type_opt: str, time_opt: str) -> np:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimental_results', day_type_opt, time_opt,
                            'optimal_D.npy')
        return np.load(path)

    @staticmethod
    def save_exp_results(day_type_opt: str, time_opt: str, exp_results: pd.DataFrame):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimental_results', day_type_opt, time_opt,
                            'exp_results' + '_' + day_type_opt + '_' + time_opt + '.csv')
        exp_results.to_csv(path, index=False)

    def test(self, day_type_opt: str, time_opt: str):

        # Phase 1: train the model
        self.train_dqn_model(day_type_opt, time_opt)

        # Phase 2: optimal deployment
        self.get_optimal_vehicle_deployment(day_type_opt, time_opt)

        # Phase 3: performance
        col_names = ['day', 'type', 'num_vehicles', 'completion', 'non-completion', 'completion_ratio', 'total_revenue',
                     'WD']
        exp_results = pd.DataFrame(columns=col_names, index=None)

        # random_total_num_vehicles = [1500, 1750, 2000, 2250, 2500]
        random_total_num_vehicles = np.arange(dm.nph_min_vehicles, dm.ph_max_vehicles, 300)

        # result including order accomplishment， total revenue, order accomplishment ratio, KL divergence
        optimal_D = self.get_optimal_D(day_type_opt, time_opt)
        test_days = self.test_working if day_type_opt == 'working' else self.test_weekend
        for day in test_days:
            print(day)
            # step 1: order distribution
            orders, order_distribution = self.get_order_distribution(day, day_type_opt, time_opt)
            # step 2: random deployment
            for num_vehicles in random_total_num_vehicles:
                rd = self.random_deployment(num_vehicles)
                initial_all_vehicles = self.assign_vehicles(rd)
                result = self.get_exp_result(initial_all_vehicles, orders)
                col = [day, 'r', num_vehicles, result[0], result[1], result[2], result[3],
                       wd(rd.reshape(-1), optimal_D.reshape(-1))]
                exp_results = exp_results.append(pd.DataFrame([col], columns=col_names), ignore_index=True)
            # step 3: optimal deployment
            initial_all_vehicles = self.assign_vehicles(optimal_D)
            result = self.get_exp_result(initial_all_vehicles, orders)
            col = [day, 'opt', np.sum(optimal_D), result[0], result[1], result[2], result[3], 0]
            exp_results = exp_results.append(pd.DataFrame([col], columns=col_names), ignore_index=True)
            # step 4: random deployment based on total number of vehicles of optimal deployment
            rd = self.random_deployment(int(np.sum(optimal_D)))
            initial_all_vehicles = self.assign_vehicles(rd)
            result = self.get_exp_result(initial_all_vehicles, orders)
            col = [day, 'r_b_opt', np.sum(optimal_D), result[0], result[1], result[2], result[3],
                   wd(rd.reshape(-1), optimal_D.reshape(-1))]
            exp_results = exp_results.append(pd.DataFrame([col], columns=col_names), ignore_index=True)
            # step 5: ratio deployment based on random deployment
            for num_vehicles in random_total_num_vehicles:
                rtd = np.around(num_vehicles * order_distribution).astype(int)
                initial_all_vehicles = self.assign_vehicles(rtd)
                result = self.get_exp_result(initial_all_vehicles, orders)
                col = [day, 'rt_b_r', num_vehicles, result[0], result[1], result[2], result[3],
                       wd(rtd.reshape(-1), optimal_D.reshape(-1))]
                exp_results = exp_results.append(pd.DataFrame([col], columns=col_names), ignore_index=True)
            # step 6: ratio deployment based on total number of vehicles of optimal deployment
            rtd = np.around(int(np.sum(optimal_D)) * order_distribution).astype(int)
            initial_all_vehicles = self.assign_vehicles(rtd)
            result = self.get_exp_result(initial_all_vehicles, orders)
            col = [day, 'rt_b_opt', np.sum(optimal_D), result[0], result[1], result[2], result[3],
                   wd(rtd.reshape(-1), optimal_D.reshape(-1))]
            exp_results = exp_results.append(pd.DataFrame([col], columns=col_names), ignore_index=True)

        self.save_exp_results(day_type_opt, time_opt, exp_results)


if __name__ == '__main__':
    fm = FleetManager()
    fm.train_dqn_model('working', '8-10')
