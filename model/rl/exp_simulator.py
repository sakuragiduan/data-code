import numpy as np
import pandas as pd
from data import vehicle_manager as vm
from data import order_manager as om
from data import time_manager as tm
from data import map_manager as mm
from typing import List, Dict, Tuple

MAX_RESPONSE_TIME_WINDOWS = int(5 * 60 / tm.TIME_WINDOW_LEN)


class Simulator(object):
    class DispatchAlgorithm(object):

        def __init__(self):
            pass

        @staticmethod
        def ldd_alg(order_ids: List[str], vehicle_ids: List[str], matrix_reward: np, matrix_x_variables: np) -> \
                List[Dict]:

            M = len(order_ids)  # the number of orders
            N = len(vehicle_ids)  # the number of vehicles
            theta = 1.0
            gap = 0.0001
            max_iterations = 100  # maximum iterations
            u = np.zeros(N)  # operators
            z_lb = 0  # lower bound
            z_up = float('inf')  # up bound
            best_solution = np.zeros([M, N])

            dispatch_action = list()

            for t in range(1, max_iterations + 1):
                matrix_x = np.zeros([M, N])  # x is the set of binary variables
                QI = np.zeros([M, N])
                for m in range(M):  # for each m
                    QI[m, :] = matrix_reward[m, :] - u

                for m in range(M):
                    idx_1 = np.argwhere(matrix_x_variables[m, :] == 1)
                    if len(idx_1) > 0:
                        max_QI_idx_1 = np.argmax(QI[m, idx_1])
                        matrix_x[m, idx_1[max_QI_idx_1]] = 1

                z_d = np.sum(matrix_reward * matrix_x) + np.sum(u)
                z_up = z_d if z_d < z_up else z_up

                # update the new lower bound of the problem
                best_solution = matrix_x.copy()
                new_z_lb = np.sum(matrix_x * matrix_reward)

                if new_z_lb > z_lb:
                    z_lb = new_z_lb
                    best_solution = matrix_x.copy()

                # update u
                total_sum = 0
                for n in range(N):
                    sum_m = np.sum(matrix_x[:, n])
                    total_sum += (1 - sum_m) ** 2
                if total_sum != 0:
                    k_t = theta * (z_d - z_lb) / total_sum
                else:
                    k_t = theta * (z_d - z_lb)

                for n in range(N):
                    sum_m = np.sum(matrix_x[:, n])
                    temp = u[n] + k_t * (sum_m - 1) / t
                    u[n] = temp if temp > 0 else 0

                if (z_up - z_lb) / z_up <= gap:
                    break

            # find the situation that a vehicle is allocated to multiple orders
            for n in range(N):
                index_orders_allocated_to_n = np.argwhere(best_solution[:, n] == 1)
                # to justify whether a taxi is only allocated to an unique order
                if len(index_orders_allocated_to_n) > 1:
                    index_m_max_reward = index_orders_allocated_to_n[
                        np.argmax(matrix_reward[index_orders_allocated_to_n, n])]
                    best_solution[:, n] = 0
                    best_solution[index_m_max_reward, n] = 1

            # use naive method to increase the reward if there is order without assigned any vehicle
            index_drivers_with_order = list()
            index_drivers_without_order = list()
            index_orders_without_driver = list()
            for m in range(M):
                indices = np.argwhere(best_solution[m, :] == 1)
                if len(indices) > 0:
                    idx = indices[0][0]
                    index_drivers_with_order.append(idx)
                else:
                    index_orders_without_driver.append(m)

            if len(index_orders_without_driver) != 0:  # there is order without driver to allocate
                for n in range(N):
                    if n not in index_drivers_with_order:
                        index_drivers_without_order.append(n)

                second_allocated_driver = []
                for m in index_orders_without_driver:
                    index_driver = -1
                    current_reward = -1
                    for n in index_drivers_without_order:
                        if n not in second_allocated_driver:
                            if matrix_reward[m][n] > current_reward:
                                index_driver = n
                                current_reward = matrix_reward[m][n]
                    if index_driver != -1:
                        second_allocated_driver.append(index_driver)
                        best_solution[m][index_driver] = 1

            # solution
            for m in range(M):
                indices = np.argwhere(best_solution[m, :] == 1)
                if len(indices) > 0:
                    idx = indices[0][0]
                    dispatch_action.append(dict(order_id=order_ids[m], vehicle_id=vehicle_ids[idx]))

            return dispatch_action

        @staticmethod
        def greedy_alg(candidates: List[Dict]) -> List[Dict]:

            dispatch_action = list()
            candidates.sort(key=lambda od_info: od_info['reward_unit'], reverse=True)
            assigned_order_ids = set()
            assigned_vehicle_ids = set()
            for candidate in candidates:
                if (candidate['order_id'] in assigned_order_ids) or (candidate['vehicle_id'] in assigned_vehicle_ids):
                    continue
                assigned_order_ids.add(candidate['order_id'])
                assigned_vehicle_ids.add(candidate['vehicle_id'])
                dispatch_action.append(dict(order_id=candidate['order_id'], vehicle_id=candidate['vehicle_id']))

            return dispatch_action

    def __init__(self, all_orders: pd.DataFrame, all_vehicles: pd.DataFrame):
        """

        Args:
            all_orders: all orders
            num_vehicles: the initial number of vehicles deployed in the study site
            opt: 'random' or 'uniform'

        Returns: None

        Author: Peibo Duan

        Date: 16/01/2021

        Fun: get all orders in the day, all available vehicles

        """

        # all orders
        self.all_orders = all_orders  # dataframe

        # all vehicles
        self.all_vehicles = all_vehicles  # dataframe

        # adjacent matrix
        self.adjacent_matrix = mm.get_adjacent_matrix()

    def get_orders_in_time_window(self, time_window: int) -> List[om.Order]:
        """

        Args:
            time_window: time window

        Returns:
            List[Order], the list of orders

        Author: Peibo Duan

        Date: 16/01/2021

        Fun: get orders within the time window

        """

        df = self.all_orders[self.all_orders['start_time_window'] == time_window]
        orders = list()
        for idx, row in df.iterrows():
            kwargs = {'order_id': row['order_id'], 'start_grid_id': row['start_grid_id'],
                      'end_grid_id': row['end_grid_id'], 'start_time_window': row['start_time_window'],
                      'end_time_window': row['end_time_window'], 'reward': row['reward']}
            orders.append(om.Order(**kwargs))

        return orders

    def get_avail_vehicles(self) -> Dict[int, List[str]]:
        """

        Args: None

        Returns: the list of available vehicles, {grid id: [avail_vehicle id,....]}

        Author: Peibo Duan

        Date: 16/01/2021

        Fun: get all available vehicles within the time window

        """
        avail_vehicles = dict()
        df_avail_vehicles = self.all_vehicles[self.all_vehicles['state'] == 'cruising']
        for index, row in df_avail_vehicles.iterrows():
            grid_id = row['grid_id']
            if grid_id not in avail_vehicles.keys():
                avail_vehicles[grid_id] = list()
            avail_vehicles[grid_id].append(row['vehicle_id'])

        return avail_vehicles

    def release_occupied_vehicles(self, time_window: int):
        """

        Args: time_window: time window

        Returns: None

        Author: Peibo Duan

        Date: 16/01/2021

        Fun: release occupied vehicles

        """

        # change states of vehicles who will complete the orders in the current time window
        self.all_vehicles.loc[(self.all_vehicles['time_window'] == time_window) & (
                self.all_vehicles['state'] == 'occupied'), 'state'] = 'cruising'

    def lock_occupied_vehicles(self, assigned_vehicles: Dict[str, int]):
        """

        Args: assigned_vehicles: {vehicle_id: time window when the order will be completed}

        Returns: None

        Author: Peibo Duan

        Date: 16/01/2021

        Fun: lock occupied vehicles

        """

        # change states of vehicles who are assigned orders
        for assigned_vehicle_id in assigned_vehicles.keys():
            self.all_vehicles.loc[self.all_vehicles['vehicle_id'] == assigned_vehicle_id, 'state'] = 'occupied'
            self.all_vehicles.loc[self.all_vehicles['vehicle_id'] == assigned_vehicle_id, 'time_window'] = \
                assigned_vehicles[assigned_vehicle_id]

    def taxi_dispatching(self, time_window: int, accumulated_orders: List[om.Order]) -> Tuple[
                        List[om.Order], Dict[str, om.Order]]:

        # delete orders with waiting time longer than the maximum waiting time windows
        filtered_accumulated_orders = list()
        for order in accumulated_orders:
            if time_window - order.start_time_window > MAX_RESPONSE_TIME_WINDOWS:
                continue
            else:
                filtered_accumulated_orders.append(order)

        # all orders  = orders accumulated in previous time windows + orders generated in the current time
        orders_in_time_window = self.get_orders_in_time_window(time_window)
        orders_in_time_window.extend(filtered_accumulated_orders)

        # update states of vehicles
        self.release_occupied_vehicles(time_window)

        # all available vehicles, note there may be no available vehicles surrounding a grid with orders
        avail_vehicles = self.get_avail_vehicles()

        order_ids = list()
        vehicle_ids = set()
        non_dispatched_orders = list()
        dispatched_orders = dict()

        # get candidates
        candidates = list()  # List[Dict[order_id: _, driver_id: _, reward_unit: _]]
        for order in orders_in_time_window:

            neighboring_grid_ids = np.where(self.adjacent_matrix[order.start_grid_id] == 1)
            neighboring_grid_ids = list(neighboring_grid_ids[0])
            neighboring_grid_ids.append(order.start_grid_id)

            if len(set(neighboring_grid_ids) & set(avail_vehicles.keys())) == 0:
                # there are no available vehicles in neighboring grids
                non_dispatched_orders.append(order)
                continue
            else:
                order_ids.append(order.order_id)
                for grid_id in neighboring_grid_ids:
                    if grid_id in avail_vehicles.keys():  # there are available vehicles in neighboring grids
                        for vehicle_id in avail_vehicles[grid_id]:
                            vehicle_ids.add(vehicle_id)
                            candidates.append(
                                dict(order_id=order.order_id, order_idx=len(order_ids) - 1, vehicle_id=vehicle_id,
                                     vehicle_idx=len(vehicle_ids) - 1, reward_unit=order.reward))

        # dispatching
        if order_ids:
            vehicle_ids = list(vehicle_ids)
            expected_reward = np.zeros([len(order_ids), len(vehicle_ids)])
            match_state = np.zeros([len(order_ids), len(vehicle_ids)])
            for candidate in candidates:
                m = candidate['order_idx']
                n = candidate['vehicle_idx']
                expected_reward[m][n] = candidate['reward_unit']
                match_state[m][n] = 1
            # ldd based algorithm
            da = self.DispatchAlgorithm()
            dispatch_action = da.ldd_alg(order_ids, vehicle_ids, expected_reward, match_state)
        else:  # there is no order
            dispatch_action = list()

        # dispatched order ids and left order ids
        dispatched_order_ids = list()
        for action in dispatch_action:
            dispatched_order_ids.append(action['order_id'])
        left_order_ids = list(set(order_ids) - set(dispatched_order_ids))  # left orders after implementing LDD

        # dispatched and non dispatched orders
        for order in orders_in_time_window:
            if order.order_id in dispatched_order_ids:
                dispatched_orders[order.order_id] = order
            elif order.order_id in left_order_ids:
                non_dispatched_orders.append(order)

        # update states of vehicles
        dispatched_vehicles = dict()
        for action in dispatch_action:
            dispatched_vehicles[action['vehicle_id']] = dispatched_orders[action['order_id']].end_time_window
        self.lock_occupied_vehicles(dispatched_vehicles)

        return non_dispatched_orders, dispatched_orders
