from typing import List, Dict
import numpy as np
import math
import pandas as pd
import data.map_manager as mm
import data.data_manager as dm
import os


class Vehicle(object):

    def __init__(self, vehicle_id: str, time_window: int, grid_id: str, state: str):
        self.vehicle_id = vehicle_id  # vehicle id
        self.time_window = time_window
        self.grid_id = grid_id  # the grid id where the taxi stays
        self.state = state  # the state of vehicle, including ['offline', 'occupied', 'cruising']

    def __eq__(self, other):
        return self.vehicle_id == other.id

    def __hash__(self):
        return hash(self.vehicle_id)


def initialize_vehicles(num_vehicles: int, opt: str) -> pd.DataFrame:  # {vehicle_id, vehicle information}
    """

    Args:
        num_vehicles: the number of vehicles
        opt: the option used for initializing vehicles

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: initialize vehicles

    """

    vehicles = dict()
    # generate vehicle ids
    for num in range(num_vehicles):
        vehicles[str(num)] = Vehicle(str(num), 0, 'null', 'cruising')

    # assign vehicles into different grid ids
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'study_site.csv')
    study_site = mm.read_map(file_path)
    grid_ids = study_site['grid_id'].values.tolist()
    initialize_vehicle_distribution(grid_ids, vehicles, opt)
    # trained_model
    col_names = ['vehicle_id', 'time_window', 'grid_id', 'state']
    l_vehicles = list()
    for vehicle in vehicles.values():
        l_vehicles.append([vehicle.vehicle_id, vehicle.time_window, vehicle.grid_id, vehicle.state])
    df_vehicles = pd.DataFrame(l_vehicles, columns=col_names)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vehicle', 'vehicles_initialization.csv')
    dm.save_data(file_path, df_vehicles, True)
    df_available_vehicles = df_vehicles[df_vehicles['state'] == 'cruising']
    return df_available_vehicles


def initialize_vehicle_distribution(grids: List[str], vehicles: Dict[str, Vehicle], opt: str):
    """

    Args:
        grids: grids in the study site
        vehicles: vehicles
        opt: the option used for initializing vehicles, e.g., uniform, random...

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: initialize the geographical distribution of given nubmer of vehicles

    """
    # vehicle ids
    vehicles_to_allocated = list()
    for vehicle_id in vehicles.keys():
        vehicles_to_allocated.append(vehicle_id)

    if opt == 'random':
        positions_idx = np.random.choice(len(grids), size=len(vehicles_to_allocated), replace=True)
        index = 0
        for idx in positions_idx:
            grid_id = grids[idx]
            vehicle_id = vehicles_to_allocated[index]
            vehicles[vehicle_id].grid_id = grid_id
            index += 1
    elif opt == 'uniform':
        num_vehicles = len(vehicles_to_allocated)
        num_grids = len(grids)
        avg_num_vehicles_in_grid = math.floor(num_vehicles / num_grids)
        for i in range(num_grids):
            grid_id = grids[i]
            for j in range(avg_num_vehicles_in_grid):
                vehicle_id = vehicles_to_allocated[i * avg_num_vehicles_in_grid + j]
                vehicles[vehicle_id].grid_id = grid_id
        vehicles_left = dict()
        if num_vehicles - avg_num_vehicles_in_grid * num_grids > 0:
            vehicles_to_allocated = vehicles_to_allocated[avg_num_vehicles_in_grid * num_grids:num_vehicles]
            for vehicle_id in vehicles_to_allocated:
                vehicles_left[vehicle_id] = vehicles[vehicle_id]
            initialize_vehicle_distribution(grids, vehicles_left, 'random')
            for vehicle_id in vehicles_left.keys():
                vehicles[vehicle_id].grid_id = vehicles_left[vehicle_id].grid_id
    elif type == 'ratio':
        """ 
        It will be completed in the future. The initialization of vehicle distribution is based on the statistic of 
        taxi trajectories distribution using real experimental data from 30 days
        """
        pass


def get_vehicle_density_distribution(vehicles: pd.DataFrame) -> np:
    """

    Args:
        vehicles: vehicles

    Returns:
        the vehicle density distribution of vehicles

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get the vehicle density distribution of vehicles

    """
    vehicle_density_distribution = np.zeros([mm.CITY_DIVISION, mm.CITY_DIVISION])
    available_vehicles = vehicles[vehicles['state'] == 'cruising']
    grid_ids = np.array(available_vehicles['grid_id'])
    for row in range(mm.CITY_DIVISION):
        for col in range(mm.CITY_DIVISION):
            grid_id = row * mm.CITY_DIVISION + col
            vehicle_density_distribution[row][col] = np.sum(grid_ids == grid_id)
    return vehicle_density_distribution
