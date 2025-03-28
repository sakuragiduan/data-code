import os
from typing import List
import pandas as pd
import data.map_manager as mm
from data import time_manager as tm
from model.rl.simulator import Simulator
import numpy as np
from tqdm import tqdm
import zipfile
from zipfile import ZipFile
import pickle

ph_min_vehicles = 1500
ph_max_vehicles = 3000
nph_min_vehicles = 1200
nph_max_vehicles = 2700
delta_mum_vehicles = 2700
opt = 'random'


def load_data(file_path, header: bool, is_add_head: bool, col_names: List) -> pd.DataFrame:
    """

    Args:
        file_path: the file Path of experimental data
        header: True if there is head in the experimental data document, False if there is no head in the experimental
                data document
        is_add_head: True if we add a head for the experimental data table. The para. in only considered when
                     header=False
        col_names: the names of columns. The para. in only considered when header=False

    Returns: experimental data

    Author: Peibo Duan

    Date: 06/01/2021

    Fun: load data

    """
    if header:
        data = pd.read_csv(file_path)
    else:
        if is_add_head:
            data = pd.read_csv(file_path, header=None, names=col_names)
        else:
            data = pd.read_csv(file_path, header=None)
    return data


def save_data(file_path: str, data: pd.DataFrame, header: bool):
    """

    Args:
        file_path: file path
        data: experimental data
        header: ture if there is head, otherwise false

    Returns: None

    Author: Peibo Duan

    Date: 06/01/2021

    Fun: save experimental data

    """
    data.to_csv(file_path, header=header, index=False)


def delete_duplicated_data(data: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        data: experimental data

    Returns: experimental data without duplication

    Author: Peibo Duan

    Date: 06/01/2021

    Fun: delete duplicated data

    """
    data = data.drop_duplicates(subset=None, keep='first', inplace=False)
    return data


def data_cleaning(days: List[str]):
    """

    Args:
        days: the list of days

    Returns: None

    Author: Peibo Duan

    Date: 06/01/2021

    Fun: cleaning data (mainly delete duplicated data)

    """
    # the root file path of raw data
    raw_data_r_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'raw_data', 'raw_data.zip')

    # the root file path of experimental data
    cleaned_data_r_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'cleaned_data')

    # column names
    col_names = ['order_id', 'start_timestamp', 'end_timestamp', 'start_long', 'start_lat', 'end_long', 'end_lat',
                 'reward']

    zip_data = ZipFile(raw_data_r_path)
    for day in days:
        print('cleaning data in the day: ', day)
        f = zip_data.open('order_' + day)
        data = load_data(f, False, True, col_names)
        f.close()
        data = delete_duplicated_data(data)
        file_path = os.path.join(cleaned_data_r_path, 'order_' + day + '.csv')
        save_data(file_path, data, True)
    zip_data.close()


def get_data_in_study_site(day: str, start_time: str, end_time: str) -> pd.DataFrame:
    """

    Args:
        day: day
        start_time: start time, e.g., '00:00:00'
        end_time: end time

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get data within the study site

    """

    col_names = ['order_id', 'start_time_window', 'end_time_window', 'start_grid_id', 'start_long', 'start_lat',
                 'end_grid_id', 'end_long', 'end_lat', 'reward']
    root_path_cleaned_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_data')

    print('extract data in the day: ', day)
    start_timestamp = tm.get_timestamp(day, start_time)
    end_timestamp = tm.get_timestamp(day, end_time)
    data_file_path = os.path.join(root_path_cleaned_data, 'order_' + day + '.csv')
    extracted_data = list()
    data = load_data(data_file_path, True, True, [])

    for index, row in data.iterrows():
        start_point = [row['start_long'], row['start_lat']]
        end_point = [row['end_long'], row['end_lat']]
        # outside the study site
        if not mm.is_poi_within_poly(start_point, mm.RECTANGLE):
            continue
        if not mm.is_poi_within_poly(end_point, mm.RECTANGLE):
            continue
        # outside the given time period
        if int(row['start_timestamp']) < start_timestamp or int(row['start_timestamp']) > end_timestamp:
            continue
        if int(row['end_timestamp']) < start_timestamp or int(row['end_timestamp']) > end_timestamp:
            continue

        # get grid id
        start_grid_id = int(mm.get_grid_id_for_poly(start_point))
        end_grid_id = int(mm.get_grid_id_for_poly(end_point))

        # time window
        start_time_window = tm.get_time_window(start_timestamp, int(row['start_timestamp']),
                                               tm.TIME_WINDOW_LEN)
        end_time_window = tm.get_time_window(start_timestamp, int(row['end_timestamp']), tm.TIME_WINDOW_LEN)
        extracted_data.append(
            [row['order_id'], start_time_window, end_time_window, start_grid_id, row['start_long'],
             row['start_lat'], end_grid_id, row['end_long'], row['end_lat'], float(row['reward'])])

    df = pd.DataFrame(extracted_data, columns=col_names)
    df = df.sort_values(by='start_time_window', ascending=True)

    return df


def get_training_data():
    """

    Args: None

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get training data

    """

    # training data for working day morning peak hours
    print('training data for working day morning peak hours')
    start = str(int(tm.wd_m_sph.split(':')[0]))
    end = str(int(tm.wd_m_eph.split(':')[0]))
    train_wd_mph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'training_data',
                                       'working', start + '-' + end)
    for day in tm.train_working:
        data = get_data_in_study_site(day, tm.wd_m_sph, tm.wd_m_eph)
        train_wd_mph_path = os.path.join(train_wd_mph_r_path, day + '.csv')
        save_data(train_wd_mph_path, data, True)

    # training data for working day evening peak hours
    print('training data for working day evening peak hours')
    start = str(int(tm.wd_e_sph.split(':')[0]))
    end = str(int(tm.wd_e_eph.split(':')[0]))
    train_wd_eph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'training_data',
                                       'working', start + '-' + end)
    for day in tm.train_working:
        data = get_data_in_study_site(day, tm.wd_e_sph, tm.wd_e_eph)
        train_wd_eph_path = os.path.join(train_wd_eph_r_path, day + '.csv')
        save_data(train_wd_eph_path, data, True)

    # training data for weekend morning peak hours
    print('training data for weekend morning peak hours')
    start = str(int(tm.wkd_m_sph.split(':')[0]))
    end = str(int(tm.wkd_m_eph.split(':')[0]))
    train_wkd_mph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'training_data',
                                        'weekend', start + '-' + end)
    for day in tm.train_weekend:
        data = get_data_in_study_site(day, tm.wkd_m_sph, tm.wkd_m_eph)
        train_wkd_mph_path = os.path.join(train_wkd_mph_r_path, day + '.csv')
        save_data(train_wkd_mph_path, data, True)

    # training data for weekend evening peak hours
    print('training data for weekend evening peak hours')
    start = str(int(tm.wkd_e_sph.split(':')[0]))
    end = str(int(tm.wkd_e_eph.split(':')[0]))
    train_wkd_eph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'training_data',
                                        'weekend', start + '-' + end)
    for day in tm.train_weekend:
        data = get_data_in_study_site(day, tm.wkd_e_sph, tm.wkd_e_eph)
        train_wkd_eph_path = os.path.join(train_wkd_eph_r_path, day + '.csv')
        save_data(train_wkd_eph_path, data, True)


def get_testing_data():
    """

    Args: None

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get testing data

    """
    # testing data for working day morning peak hours
    print('testing data for working day morning peak hours')
    start = str(int(tm.wd_m_sph.split(':')[0]))
    end = str(int(tm.wd_m_eph.split(':')[0]))
    test_wd_mph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'testing_data',
                                      'working', start + '-' + end)
    for day in tm.test_working:
        data = get_data_in_study_site(day, tm.wd_m_sph, tm.wd_m_eph)
        test_wd_mph_path = os.path.join(test_wd_mph_r_path, day + '.csv')
        save_data(test_wd_mph_path, data, True)

    # testing data for working day evening peak hours
    print('testing data for working day evening peak hours')
    start = str(int(tm.wd_e_sph.split(':')[0]))
    end = str(int(tm.wd_e_eph.split(':')[0]))
    test_wd_eph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'testing_data',
                                      'working', start + '-' + end)
    for day in tm.test_working:
        data = get_data_in_study_site(day, tm.wd_e_sph, tm.wd_e_eph)
        test_wd_eph_path = os.path.join(test_wd_eph_r_path, day + '.csv')
        save_data(test_wd_eph_path, data, True)

    # testing data for weekend morning peak hours
    print('testing data for weekend morning peak hours')
    start = str(int(tm.wkd_m_sph.split(':')[0]))
    end = str(int(tm.wkd_m_eph.split(':')[0]))
    test_wkd_mph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'testing_data',
                                       'weekend', start + '-' + end)
    for day in tm.test_weekend:
        data = get_data_in_study_site(day, tm.wkd_m_sph, tm.wkd_m_eph)
        test_wkd_mph_path = os.path.join(test_wkd_mph_r_path, day + '.csv')
        save_data(test_wkd_mph_path, data, True)

    # testing data for weekend evening peak hours
    print('testing data for weekend evening peak hours')
    start = str(int(tm.wkd_e_sph.split(':')[0]))
    end = str(int(tm.wkd_e_eph.split(':')[0]))
    test_wkd_eph_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'testing_data',
                                       'weekend', start + '-' + end)
    for day in tm.test_weekend:
        data = get_data_in_study_site(day, tm.wkd_e_sph, tm.wkd_e_eph)
        test_wkd_eph_path = os.path.join(test_wkd_eph_r_path, day + '.csv')
        save_data(test_wkd_eph_path, data, True)


def get_population_in_hours(day: str, day_type: str, start_hour: str, end_hour: str, period_type: str):
    """

    Args:
        day: day, e.g., "20121106"
        day_type: "working" or "weekend"
        start_hour: start hour of a day, e.g., "06:00:00"
        end_hour: end hour of a day, e.g., "08:00:00"
        period_type: ph for peak hour and nph for non peak hour

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get population in the given hours of a day based on training data

    """
    start = str(int(start_hour.split(':')[0]))
    end = str(int(end_hour.split(':')[0]))
    population_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'population',
                                     day_type, start + '-' + end)
    training_data_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data',
                                        'training_data', day_type, start + '-' + end)

    if period_type == 'ph':
        min_vehicles = ph_min_vehicles
        max_vehicles = ph_max_vehicles
    else:
        min_vehicles = nph_min_vehicles
        max_vehicles = nph_max_vehicles

    for num_vehicles in range(min_vehicles, max_vehicles + delta_mum_vehicles, delta_mum_vehicles):
        sim = Simulator(os.path.join(training_data_r_path, day + '.csv'), num_vehicles, opt)
        non_assigned_orders = list()
        population_in_hours = list()
        total_completed_orders = 0
        accumulated_reward = 0
        for time_window in range(tm.get_num_time_windows(start_hour, end_hour)):
            avail_vehicles = sim.get_avail_vehicles()
            vehicle_density_distribution = np.zeros(mm.CITY_DIVISION ** 2)
            for grid_id in avail_vehicles.keys():
                vehicle_density_distribution[grid_id] = len(avail_vehicles[grid_id])
            non_assigned_orders, assigned_orders = sim.taxi_dispatching(time_window, non_assigned_orders)
            if assigned_orders.keys():
                for order in assigned_orders.values():
                    population_in_hours.append(
                        [order.start_grid_id, order.start_time_window, order.end_grid_id, order.end_time_window,
                         order.reward, vehicle_density_distribution])
            total_completed_orders += len(assigned_orders)
            if assigned_orders.keys():
                for order in assigned_orders.values():
                    accumulated_reward += order.reward

        print('total orders: {0}, completed orders: {1}, non completed orders: {2}, completion ratio: {3}, '
              'total reward: {4}'.format(len(sim.all_orders), total_completed_orders,
                                         len(sim.all_orders) - total_completed_orders,
                                         total_completed_orders / len(sim.all_orders),
                                         accumulated_reward))

        file_name = day + '_' + opt + '_' + str(num_vehicles)
        file_path = os.path.join(population_r_path, file_name + '.pkl')
        col_names = ['start_grid_id', 'start_time_window', 'end_grid_id', 'end_time_window', 'reward',
                     'vehicle_density_distribution']
        df = pd.DataFrame(population_in_hours, columns=col_names)
        df.to_pickle(file_path)

    # compress the files to save space
    zip_data = ZipFile(os.path.join(population_r_path, 'population.zip'), 'w', zipfile.ZIP_DEFLATED)
    for path, _, filenames in os.walk(population_r_path):
        this_path = os.path.abspath('.')
        r_path = path.replace(this_path, '')
        for filename in filenames:
            if filename[-4:] != '.zip':
                zip_data.write(os.path.join(path, filename), os.path.join(r_path, filename))
    zip_data.close()

    # remove the files
    for _, _, filenames in os.walk(population_r_path):
        for filename in filenames:
            if filename[-4:] != '.zip':
                os.remove(filename)


def get_population():
    """

    Args: None

    Returns: None

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get the population. We select samples from the population to train DQN model

    """
    for day in tm.train_working:
        start = str(int(tm.wd_m_sph.split(':')[0]))
        end = str(int(tm.wd_m_eph.split(':')[0]))
        print('get population within morning peak hour {0}-{1} in working day {2}'.format(start, end, day))
        get_population_in_hours(day, 'working', tm.wd_m_sph, tm.wd_m_eph, 'ph')
        start = str(int(tm.wd_e_sph.split(':')[0]))
        end = str(int(tm.wd_e_eph.split(':')[0]))
        print('get population within evening peak hour {0}-{1} in working day {2}'.format(start, end, day))
        get_population_in_hours(day, 'working', tm.wd_e_sph, tm.wd_e_eph, 'ph')
        start = str(int(tm.wd_snph.split(':')[0]))
        end = str(int(tm.wd_enph.split(':')[0]))
        print('get population within non peak hour {0}-{1} in working day {2}'.format(start, end, day))
        get_population_in_hours(day, 'working', tm.wd_snph, tm.wd_enph, 'nph')

    for day in tm.train_weekend:
        start = str(int(tm.wkd_m_sph.split(':')[0]))
        end = str(int(tm.wkd_m_eph.split(':')[0]))
        print('get population within morning peak hour {0}-{1} at weekend {2}'.format(start, end, day))
        get_population_in_hours(day, 'weekend', tm.wkd_m_sph, tm.wkd_m_eph, 'ph')
        start = str(int(tm.wkd_e_sph.split(':')[0]))
        end = str(int(tm.wkd_e_eph.split(':')[0]))
        print('get population within evening peak hour {0}-{1} at weekend {2}'.format(start, end, day))
        get_population_in_hours(day, 'weekend', tm.wkd_e_sph, tm.wkd_e_eph, 'ph')
        start = str(int(tm.wkd_snph.split(':')[0]))
        end = str(int(tm.wkd_enph.split(':')[0]))
        print('get population within non peak hour {0}-{1} at weekend {2}'.format(start, end, day))
        get_population_in_hours(day, 'weekend', tm.wkd_snph, tm.wkd_enph, 'nph')


def get_samples(population_r_path: str, samples_r_path: str, last_time_window: int, state_size: int):
    """

    Args:
        population_r_path: root path of population
        samples_r_path: path where samples are stored
        last_time_window: the last time window
        state_size: size of state

    Returns: None

    Author: Peibo Duan

    Date: 27/03/2021

    Fun: get the population. We select samples from the population to train DQN model

    """
    #  population_info = os.listdir(population_r_path)
    zip_data = ZipFile(os.path.join(population_r_path, 'population.zip'))
    population_info = zip_data.namelist()
    states = list()
    next_states = list()
    flags = list()
    rewards = list()

    for file_name in tqdm(population_info):
        population = pickle.loads(zip_data.open(file_name).read())
        samples = dict()
        for index, row in population.iterrows():
            if row['end_time_window'] > last_time_window:
                continue
            if row['start_time_window'] not in samples.keys():
                samples[row['start_time_window']] = list()
            sample = [row['start_grid_id'], row['start_time_window'], row['end_grid_id'], row['end_time_window'],
                      row['reward']]
            sample.extend(row['vehicle_density_distribution'])
            samples[row['start_time_window']].append(sample)

        for sample in samples.values():
            if sample[0][3] not in samples.keys():  # there is no next state for current state
                continue
            else:
                # we assume that the the vehicle density distribution (vdd) is constant during a time windows (2 seconds
                # in this paper)

                # vvd in the next state
                candidate_next_states = samples[sample[0][3]]
                next_state_vdd = candidate_next_states[0][5:len(candidate_next_states[0])]

                # vvd in the current state
                state_vdd = sample[0][5:len(sample[0])]

                for s in sample:
                    # state
                    state = [s[0], s[1]]
                    state.extend(state_vdd)
                    states.append(state)

                    # next state
                    next_state = [s[2], s[3]]
                    next_state.extend(next_state_vdd)
                    next_states.append(next_state)

                    # target
                    if last_time_window == s[1]:
                        flags.append(0)
                    else:
                        flags.append(1)
                    rewards.append(s[4])

    zip_data.close()

    # normalized rewards
    rewards = np.array(rewards)
    # print('max - min = {0}'.format(np.max(rewards) - np.min(rewards)))
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

    # normalize vdd in states and next states
    print('get states and next states...')
    states = np.array(states)
    len_states = len(states)
    states_vdd = states[:, 2: state_size]
    # temporally store states in hardware due to limited memory (16G)
    np.save(os.path.join(samples_r_path, 'states.npy'), states)
    del states
    next_states = np.array(next_states)
    next_states_vdd = next_states[:, 2: state_size]
    # temporally store next_states in hardware due to limited memory (16G)
    np.save(os.path.join(samples_r_path, 'next_states.npy'), next_states)
    del next_states

    vdd = np.vstack((states_vdd, next_states_vdd))
    # release memory
    del states_vdd
    del next_states_vdd
    print('normalize vvd...')
    vdd = (vdd - np.min(vdd)) / (np.max(vdd) - np.min(vdd))

    # load states
    print('normalize states and next states...')
    states = np.load(os.path.join(samples_r_path, 'states.npy'))
    states[:, 2: state_size] = vdd[0:len_states, :]

    # load next_states
    next_states = np.load(os.path.join(samples_r_path, 'next_states.npy'))
    next_states[:, 2: state_size] = vdd[len_states: 2 * len_states, :]

    # compress files and remove states.npy and next_states.npy to save space
    np.savez_compressed(os.path.join(samples_r_path, 'samples.npz'), states=states, next_states=next_states,
                        flags=flags, rewards=rewards)
    os.remove(os.path.join(samples_r_path, 'states.npy'))
    os.remove(os.path.join(samples_r_path, 'next_states.npy'))


# if __name__ == '__main__':
#     samples_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'samples',
#                                   'weekend', '19-21')
#     population_r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulated_data', 'population',
#                                      'weekend', '19-21')
#     get_samples(population_r_path, samples_r_path, 3599, 227)
