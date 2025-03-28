import data.map_manager as mm
import numpy as np
from model.rl.dqn import DQN


c = 0.05


def cal_phi(D: np, grid_ids: np, t: int, dqn: DQN) -> np:
    """

    Args:
        D: vehicle density distribution
        grid_ids: grid ids
        t: time window
        dqn: DQN model

    Returns:
        phi: the parameter phi

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: calculate phi

    """
    states = list()
    update_vvd = np.array(D.reshape(-1))
    update_vvd = (update_vvd - np.min(update_vvd)) / (np.max(update_vvd) - np.min(update_vvd))
    for j in range(mm.CITY_DIVISION):
        for k in range(mm.CITY_DIVISION):
            state = [grid_ids[j][k], t]
            state.extend(update_vvd)
            np.array(state)
            states.append(state)
    states = np.array(states)
    reward = dqn.predict(states)
    reward = reward.reshape(-1) * D.reshape(-1)
    reward -= c * D.reshape(-1)
    phi = reward.reshape((mm.CITY_DIVISION, mm.CITY_DIVISION))

    return phi


def dgd_based_algo(max_iterations: int, dqn: DQN, lb: int, ub: int) -> np:
    """

    Args:
        max_iterations: maximum iterations
        dqn: DQN model
        lb: lower bound
        ub: upper bound

    Returns:
        final_D: optimal vehicle deployment

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: discrete GD based algorithm

    """

    # initialization
    final_D = np.random.randint(low=lb, high=ub, size=mm.CITY_DIVISION ** 2, dtype=int)
    pre_sum_phi = 0
    for _ in range(max_iterations):
        D = np.random.randint(low=lb, high=ub, size=mm.CITY_DIVISION**2, dtype=int)
        D = D.reshape((mm.CITY_DIVISION, mm.CITY_DIVISION))
        initial_num_D = np.sum(D)
        grid_ids = np.array(range(0, mm.CITY_DIVISION*mm.CITY_DIVISION)).reshape((mm.CITY_DIVISION, mm.CITY_DIVISION))
        t = 0
        phi = cal_phi(D, grid_ids, t, dqn)
        sum_phi = np.sum(phi)
        flag = True

        print('running algorithm...')

        f = np.zeros((mm.CITY_DIVISION, mm.CITY_DIVISION), dtype=int)

        while flag:
            for j in range(mm.CITY_DIVISION):
                for k in range(mm.CITY_DIVISION):
                    if f[j][k] == 0:
                        add_D = D.copy()
                        sub_D = D.copy()

                        # add 1
                        add_D[j][k] += 1
                        update_phi_j_k_add = np.sum(cal_phi(add_D, grid_ids, t, dqn))

                        # subscribe 1
                        sub_D[j][k] = max(0, sub_D[j][k] - 1)
                        update_phi_j_k_sub = np.sum(cal_phi(sub_D, grid_ids, t, dqn))

                        if max(update_phi_j_k_sub, update_phi_j_k_add) < sum_phi:
                            f[j][k] = 1
                        else:
                            if update_phi_j_k_sub > update_phi_j_k_add:
                                D[j][k] = sub_D[j][k]
                                sum_phi = update_phi_j_k_sub
                            else:
                                D[j][k] = add_D[j][k]
                                sum_phi = update_phi_j_k_add

            print(sum_phi, np.sum(D))
            if np.all(f):  # or abs(pre_sum_phi - sum_phi) < epsilon:
                flag = False
            else:
                f = np.zeros((mm.CITY_DIVISION, mm.CITY_DIVISION), dtype=int)

        if pre_sum_phi < sum_phi:
            pre_sum_phi = sum_phi
            final_D = np.copy(D)

        print('the number of vehicles needed is {0} with initial distribution {1}'.format(np.sum(D), initial_num_D))

    print(final_D)

    return final_D


# if __name__ == '__main__':
#     dgd_based_algo()
