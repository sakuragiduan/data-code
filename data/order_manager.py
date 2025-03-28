class Order(object):

    def __init__(self, **kwargs):
        self.order_id = kwargs['order_id']
        self.start_grid_id = kwargs['start_grid_id']
        self.end_grid_id = kwargs['end_grid_id']
        self.start_time_window = kwargs['start_time_window']
        self.end_time_window = kwargs['end_time_window']
        self.reward = kwargs['reward']
        self.avail_drivers = []  # list of available drivers
        self.state = 0  # 0 means not finished, waiting for allocation; 1: finished; default: 0
