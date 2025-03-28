from datetime import datetime


# global variable, we have data collected from 30 days in Nov. 2016
TIME_WINDOW_LEN = 2  # 2 seconds for a time window

# working days with data used for training
train_working = ['20161107', '20161108', '20161109', '20161110', '20161111',
                 '20161114', '20161115', '20161116', '20161117', '20161118',
                 '20161121', '20161122', '20161123', '20161124', '20161125']
# weekend days with data used for training
train_weekend = ['20161105', '20161106', '20161112', '20161113', '20161119', '20161120']
# working days with data used for testing
test_working = ['20161128', '20161129', '20161130']
# weekend days with data used for testing
test_weekend = ['20161126', '20161127']
wd_m_sph = '08:00:00'  # working day morning start time of peak hour
wd_m_eph = '10:00:00'  # working day morning end time of peak hour
wd_e_sph = '18:00:00'  # working day evening start time of peak hour
wd_e_eph = '20:00:00'  # working day evening end time of peak hour
wd_snph = '11:00:00'  # working day start time of non peak hour
wd_enph = '13:00:00'  # working day end time of non peak hour
wkd_m_sph = '09:00:00'  # weekend morning start peak hour
wkd_m_eph = '11:00:00'  # weekend morning end peak hour
wkd_e_sph = '19:00:00'  # weekend evening start peak hour
wkd_e_eph = '21:00:00'  # weekend evening end peak hour
wkd_snph = '13:00:00'  # weekend day start time of non peak hour
wkd_enph = '15:00:00'  # weekend day end time of non peak hour


# calculate timestamp based on the time
def get_timestamp(day: str, time: str) -> int:
    """

    Args:
        day: '20161101'
        time: '00:00:00'

    Returns:
        timestamp

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get timestamp of the given time

    """
    year = int(day[0:4])
    month = int(day[4:6])
    day = int(day[6:8])
    start_timestamp_of_day = int(datetime(year, month, day, 0, 0).timestamp())

    hour = int(time[0:2])
    minute = int(time[3:5])
    second = int(time[6:8])

    return start_timestamp_of_day + hour * 60 * 60 + minute * 60 + second


def get_time_window(start_timestamp: int, current_timestamp: int, time_window_len: int) -> int:
    """

    Args:
        start_timestamp: start timestamp
        current_timestamp: current timestamp
        time_window_len: the length of a time window

    Returns:
        time window for current timestamp

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get time window of a given timestamp

    """
    return int((current_timestamp - start_timestamp) / time_window_len)


def get_num_time_windows(start_time: str, end_time: str) -> int:
    """

    Args:
        start_time: start time, e.g., '12:00:00'
        end_time: end time

    Returns:
        the number of time windows between start and end time

    Author: Peibo Duan

    Date: 12/01/2021

    Fun: get the number of time windows

    """
    start_time = datetime.strptime(start_time, '%H:%M:%S')
    end_time = datetime.strptime(end_time, '%H:%M:%S')
    seconds = (end_time - start_time).seconds
    return int(seconds / TIME_WINDOW_LEN)
