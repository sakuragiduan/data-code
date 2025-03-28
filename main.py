from fleet_manager import FleetManager

if __name__ == '__main__':
    fm = FleetManager()
    # working day morning peak hour 8-10: pass
    # day_type_opt = 'working'
    # time_opt = '8-10'
    # print(day_type_opt, time_opt)
    # fm.test(day_type_opt, time_opt)
    # working day non peak hour 11-13
    day_type_opt = 'working'
    time_opt = '11-13'
    print(day_type_opt, time_opt)
    fm.test(day_type_opt, time_opt)
    # # working day evening peak hour 18-20
    # day_type_opt = 'working'
    # time_opt = '18-20'
    # print(day_type_opt, time_opt)
    # fm.test(day_type_opt, time_opt)
    # # weekend morning peak hour 9-11
    # day_type_opt = 'weekend'
    # time_opt = '9-11'
    # print(day_type_opt, time_opt)
    # fm.test(day_type_opt, time_opt)
    # weekend non peak hour 13-15
    # day_type_opt = 'weekend'
    # time_opt = '13-15'
    # print(day_type_opt, time_opt)
    # fm.test(day_type_opt, time_opt)
    # weekend evening peak hour 19-21
    # day_type_opt = 'weekend'
    # time_opt = '19-21'v
    # print(day_type_opt, time_opt)
    # fm.test(day_type_opt, time_opt)
