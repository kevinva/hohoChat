import time


def logTime():
    current_time = time.time()
    local_time = time.localtime(current_time)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    return time_str


def strForTime_YmdHMS():
    current_time = time.time()
    local_time = time.localtime(current_time)
    time_str = time.strftime('%Y%m%d%H%M%S', local_time)
    return time_str