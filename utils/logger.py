from IPython.display import clear_output
import numpy as np
import timeit

class ProgressLogger:
    def __init__(self, total, func=None):
        self.total = total
        self.cnt = 0
        self.start = timeit.default_timer()
        self.func = func

    def log_progress(self):
        self.cnt += 1
        clear_output(wait=True)
        stop = timeit.default_timer()
        if (self.cnt * 100 / self.total) < 5:
            expected_time = "Calculating..."
        else:
            expected_time = np.round(((stop - self.start) / (self.cnt / self.total)) / 60, 2)
        curr_prog = np.round(self.cnt * 100 / self.total, 2)
        curr_rt = np.round((stop - self.start) / 60, 2)
        if isinstance(expected_time, (int, float)):
            expc_rmt = np.round(expected_time - curr_rt, 2)
        else:
            expc_rmt = "Calculating..."
        if self.func is not None:
            print("Running function '{}'..".format(self.func.__name__))
        print("Current Progress: {}%".format(curr_prog))
        print("Current Run Time: {} minutes".format(curr_rt))
        print("Expected Run Time: {} minutes".format(expected_time))
        print("-------------------------------------")
        print("Expected Remaining Time: {} minutes".format(expc_rmt))