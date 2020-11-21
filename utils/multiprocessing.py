from multiprocessing import Pool

def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

class ThreadPool:
    def __init__(self, func, args, num_threads=None):
        self.func = func
        self.args = args
        self.process_pool = Pool(num_threads)

    def start(self):
        return self.process_pool.starmap(self.func, self.args) # -> result list

    def start_async(self):
        return self.process_pool.starmap_async(self.func, self.args) # access via result.get() --> async result list

    def join(self):
        self.process_pool.close()
        self.process_pool.join()