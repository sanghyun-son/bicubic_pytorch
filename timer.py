import time


class Timer(object):

    def __init__(self, msg):
        self.msg = msg.replace('{}', '{:.6f}s')
        self.tic = None

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, *args, **kwargs):
        toc = time.time() - self.tic
        print('\n' + self.msg.format(toc))

