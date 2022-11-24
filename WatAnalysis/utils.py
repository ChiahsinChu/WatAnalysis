import numpy as np

def get_cum_ave(data):
    cum_sum = data.cumsum()
    cum_ave = cum_sum / (np.arange(len(data)) + 1)
    return cum_ave
