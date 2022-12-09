import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from tqdm import tqdm
from multiprocessing import Pool

def get_root(id, budget, roi, eps = 1e-5):
    data = pd.read_csv("1458_2_data.csv")
    s, _, scale = lognorm.fit(data['paying_price'])
    f = lambda x : lognorm.pdf(x, s = s, loc=0, scale=scale)
    g = lambda x : lognorm.pdf(x, s = s, loc = 0, scale = scale)
    v_min = 0
    v_max = 300
    beta_budget = budget_constraint(budget, f, v_min, v_max, g, eps=eps)
    beta_roi = ROI_constraint(roi, f, v_min, v_max, g, eps=eps)
    return [int(id), budget, roi, beta_budget, beta_roi, min(beta_budget, beta_roi)]

def f(x):
    if x >= 0 and x <= 300:
        return 1.0 / 300
    else:
        return 0

if __name__ == "__main__":
    budget_min, budget_max = 0, 300
    ROI_min, ROI_max = 0, 5
    data = sample(budget_min, budget_max, ROI_min, ROI_max, 10)
    pool = Pool(processes=35)
    result = []
    def func(x, y, z):
        return get_root(x, y, z,)
    for idx, value in data.iterrows():
        result.append(pool.apply_async(get_root, args=(value.id, value.budget, value.roi, )))
    pool.close()
    pool.join()
    data = []
    for i in result:
        data.append(i.get())

    data = pd.DataFrame(data, columns=['id', 'budget', 'roi', 'beta_budget', 'beta_roi', 'beta'])
    data.to_csv("exp_3.csv")