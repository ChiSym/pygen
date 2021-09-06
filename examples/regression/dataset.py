import random

import torch

def make_data_set(n, seed=1):
    random.seed(seed)
    p_outlier = .5
    inlier_std = .5
    outlier_std = 5.
    slope = -1
    intercept = 2
    xs = torch.linspace(-5, 5, n)
    ys = torch.zeros(len(xs))
    for i, x in enumerate(xs):
        epsilon = random.normalvariate(0, 1)
        noise = [inlier_std, outlier_std][random.random() < p_outlier]
        ys[i] = slope * x + intercept + epsilon * noise
    return xs, ys
