from typing import List

import numpy as np
from numpy import sqrt

G_beta_0 = np.array([
    [0],
])

G_beta_1 = np.array([
    [0, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
])

G_beta_2 = np.array([
    [0, 1, 0, 0, 0],
    [-1, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(3), 0],
    [0, 0, sqrt(3), 0, -1],
    [0, 0, 0, 1, 0],
])

G_beta_3 = np.array([
    [0, sqrt(6)/2, 0, 0, 0, 0, 0],
    [-sqrt(6)/2, 0, sqrt(10)/2, 0, 0, 0, 0],
    [0, -sqrt(10)/2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -sqrt(6), 0, 0],
    [0, 0, 0, sqrt(6), 0, -sqrt(10)/2, 0],
    [0, 0, 0, 0, sqrt(10)/2, 0, -sqrt(6)/2],
    [0, 0, 0, 0, 0, sqrt(6)/2, 0],
])

G_beta_4 = np.array([
    [0, sqrt(2), 0, 0, 0, 0, 0, 0, 0],
    [-sqrt(2), 0, sqrt(14)/2, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(14)/2, 0, 3*sqrt(2)/2, 0, 0, 0, 0, 0],
    [0, 0, -3*sqrt(2)/2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -sqrt(10), 0, 0, 0],
    [0, 0, 0, 0, sqrt(10), 0, -3*sqrt(2)/2, 0, 0],
    [0, 0, 0, 0, 0, 3*sqrt(2)/2, 0, -sqrt(14)/2, 0],
    [0, 0, 0, 0, 0, 0, sqrt(14)/2, 0, -sqrt(2)],
    [0, 0, 0, 0, 0, 0, 0, sqrt(2), 0],
])

G_beta_5 = np.array([
    [0, sqrt(10)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-sqrt(10)/2, 0, 3*sqrt(2)/2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -3*sqrt(2)/2, 0, sqrt(6), 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -sqrt(6), 0, sqrt(7), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(7), 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -sqrt(15), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, sqrt(15), 0, -sqrt(7), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, sqrt(7), 0, -sqrt(6), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, sqrt(6), 0, -3*sqrt(2)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(2)/2, 0, -sqrt(10)/2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(10)/2, 0],
])

G_beta_6 = np.array([
    [0, sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-sqrt(3), 0, sqrt(22)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(22)/2, 0, sqrt(30)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -sqrt(30)/2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -3, 0, sqrt(10), 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -sqrt(10), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -sqrt(21), 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, sqrt(21), 0, -sqrt(10), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, sqrt(10), 0, -3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -sqrt(30)/2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(30)/2, 0, -sqrt(22)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(22)/2, 0, -sqrt(3)],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(3), 0],
])

G_beta_7 = np.array([
    [0, sqrt(14)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-sqrt(14)/2, 0, sqrt(26)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(26)/2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -3, 0, sqrt(11), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(11), 0, 5*sqrt(2)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -5*sqrt(2)/2, 0, 3*sqrt(6)/2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -3*sqrt(6)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -2*sqrt(7), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2*sqrt(7), 0, -3*sqrt(6)/2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(6)/2, 0, -5*sqrt(2)/2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5*sqrt(2)/2, 0, -sqrt(11), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(11), 0, -3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -sqrt(26)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(26)/2, 0, -sqrt(14)/2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(14)/2, 0],
])

G_beta_8 = np.array([
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2, 0, sqrt(30)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(30)/2, 0, sqrt(42)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -sqrt(42)/2, 0, sqrt(13), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(13), 0, sqrt(15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -sqrt(15), 0, sqrt(66)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -sqrt(66)/2, 0, sqrt(70)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -sqrt(70)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, -sqrt(70)/2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(70)/2, 0, -sqrt(66)/2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(66)/2, 0, -sqrt(15), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15), 0, -sqrt(13), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(13), 0, -sqrt(42)/2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(42)/2, 0, -sqrt(30)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(30)/2, 0, -2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
])

G_beta_9 = np.array([
    [0, 3*sqrt(2)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-3*sqrt(2)/2, 0, sqrt(34)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(34)/2, 0, 2*sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -2*sqrt(3), 0, sqrt(15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(15), 0, sqrt(70)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -sqrt(70)/2, 0, sqrt(78)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -sqrt(78)/2, 0, sqrt(21), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -sqrt(21), 0, sqrt(22), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -sqrt(22), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3*sqrt(5), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(5), 0, -sqrt(22), 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(22), 0, -sqrt(21), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(21), 0, -sqrt(78)/2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(78)/2, 0, -sqrt(70)/2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(70)/2, 0, -sqrt(15), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15), 0, -2*sqrt(3), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*sqrt(3), 0, -sqrt(34)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(34)/2, 0, -3*sqrt(2)/2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(2)/2, 0],
])

G_beta_10 = np.array([
    [0, sqrt(5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-sqrt(5), 0, sqrt(38)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(38)/2, 0, 3*sqrt(6)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -3*sqrt(6)/2, 0, sqrt(17), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(17), 0, 2*sqrt(5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -2*sqrt(5), 0, 3*sqrt(10)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -3*sqrt(10)/2, 0, 7*sqrt(2)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -7*sqrt(2)/2, 0, sqrt(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -sqrt(26), 0, 3*sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3*sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sqrt(55), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(55), 0, -3*sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(3), 0, -sqrt(26), 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(26), 0, -7*sqrt(2)/2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7*sqrt(2)/2, 0, -3*sqrt(10)/2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(10)/2, 0, -2*sqrt(5), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*sqrt(5), 0, -sqrt(17), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(17), 0, -3*sqrt(6)/2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(6)/2, 0, -sqrt(38)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(38)/2, 0, -sqrt(5)],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(5), 0],
])

G_beta_11 = np.array([
    [0, sqrt(22)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-sqrt(22)/2, 0, sqrt(42)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -sqrt(42)/2, 0, sqrt(15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -sqrt(15), 0, sqrt(19), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -sqrt(19), 0, 3*sqrt(10)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -3*sqrt(10)/2, 0, sqrt(102)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -sqrt(102)/2, 0, 2*sqrt(7), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -2*sqrt(7), 0, sqrt(30), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -sqrt(30), 0, 3*sqrt(14)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3*sqrt(14)/2, 0, sqrt(130)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -sqrt(130)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sqrt(66), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(66), 0, -sqrt(130)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(130)/2, 0, -3*sqrt(14)/2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(14)/2, 0, -sqrt(30), 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(30), 0, -2*sqrt(7), 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*sqrt(7), 0, -sqrt(102)/2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(102)/2, 0, -3*sqrt(10)/2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*sqrt(10)/2, 0, -sqrt(19), 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(19), 0, -sqrt(15), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15), 0, -sqrt(42)/2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(42)/2, 0, -sqrt(22)/2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(22)/2, 0],
])

G_beta: List[np.array] = [G_beta_0, G_beta_1, G_beta_2, G_beta_3, G_beta_4, G_beta_5, G_beta_6, G_beta_7, G_beta_8, G_beta_9, G_beta_10, G_beta_11]
