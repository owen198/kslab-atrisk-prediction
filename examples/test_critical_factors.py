# coding: utf-8
import pandas as pd

from critical_factors.critical_factors import CriticalFactors

data = pd.read_csv('data/data.csv', sep=',')
critical_factors = CriticalFactors(data, 1, 21)

critical_factors.save_to_csv('result/test.csv')
