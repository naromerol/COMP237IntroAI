"""
@author Nestor Romero leon
StudentId 301133331
Linear regression Assignment - Exercise 2
"""

import pandas as pd

ecom_exp_nestor = pd.read_csv('./Ecom Expense.csv')

print(ecom_exp_nestor.head(3))
print('Shape data (rows,cols): ', ecom_exp_nestor.shape)
print(ecom_exp_nestor.dtypes)
print(ecom_exp_nestor.columns.values)


#print(ecom_exp_nestor.describe())