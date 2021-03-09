# -*- coding: utf-8 -*-
"""
Logistic Regression Assignment - Exercise 1

@author: Nestor Romero leon
Student Id - 301133331 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

titanic_nestor = pd.read_csv('titanic.csv')

"""
    INITIAL DATA ANALYSIS
"""

print('DATAFRAME HEAD SAMPLE\n', titanic_nestor.head(3))
print('\nDATAFRAME SHAPE\n', titanic_nestor.shape)
print('\nDATAFRAME INFO\n')
print(titanic_nestor.info(show_counts=True))

print('>> Example unique information id : ', titanic_nestor['PassengerId'][30])
print('>> Example passenger name : ', titanic_nestor['Name'][30])
print('>> Example ticket number : ', titanic_nestor['Ticket'][30])

print('>> Unique Passenger id Values : ', len(
    titanic_nestor['PassengerId'].unique()))
print('>> Unique Passenger Name Values : ',
      len(titanic_nestor['Name'].unique()))
print('>> Unique Ticket Values : ', len(titanic_nestor['Ticket'].unique()))
print('>> Unique Cabin Values len : ', len(titanic_nestor['Cabin'].unique()))

print('\n\nUNIQUE VALUES SEX / PCLASS')
print(titanic_nestor['Sex'].unique())
print(titanic_nestor['Pclass'].unique())


"""
    DATA VISUALIZATION 
"""

print('\n\nBAR CHARTS FOR SURVIVALS COMPARISON')
pd.crosstab(titanic_nestor.Pclass, titanic_nestor.Survived,).plot(kind='bar')
plt.title('Survived by Class (Nestor)')
plt.xlabel('Class')
plt.ylabel('Frequency (Survived)')

pd.crosstab(titanic_nestor.Sex, titanic_nestor.Survived,).plot(kind='bar')
plt.title('Survived by Gender (Nestor)')
plt.xlabel('Gender')
plt.ylabel('Frequency (Survived)')

print('\n\nSCATTER MATRIX PLOT')
pd.plotting.scatter_matrix(titanic_nestor[[
    'Survived', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']], 
    alpha=0.2, 
    figsize=(10, 10))

# titanic_nestor['Survived'].hist()
#plt.title('Survived Histogram')

"""
    DATA TRANSFORMATION
"""

print('\n\nTRANSFORMING COLUMNS')
titanic_nestor_set = titanic_nestor.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

categorical_vars = ['Sex', 'Embarked']
for var in categorical_vars:
    categorical_var_dummy = pd.get_dummies(titanic_nestor_set[var], prefix=var)
    titanic_nestor_set = titanic_nestor_set.join(categorical_var_dummy)

titanic_nestor_set = titanic_nestor_set.drop(categorical_vars, axis=1)

titanic_nestor_set['Age'].fillna(value = titanic_nestor_set['Age'].mean(), inplace=True)
titanic_nestor_set = titanic_nestor_set.astype('float64')
print(titanic_nestor_set.info(show_counts=True))

def normalize_dataframe(dataframe):
    """
    This function normalizes the values for a dataframe with all numeric 
    columns

    Parameters
    ----------
    dataframe 
        All numeric dataframe

    Returns
    -------
    Normalized dataframe

    """
    for col in dataframe.columns.values:
        min_col = dataframe[col].min()
        max_col = dataframe[col].max()
        dataframe[col] = dataframe[col].apply(lambda x : ((x-min_col)/(max_col-min_col)))
    
    return dataframe

titanic_nestor_normal = normalize_dataframe(titanic_nestor_set)
print(titanic_nestor_normal.head(2))
titanic_nestor_normal.hist(figsize=(9,10))

titanic_nestor_normal[['Embarked_C','Embarked_Q','Embarked_S']].hist(figsize=(5,5))
