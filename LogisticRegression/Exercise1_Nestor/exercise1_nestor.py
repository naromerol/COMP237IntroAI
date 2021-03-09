# -*- coding: utf-8 -*-
"""
Logistic Regression Assignment - Exercise 1

@author: Nestor Romero leon
Student Id - 301133331 
"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
    INITIAL DATA ANALYSIS
"""
titanic_nestor = pd.read_csv('titanic.csv')
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

titanic_nestor_set['Age'].fillna(
    value=titanic_nestor_set['Age'].mean(), inplace=True)
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
        dataframe[col] = dataframe[col].apply(
            lambda x: ((x-min_col)/(max_col-min_col)))

    return dataframe


titanic_nestor_normal = normalize_dataframe(titanic_nestor_set)
print(titanic_nestor_normal.head(2))
titanic_nestor_normal.hist(figsize=(9, 10))

titanic_nestor_normal[['Embarked_C', 'Embarked_Q',
                       'Embarked_S']].hist(figsize=(5, 5))

y_nestor = titanic_nestor_normal['Survived']
x_nestor = titanic_nestor_normal.drop('Survived', axis=1)

# Last two digits of student id as seed
x_train_nestor, x_test_nestor, y_train_nestor, y_test_nestor = train_test_split(
    x_nestor, y_nestor, test_size=0.3, random_state=31)

"""
    LOGISTIC REGRESSION MODEL
"""

# Fit Logistic Regression Model
nestor_model = linear_model.LogisticRegression(solver='lbfgs')
nestor_model.fit(x_train_nestor, y_train_nestor)

print('\n\nDISPLAY MODEL COEFFICIENTS')
coef_df = pd.DataFrame(
    zip(x_train_nestor.columns, np.transpose(nestor_model.coef_)))
print(coef_df)


print('\n\nDISPLAY CROSS-VALIDATION RESULTS')
print('Test Size - Min Mean Max Range')
for ts in np.arange(0.10, 0.55, 0.05):
    x_train_cross, x_test_cross, y_train_cross, y_test_cross = train_test_split(
        x_nestor, y_nestor, test_size=ts, random_state=31)

    scores_cross = cross_val_score(linear_model.LogisticRegression(
        solver='lbfgs'), x_train_cross, y_train_cross, scoring='accuracy', cv=10)

    score_min = scores_cross.min()
    score_max = scores_cross.max()
    score_mean = scores_cross.mean()
    line = "Test Size: {0:.4f}  || Metrics: {1:.4f}   {2:.4f}   {3:.4f}   {4:.4f}".format(
        ts, score_min, score_mean, score_max, score_max - score_min)
    print(line)


"""
    MODEL TESTING
"""
# Last two digits of student id as seed
x_train_nestor, x_test_nestor, y_train_nestor, y_test_nestor = train_test_split(
    x_nestor, y_nestor, test_size=0.3, random_state=31)
# Fit Logistic Regression Model
nestor_model = linear_model.LogisticRegression(solver='lbfgs')
nestor_model.fit(x_train_nestor, y_train_nestor)

print("\n\n*** METRICS 0.5 THRESHOLD ***")

y_pred_nestor = nestor_model.predict_proba(x_test_nestor)
y_pred_nestor_flag = y_pred_nestor[:, 1] > 0.5
y_predicted = y_pred_nestor_flag.astype(int)
y_predicted = np.array(y_predicted)

cmatrix = confusion_matrix(y_test_nestor.values, y_predicted)
ascore = accuracy_score(y_test_nestor.values, y_predicted)
creport = classification_report(y_test_nestor.values, y_predicted)


print("\n>> CONFUSION MATRIX\n", cmatrix)
print("\n>> ACCURACY SCORE\n", ascore)
print("\n>> CLASSIFICATION REPORT\n", creport)


print("\n\n*** METRICS 0.75 THRESHOLD ***")

# y_pred_nestor = nestor_model.predict_proba(x_test_nestor)
y_pred_nestor_flag = y_pred_nestor[:, 1] > 0.75
y_predicted2 = y_pred_nestor_flag.astype(int)
y_predicted2 = np.array(y_predicted2)

cmatrix2 = confusion_matrix(y_test_nestor.values, y_predicted2)

ascore2 = accuracy_score(y_test_nestor.values, y_predicted2)
creport2 = classification_report(y_test_nestor.values, y_predicted2)

print("\n>> CONFUSION MATRIX\n", cmatrix2)
print("\n>> ACCURACY SCORE\n", ascore2)
print("\n>> CLASSIFICATION REPORT\n", creport2)

tn, fp, fn, tp = confusion_matrix(y_test_nestor.values, y_predicted2).ravel()