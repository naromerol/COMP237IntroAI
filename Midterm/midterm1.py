# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:07:37 2021

@author: Nestor Romero - 301133331
"""
import pandas as pd
import os
import matplotlib.pyplot as plt

path = "./"
filename = 'AA2.csv'
fullpath = os.path.join(path,filename)
df_nestor = pd.read_csv(fullpath,sep=',')

"""
Print the names of columns
Print the types of columns
Print the unique values in each column.
Print the statistics count, min, mean, standard deviation, 
1st quartile, median, 3rd quartile max of all the 
numeric columns(use one command).
Print the first three records.
Print a summary of all missing values in all columns (use one command).
Print the total number (count) of each unique value in the following
 categorical columns: Model Color
"""
col_names = df_nestor.columns.values
col_types = df_nestor.dtypes

print(col_names)
print(col_types)

for col in col_names :
    print(df_nestor[col].unique())

print(df_nestor.describe())
print(df_nestor.head(3))

print(df_nestor['model'].value_counts())
print(df_nestor['color'].value_counts())

"""
Visualize the data (10 marks)
Plot a histogram for the millage use 10 bins, name the x and y axis’
 appropriately, give the plot a title "firstname_millage".
Create a scatterplot showing "millage" versus "value", name the x and 
y axis’ appropriately, give the plot a title "firstname_millage_scatter".
Plot a "scatter matrix" showing the relationship between all columns 
of the dataset on the diagonal of the matrix plot the kernel density 
function.
Create a "boxplot" for the “value” column; name the x and y axis’ 
appropriately, give the plot a title "firstname_box_value"
Create a "bar chart" indicating stolen vehicles by type of vehicle 
i.e. for each Type plot two bars one in red color showing the total 
stolen and one blue showing the total not stolen., name the x and y axis’ 
appropriately, give the plot a title "firstname_stolen_by_type"
"""

plt.hist(df_nestor['millage'], bins=10)
plt.title('nestor_millage')
plt.xlabel('Millage')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df_nestor['millage'], df_nestor['value'])
plt.title('nestor_millage_scatter')
plt.xlabel('Millage')
plt.ylabel('Value')
plt.show()

plt.boxplot(df_nestor['value'])
plt.title('nestor_box_value')
plt.show()

pd.crosstab(df_nestor['type'],df_nestor['stolen']).plot(kind='bar')
plt.title('nestor_stolen_by_type')
plt.xlabel('Type')
plt.ylabel('Stolen')

"""
Pre-process the data (8 marks)
Remove (drop) properly the column with the most missing values. 
(hint: make sure you review and set the right arguments)  
Replace the missing values in the "millage" column with the median 
average of the column value.  
Check that there are no missing values.
Convert the all the categorical columns into numeric values and
 drop/delete the original columns. (hint:  use get dummies)      
Make sure your new data frame is completely numeric, name it 
df_firstname_numeric.
"""


df_nestor = df_nestor.drop('motor',1)
#df_nestor.describe()
df_nestor[df_nestor['millage'].isnull()] = 36
#print(df_nestor[df_nestor['millage'].isnull()])
#print(df_nestor.head(3))

cat_vars=['model','type','damage','color']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(df_nestor[var], prefix=var)
    df_nestor2=df_nestor.join(cat_list)
    df_nestor=df_nestor2

print(df_nestor.head(5))

data_vars=df_nestor.columns.values.tolist()
to_keep=[i for i in df_nestor if i not in cat_vars]
df_nestor_numeric = df_nestor[to_keep]

print(df_nestor_numeric.head(5))

Y=['stolen']
X=[i for i in df_nestor_numeric if i not in Y ]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)


from sklearn import linear_model
from sklearn import metrics
model_nestor = linear_model.LogisticRegression(solver='lbfgs')
model_nestor.fit(X_train, Y_train)

probs = model_nestor.predict_proba(X_test)
print(probs)
type(probs)
predicted = model_nestor.predict(X_test)
print (predicted)
print (metrics.accuracy_score(Y_test, predicted))
