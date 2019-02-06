# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 22:25:12 2018

@author: pdwarkanath
"""



import pandas as pd
import seaborn as sns
import re
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Extract titles from names






def getTitles(names):
    titleRegex = re.compile(r',.*\.')    
    title = []
    for str in names:
        titlePat = re.search(titleRegex,str)
        x = titlePat.group()
        x = x[2:len(x)-1]
        title.append(x)
    return title

title = getTitles(train['Name'])

print(list(set(title)))

def getCleanTitles(title):
    for i in range(len(title)):
        if title[i] in ['Capt','Don', 'Col', 'Major','Rev','Sir' ]:
            title[i] = 'Mr'
        elif title[i] in ['Mlle','Mme','Ms','the Countess','Mrs. Martin (Elizabeth L', 'Lady']:
            title[i] = 'Mrs'
        elif title[i] =='Ms':
            title[i] = 'Miss'
        elif title[i] in ['Dr', 'Jonkheer']:
            title[i] = 'Other'
    return title
    

train['Title'] = getCleanTitles(title)

# Plot survival against features

sns.barplot(x='Pclass', y='Survived',data = train)
sns.barplot(x='Sex', y='Survived',data = train)
sns.barplot(x='Title', y='Survived',data = train)
sns.barplot(x='SibSp', y='Survived',data = train)
sns.barplot(x='Parch', y='Survived',data = train)

# Extract X and y i.e. features and responses from data


features = ['Pclass','Sex', 'Title', 'SibSp', 'Parch']
ytrain = train['Survived']
Xtrain = train.loc[:,features]

# Encoding categorical variables as numbers

def convertCatValToNum(catVal):
    le = preprocessing.LabelEncoder()
    le.fit(catVal)
    catVal = le.transform(catVal)
    return catVal

Xtrain['Sex'] = convertCatValToNum(Xtrain['Sex'])
Xtrain['Title'] = convertCatValToNum(Xtrain['Title'])



print(Xtrain.head())


# Clean test data

title = getTitles(test['Name'])
title = getCleanTitles(title)

test['Title'] = title

Xtest = train.loc[:,features]

Xtest['Sex'] = convertCatValToNum(Xtest['Sex'])
Xtest['Title'] = convertCatValToNum(Xtest['Title'])




# Logistic Regression

logreg = LogisticRegression()
logreg.fit(Xtrain, ytrain)

ypred = logreg.predict(Xtest)
ypred = list(ypred)
passengerIds = list(test.loc[:,'PassengerId'])


# Write Solution file

f = open('preds.csv','w+')
f.write('PassengerID,Survived\n')
for i in range(len(passengerIds)):
    f.write("{},{}\n".format(passengerIds[i],ypred[i]))
f.close()




