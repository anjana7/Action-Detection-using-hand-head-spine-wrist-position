#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns


# In[2]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


# In[3]:


data_orig = pd.read_csv("dataset/train.csv")


# In[4]:


data_orig.head()


# In[5]:


data_orig['Action'].value_counts()


# In[6]:


data = data_orig.iloc[:,1:]


# In[7]:


data.head()


# In[8]:


label_encoder = LabelEncoder()
data.iloc[:,42] = label_encoder.fit_transform(data.iloc[:,42]).astype('int')


# In[9]:


data.head()


# In[10]:


data['Action'].value_counts()


# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[12]:


X = data.iloc[:,:42]
y = data[["Action"]]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[14]:


preds = model.predict(X_test)

print(confusion_matrix(y_test, preds))


# In[15]:


print(classification_report(y_test, preds))


# In[16]:



corr = X.corr()
sns.heatmap(corr)


# In[17]:


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.95:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
X_new = X[selected_columns]


# In[18]:


X_new.columns


# In[19]:


len(X_new.columns)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[21]:


preds = model.predict(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))


# In[22]:


data['Action'].value_counts()


# In[23]:


y_train['Action'].value_counts()


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)


# In[26]:


preds = neigh.predict(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))


# In[27]:


df = data.loc[data['Action'] == 4]
df2 = data.loc[data['Action'] == 3]


# In[28]:


data_new = pd.concat([data, df, df2])


# In[29]:


data_new


# In[30]:


data_new["Action"].value_counts()


# In[31]:


data_new.columns


# In[32]:


len(data_new.columns)


# In[33]:


corr = data_new.corr()
sns.heatmap(corr)


# In[34]:


columns = np.full((corr.shape[0],), True, dtype=bool)
print(len(columns))

for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
print("new col",columns)

selected_columns = data_new.columns[columns]
print(len(selected_columns))
print(selected_columns[1:].values)
X_features_transformed = data_new[selected_columns]


# In[35]:


X_features_transformed.head()


# In[36]:


X_features_transformed["Action"].value_counts()


# In[37]:


#import statsmodels.formula.api as sm
import statsmodels.api as sm

def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns


SL = 0.05
selected_columns = selected_columns[:-1].values
data_modeled, selected_columns = backwardElimination(X_features_transformed.iloc[:,:29].values, X_features_transformed.iloc[:,29].values, SL, selected_columns)


# In[38]:


selected_columns


# In[39]:


result = X_features_transformed[['Action']]
train = pd.DataFrame(data = data_modeled, columns = selected_columns)


# In[40]:


result['Action'].value_counts()


# In[41]:


from pandas_profiling import ProfileReport
train.profile_report()


# In[43]:


result.profile_report()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train.values, result.values, test_size = 0.2)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=12),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

figure = plt.figure(figsize=(27, 9))

for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    print("----------------",name,"----------------")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("score : ", score)
    print("confusion_matrix", confusion_matrix(y_test, preds))
    print("classification_report")
    print(classification_report(y_test, preds))
    
    


# In[ ]:


data_new.columns
data_new_train = data_new.iloc[:,:-1]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data_new_train.values, result.values, test_size = 0.2)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=12),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

figure = plt.figure(figsize=(27, 9))

for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    print("----------------",name,"----------------")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("score : ", score)
    print("confusion_matrix", confusion_matrix(y_test, preds))
    print("classification_report")
    print(classification_report(y_test, preds))
    
    


# In[ ]:


X_new_train = X_features_transformed.iloc[:,:-1]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_new_train.values, result.values, test_size = 0.2)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=12),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

figure = plt.figure(figsize=(27, 9))

for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    print("----------------",name,"----------------")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("score : ", score)
    print("confusion_matrix", confusion_matrix(y_test, preds))
    print("classification_report")
    print(classification_report(y_test, preds))
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


selected_columns = data_new.columns
selected_columns = selected_columns[:-1].values
selected_columns


# In[39]:


#import statsmodels.formula.api as sm
import statsmodels.api as sm

def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns


SL = 0.09
#selected_columns = selected_columns[:-1].values
data_modeled, selected_columns = backwardElimination(data_new.iloc[:,:-1].values, data_new.iloc[:,-1].values, SL, selected_columns)


# In[40]:


len(selected_columns)


# In[41]:


selected_columns


# In[42]:


data_modeled.shape


# In[47]:


result = data_new[['Action']]
X_data_modeled = pd.DataFrame(data = data_modeled, columns = selected_columns)


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(X_data_modeled.values, result.values, test_size = 0.2)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=12),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

figure = plt.figure(figsize=(27, 9))

for name, clf in zip(names, classifiers):
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    print("----------------",name,"----------------")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("score : ", score)
    print("confusion_matrix", confusion_matrix(y_test, preds))
    print("classification_report")
    print(classification_report(y_test, preds))
    
    


# In[ ]:




