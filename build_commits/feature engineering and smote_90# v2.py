#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


data_orig = pd.read_csv("dataset/train.csv")
data_orig.head()


# In[4]:


data_orig['Action'].value_counts()


# In[5]:


data = data_orig.iloc[:,1:]
data.head()


# In[6]:


label_encoder = LabelEncoder()
data.iloc[:,42] = label_encoder.fit_transform(data.iloc[:,42]).astype('int')
data.head()


# In[7]:


data.columns


# In[8]:


df_sub_lh = data[['lhx', 'lhy', 'lhz']]
df_sub_rh = data[['rhx', 'rhy', 'rhz']]
df_sub_hd = data[['hx', 'hy', 'hz']]
df_sub_lw = data[['lwx', 'lwy', 'lwz']]
df_sub_rw = data[['rwx', 'rwy', 'rwz']]
df_sub_sp = data[['sx', 'sy', 'sz']]

df_sub_vlh = data[['vlhx', 'vlhy', 'vlhz']]
df_sub_vrh = data[['vrhx', 'vrhy', 'vrhz']]
df_sub_vlw = data[['vlwx', 'vlwy', 'vlwz']]
df_sub_vrw = data[['vrwx', 'vrwy', 'vrwz']]
df_sub_alh = data[['alhx', 'alhy', 'alhz']]
df_sub_arh = data[['arhx', 'arhy', 'arhz']]
df_sub_alw = data[['alwx', 'alwy', 'alwz']]
df_sub_arw = data[['arwx', 'arwy', 'arwz']]


# In[9]:


cos_lh_rh = cosine_similarity(df_sub_lh, df_sub_rh)
cos_hd_lh = cosine_similarity(df_sub_hd, df_sub_lh)
cos_hd_rh = cosine_similarity(df_sub_hd, df_sub_rh)
cos_hd_sp = cosine_similarity(df_sub_hd, df_sub_sp)
cos_lh_lw = cosine_similarity(df_sub_lh, df_sub_lw)
cos_rh_rw = cosine_similarity(df_sub_rh, df_sub_rw)

cos_vlh_vrh = cosine_similarity(df_sub_vlh, df_sub_vrh)
cos_vlw_vrw = cosine_similarity(df_sub_vlw, df_sub_vrw)
cos_alh_arh = cosine_similarity(df_sub_alh, df_sub_arh)
cos_alw_arw = cosine_similarity(df_sub_alw, df_sub_arw)

cos_lh_vlh = cosine_similarity(df_sub_lh, df_sub_vlh)
cos_rh_vrh = cosine_similarity(df_sub_rh, df_sub_vrh)
cos_lw_vlw = cosine_similarity(df_sub_lw, df_sub_vlw)
cos_rw_vrw = cosine_similarity(df_sub_rw, df_sub_vrw)


# In[10]:


df_lh_rh = pd.DataFrame(np.diagonal(cos_lh_rh)).rename(columns ={0:'df_lh_rh'})
df_hd_lh = pd.DataFrame(np.diagonal(cos_hd_lh)).rename(columns ={0:'df_hd_lh'})
df_hd_rh = pd.DataFrame(np.diagonal(cos_hd_rh)).rename(columns ={0:'df_hd_rh'})
df_hd_sp = pd.DataFrame(np.diagonal(cos_hd_sp)).rename(columns ={0:'df_hd_sp'})
df_lh_lw = pd.DataFrame(np.diagonal(cos_lh_lw)).rename(columns ={0:'df_lh_lw'})
df_rh_rw = pd.DataFrame(np.diagonal(cos_rh_rw)).rename(columns ={0:'df_rh_rw'})

df_vlh_vrh = pd.DataFrame(np.diagonal(cos_vlh_vrh)).rename(columns ={0:'df_vlh_vrh'})
df_vlw_vrw = pd.DataFrame(np.diagonal(cos_vlw_vrw)).rename(columns ={0:'df_vlw_vrw'})
df_alh_arh = pd.DataFrame(np.diagonal(cos_alh_arh)).rename(columns ={0:'df_alh_arh'})
df_alw_arw = pd.DataFrame(np.diagonal(cos_alw_arw)).rename(columns ={0:'df_alw_arw'})

df_lh_vlh = pd.DataFrame(np.diagonal(cos_lh_vlh)).rename(columns ={0:'df_lh_vlh'})
df_rh_vrh = pd.DataFrame(np.diagonal(cos_rh_vrh)).rename(columns ={0:'df_rh_vrh'})
df_lw_vlw = pd.DataFrame(np.diagonal(cos_lw_vlw)).rename(columns ={0:'df_lw_vlw'})
df_rw_vrw = pd.DataFrame(np.diagonal(cos_rw_vrw)).rename(columns ={0:'df_rw_vrw'})


# In[11]:


df_new_features = pd.concat([df_lh_rh, df_hd_lh, df_hd_rh, df_hd_sp, df_lh_lw, df_rh_rw, df_vlh_vrh, df_vlw_vrw, df_alh_arh, df_alw_arw, df_lh_vlh, df_rh_vrh, df_lw_vlw, df_rw_vrw], axis=1)


# In[12]:


df_new_features.head()


# In[13]:


data_appended = pd.DataFrame()
data_appended = pd.concat([df_new_features, data], sort = False, axis = 1)


# In[14]:


data_appended


# In[15]:


data_appended.columns


# In[16]:


data_appended['Action'].value_counts()


# In[17]:


y = data_appended[["Action"]].values
X = data_appended.drop(["Action"], axis=1)


# In[18]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_smote, y_train_smote = sm.fit_sample(X, y.ravel())


# In[19]:


X_train_smote.shape


# In[20]:


from collections import Counter
Counter(y_train_smote)


# In[21]:


sc = StandardScaler()
sc.fit(X_train_smote) 
X_scaled = sc.transform(X_train_smote) 
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_train_smote, test_size=.2, random_state=1)

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)


# In[24]:


rf_tuned = rf_random.best_estimator_
cv_results = cross_validate(rf_tuned, X_scaled, y_train_smote, cv=10)
cv_results['test_score']


# In[25]:


preds = rf_tuned.predict(x_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


recr_new = map(int, list(map(lambda string: string[0][2:], data_recr)))

