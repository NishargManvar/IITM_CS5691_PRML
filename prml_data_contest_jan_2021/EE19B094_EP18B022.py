#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import IPython.display as dispaly
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import xgboost as xgb
import sklearn
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from __future__ import division


# In[2]:


train_dataset = pd.read_csv(r"C:\Users\Iball\Downloads\PRML MKN Jan-21 Dataset\PRML MKN Jan-21 Dataset\train.csv")
song_description = pd.read_csv(r'C:\Users\Iball\Downloads\PRML MKN Jan-21 Dataset\PRML MKN Jan-21 Dataset\songs.csv')
song_labels = pd.read_csv(r'C:\Users\Iball\Downloads\PRML MKN Jan-21 Dataset\PRML MKN Jan-21 Dataset\song_labels.csv')
save_for_later = pd.read_csv(r'C:\Users\Iball\Downloads\PRML MKN Jan-21 Dataset\PRML MKN Jan-21 Dataset\save_for_later.csv')
test_dataset = pd.read_csv(r'C:\Users\Iball\Downloads\PRML MKN Jan-21 Dataset\PRML MKN Jan-21 Dataset\test.csv')


# In[3]:


song_description['language'] = song_description['language'].replace(['eng','en-US','en-GB','en-CA'],'eng')


count = 0
language_number_dict = {}
for language in song_description.language.unique():
  language_number_dict[language] = count
  count += 1


# In[4]:


comments_dict = {}
language_dict = {}
for index,row in song_description.iterrows():
  song_id = row['song_id']
  comment = row['number_of_comments']
  comments_dict[song_id] = comment

  language = row['language']
  language_dict[song_id] = language_number_dict[language]

for index,row in train_dataset.iterrows():
  song_id = row['song_id']
  if song_id not in comments_dict:
    comments_dict[song_id] = 0
    language_dict[song_id] = language_number_dict[np.NaN]


# In[5]:


#List of all unique labels
labels = song_labels.label_id.unique()
 
#Calculating total count of each label
label_counts = []
for label in labels:
  rows = song_labels.loc[song_labels['label_id'] == label]
  values = rows['count'].tolist()
  count = sum(values)
  label_counts.append([label,count])
 
#Sort labels based on their count
label_counts_sorted = sorted(label_counts,key=lambda l:l[1], reverse=True)
#print(label_counts_sorted)


# In[6]:


#Number of labels with top counts to be used i.e. 
use_labels = 300
 
#Slicing the labels to get use_labels number of labels
label_counts_sorted_trimmed = label_counts_sorted[1:use_labels+1]
#print(len(label_counts_sorted_trimmed))

#Create a new data frame consisting of only selected labels and discarding rest
updated_song_labels = pd.DataFrame()
for label in label_counts_sorted_trimmed:
  label_name = label[0]
  rows = song_labels.loc[song_labels['label_id'] == label_name]
  updated_song_labels = updated_song_labels.append(rows,ignore_index=True)

labels_dict = {}
count = 0
for song_label in label_counts_sorted_trimmed:
  labels_dict[song_label[0]] = count
  count += 1


# In[7]:



song_final = pd.merge(song_description,updated_song_labels)
song_final.head(n=200)

song_label_dict={}
for index,row in song_final.iterrows():
  song_id = row['song_id']
  label_id = row['label_id']
  count = row['count']

  if song_id in song_label_dict:
    song_label_dict[song_id][labels_dict[label_id]] = count
  else:
    song_label_dict[song_id] = [0]*use_labels
    song_label_dict[song_id][labels_dict[label_id]] = count

count = 0
for index,row in train_dataset.iterrows():
  song_id = row['song_id']
  if song_id not in song_label_dict:
    count += 1
    song_label_dict[song_id] = [0]*use_labels


# In[8]:


song_label_dict_fraction = {}
for song in song_label_dict:
  label_list = song_label_dict[song]
  sum_value = sum(label_list)
  if sum_value != 0:
    song_label_dict_fraction[song] = [x/sum_value for x in label_list]
  else:
    song_label_dict_fraction[song] = label_list


# In[9]:


save_for_later_dict = {}
save_for_later_count = {}
for index,rows in save_for_later.iterrows():
  customer_id = rows['customer_id']
  song_id = rows['song_id']
  labels = song_label_dict_fraction[song_id]
  
  if customer_id in save_for_later_dict:
    old_count = save_for_later_count[customer_id]
    save_for_later_dict[customer_id] = [(g*old_count + h)/ (old_count+1) for g, h in zip(save_for_later_dict[customer_id], labels)]
    save_for_later_count[customer_id] += 1
  else:
    save_for_later_dict[customer_id] = labels
    save_for_later_count[customer_id] = 1

for index,rows in train_dataset.iterrows():
  customer_id = rows['customer_id']
  
  if customer_id not in save_for_later_dict:
    save_for_later_dict[customer_id] = [0]*use_labels


# 
# 
# ---
# 
# ** TESTING MODEL BY SPLITTING DATA INTO TRAIN AND TEST **

# In[10]:


# customers = train_dataset.customer_id.unique()
# customer_dict = {}
# count = 0
# for customer in customers:
#   customer_dict[customer] = count
#   count += 1 


# x_train1 = np.array([customer_dict[x] for x in train_dataset['customer_id'] ])
# x_train2 = np.array([ np.array(save_for_later_dict[customer])-np.array(song_label_dict_fraction[song]) for customer,song in zip(train_dataset['customer_id'],train_dataset['song_id']) ])
# #a = np.array([[0, 1], [2, 2], [4, 3]])
# #x_train2 = (x_train2 == x_train2.max(axis=1)[:,None]).astype(int)
# #x_train3 = [ comments_dict[song] for song in train_dataset['song_id'] ]
# x_train4 = [ language_dict[song] for song in train_dataset['song_id'] ]
# x_train = np.column_stack((x_train1,x_train2,x_train4))
# #x_train = np.tile(x_train,(2,1))
# print(x_train.shape)

# y_train = np.array(train_dataset['score'])
# y_train = y_train.T
# #y_train = np.tile(y_train,2)
# print(y_train.shape)
# #print(max(y_train))
# #print(y_train)
# #x_train , y_train = shuffle(x_train, y_train , random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(x_train, y_train , test_size=0.15)

# print(X_train)


# # x_test1 = [customer_dict[x] for x in test_dataset['customer_id'] ]
# # x_test2 = test_dataset['song_id']
# # x_test = np.vstack((x_test1,x_test2)).T


# In[11]:


# gbm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.08, gamma=0, subsample=0.75,
# colsample_bytree=1, max_depth=12).fit(X_train, y_train)
# y_pred = gbm.predict(X_test)


# In[12]:


# predictions = [value for value in y_pred]

# accuracy = sklearn.metrics.mean_squared_error(y_test, predictions)
# print("Accuracy: %.2f" % (accuracy))


# ----------------------------------------------------------
# **      ACTUALLY PREDICTING THE REVIEWS FOR TEST DATASET**

# In[13]:


customers = train_dataset.customer_id.unique()
customer_dict = {}
count = 0
for customer in customers:
  customer_dict[customer] = count
  count += 1
    
x_train_final1 = np.array([customer_dict[x] for x in train_dataset['customer_id'] ])
x_train_final2 = np.array([ np.array(save_for_later_dict[customer])-np.array(song_label_dict_fraction[song]) for customer,song in zip(train_dataset['customer_id'],train_dataset['song_id']) ])
x_train_final4 = [ language_dict[song] for song in train_dataset['song_id'] ]
x_train_final = np.column_stack((x_train_final1,x_train_final2,x_train_final4))

y_train_final = np.array(train_dataset['score'])
#y_train_final = y_train.T

x_test_final1 = np.array([customer_dict[x] for x in test_dataset['customer_id'] ])
x_test_final2 = np.array([ np.array(save_for_later_dict[customer])-np.array(song_label_dict_fraction[song]) for customer,song in zip(test_dataset['customer_id'],test_dataset['song_id']) ])
x_test_final4 = [ language_dict[song] for song in test_dataset['song_id'] ]
x_test_final = np.column_stack((x_test_final1,x_test_final2,x_test_final4))


# In[15]:


gbm = xgb.XGBRegressor(n_estimators=400, learning_rate=0.08, gamma=0, subsample=0.75,
colsample_bytree=1, max_depth=15).fit(x_train_final, y_train_final)
y_pred = gbm.predict(x_test_final)

predictions = [value for value in y_pred]

test_row_id = [int(x) for x in range(len(x_test_final))]
final_predictions = np.vstack((test_row_id,predictions)).T


# In[20]:


predictions = np.array(predictions)


# In[21]:


submission = pd.DataFrame(final_predictions, columns = ['test_row_id', 'score'], dtype=object)


# In[22]:


submission = submission.astype({'test_row_id': 'int'})


# In[23]:


submission.to_csv(r"C:\Users\Iball\Downloads\PRML MKN Jan-21 Dataset\PRML MKN Jan-21 Dataset/submission_testing_2.csv", index = False)


# In[ ]:




