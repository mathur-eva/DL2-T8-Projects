#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow.keras as tf


# In[10]:


ratings_df = pd.read_csv("C:/Users/LENOVO/Documents/Books/Newfolder/ratings.csv")
books_df = pd.read_csv("C:/Users/LENOVO/Documents/Books/Newfolder/books.csv")


# In[11]:


ratings_df.head()


# In[12]:


books_df.head()


# In[13]:


print(ratings_df.shape)
print(ratings_df.user_id.nunique())
print(ratings_df.book_id.nunique())
ratings_df.isna().sum()


# In[14]:




from sklearn.model_selection import train_test_split
Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")


# In[15]:


nbook_id = ratings_df.book_id.nunique()
nuser_id = ratings_df.user_id.nunique()


# In[16]:


input_books = tf.layers.Input(shape=[1])
embed_books = tf.layers.Embedding(nbook_id + 1,15)(input_books)
books_out = tf.layers.Flatten()(embed_books)

#user input network
input_users = tf.layers.Input(shape=[1])
embed_users = tf.layers.Embedding(nuser_id + 1,15)(input_users)
users_out = tf.layers.Flatten()(embed_users)

conc_layer = tf.layers.Concatenate()([books_out, users_out])
x = tf.layers.Dense(128, activation='relu')(conc_layer)
x_out = x = tf.layers.Dense(1, activation='relu')(x)
model = tf.Model([input_books, input_users], x_out)


# In[17]:


opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error')
model.summary()


# In[18]:




hist = model.fit([Xtrain.book_id, Xtrain.user_id], Xtrain.rating, 
                 batch_size=128, 
                 epochs=5, 
                 verbose=1,
                 validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating))


# In[19]:


#Data Cleaning
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()


# In[20]:




model.save('model')
model.summary()


# In[21]:


book_em = model.get_layer('embedding')
book_em_weights = book_em.get_weights()[0]
book_em_weights.shape


# In[22]:




books_df_copy = books_df.copy()
books_df_copy = books_df_copy.set_index("book_id")


# In[23]:




b_id =list(ratings_df.book_id.unique())
b_id.remove(10000)
dict_map = {}
for i in b_id:
    dict_map[i] = books_df_copy.iloc[i]['title']
    
out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')
for i in b_id:
    book = dict_map[i]
    embeddings = book_em_weights[i]
    out_m.write('book')
    out_v.write('/t'.join([str(x) for x in embeddings]) + '/n')
    
out_v.close()
out_m.close()


# In[24]:


#Recommending Books for unique Id 250


# In[25]:


book_arr = np.array(b_id) #get all book IDs
user = np.array([250 for i in range(len(b_id))])
pred = model.predict([book_arr, user])
pred


# In[26]:


pred = pred.reshape(-1) #reshape to single dimension
pred_ids = (-pred).argsort()[0:5]
pred_ids


# In[30]:


books_df.iloc[pred_ids]


# In[ ]:


#name of Team Member
Aman S. Kankariya
Amatya Trivedi
Divya Sharma
Eva Mathur
Himanshu Mishra

