#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


# In[4]:


iris=load_iris()


# In[11]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
X=iris.data
df


# In[12]:


X


# In[14]:


y=iris.target
y


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[18]:


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(X_train,y_train)


# In[19]:


reg.coef_


# In[21]:


reg.intercept_


# In[22]:


reg.score(X_test,y_test)


# In[24]:


y[120]


# In[29]:


reg.predict(X_test)


# In[30]:


y_test


# In[32]:


reg.predict([iris.data[120]])


# In[33]:


y_predicted=reg.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm


# In[34]:


import seaborn as sns
sns.heatmap(cm,annot=True)
plt.show()


# In[39]:


import pickle as pkl
with open('iris.pkl','wb')as f:
    pickle.dump(reg,f)


# In[40]:


with open('iris.pkl','rb')as f2:
    mp=pickle.load(f2)


# In[42]:


mp.predict([[6.7,3.0,5.2,2.3]])

