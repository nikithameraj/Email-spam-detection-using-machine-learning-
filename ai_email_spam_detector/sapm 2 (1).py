#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts


# In[2]:


# reading the dataset
data=pd.read_csv('spam.csv')


# In[3]:


# head values of the dataset
data.head()


# In[4]:


# describing the datset
data.describe()


# In[5]:


# shape of the dataset 
data.shape


# In[6]:


# column names
data.columns


# In[7]:


# count of unique values in column name Label
data['Label'].value_counts()


# In[8]:


# removing the html tags
def clean_html(text):
    clean=re.compile('<.*?>')
    cleantext=re.sub(clean,'',text)
    return cleantext
    
# first round of cleaning
def clean_text1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

# second round of cleaning
def clean_text2(text):
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    return text
    
cleaned_html=lambda x:clean_html(x)
cleaned1=lambda x:clean_text1(x)
cleaned2=lambda x:clean_text2(x)

data['EmailText']=pd.DataFrame(data.EmailText.apply(cleaned_html))
data['EmailText']=pd.DataFrame(data.EmailText.apply(cleaned1))
data['EmailText']=pd.DataFrame(data.EmailText.apply(cleaned2))


# In[9]:


x=data.iloc[0:,1].values
y=data.iloc[0:,0].values


# In[10]:


xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.20,random_state=42)


# In[11]:


cv = CountVectorizer()  
xtrain = cv.fit_transform(xtrain)


# In[12]:


tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

classifier = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)
#classifier = svm.SVC(kernel='rbf',gamma=1e-3,C=100)
classifier.fit(xtrain,ytrain)

# printing the best model
classifier.best_params_


# In[13]:


xtest = cv.transform(xtest)
ypred = classifier.predict(xtest)


# In[14]:


# model score
accuracy_score(ytest,ypred)


# In[15]:


# confusion matrix
A=confusion_matrix(ytest,ypred)
print(A)


# In[16]:


recall=A[0][0]/(A[0][0]+A[1][0])
precision=A[0][0]/(A[0][0]+A[0][1])
F1=2*recall*precision/(recall+precision)
print(F1)


# In[17]:


# saving the model to disk
import pickle
pickle.dump(classifier, open('model.pkl','wb'))
pickle.dump(cv,open('cv.pkl','wb'))


# In[18]:


test="hello. You have won 300000000$. Give your contact details."
test=clean_html(test)
test=clean_text1(test)
test=clean_text2(test)
test=cv.transform([test])
labpred=classifier.predict(test)
print(labpred[0])


# In[ ]:





# In[ ]:




