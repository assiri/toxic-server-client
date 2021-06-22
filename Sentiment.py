#!/usr/bin/env python
# coding: utf-8

# # 1. Install and Import Dependencies

# In[1]:


#!pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


# In[2]:


#!pip install transformers requests beautifulsoup4 pandas numpy


# In[3]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


# # 2. Instantiate Model

# In[4]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# # 3. Encode and Calculate Sentiment

# In[5]:


tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')


# In[6]:


result = model(tokens)


# In[7]:


result.logits


# In[8]:


int(torch.argmax(result.logits))+1


# # 4. Collect Reviews

# In[9]:


r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]


# In[10]:


reviews


# # 5. Load Reviews into DataFrame and Score

# In[11]:


import numpy as np
import pandas as pd


# In[12]:


df = pd.DataFrame(np.array(reviews), columns=['review'])


# In[13]:


df['review'].iloc[0]


# In[14]:


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[15]:


sentiment_score(df['review'].iloc[1])


# In[16]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[17]:


df


# In[18]:


df['review'].iloc[3]


# In[19]:


df.to_csv("reviews.csv")


# In[ ]:




