#!/usr/bin/env python
# coding: utf-8

# In[7]:


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;}</style>')


# In[8]:


get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas')


# # Goalie Pull Bayes Optimize
# 
#  - Exploratory analysis

# ## Explore Parsed Goalie Pull Data

# In[9]:


import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import glob
from tqdm import tqdm_notebook
from colorama import Fore, Style


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelpad'] = 10
plt.style.use('ggplot')

def savefig(plt, name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)


# In[11]:


ls ../../data/processed/


# In[12]:


ls ../../data/processed/pkl


# In[13]:


def load_data():
    return pd.concat((
    pd.read_pickle('../../data/processed/pkl/20032004_goalie_pulls_2019-03-01.pkl'),
    pd.read_pickle('../../data/processed/pkl/20052006_goalie_pulls_2019-03-01.pkl'),
    pd.read_pickle('../../data/processed/pkl/20062007_goalie_pulls_2019-03-01.pkl'),
))

def clean_df(df):
    _df = df.copy()
    
    len_0 = _df.shape[0]
    print('Removing goal_for_time < 15 mins')
    _df = _df[~(_df.goal_for_time < datetime.timedelta(seconds=15*60))]
    print(f'Removed {len_0 - _df.shape[0]} total rows')
    
    if 'game_end_time' in df.columns:
        len_0 = _df.shape[0]
        print('Removing game_end_time < 15 mins')
        _df = _df[~(_df.game_end_time < datetime.timedelta(seconds=60*15))]
        print(f'Removed {len_0 - _df.shape[0]} total rows')
    
    return _df


# In[14]:


df = load_data()
df = clean_df(df)


# In[15]:


df.head()


# In[16]:


df.tail()


# In[12]:


df.describe().T


# In[13]:


df.dtypes


# In[14]:


df.isnull().sum() / df.shape[0]


# In[17]:


plt.ylabel('Frequency')
plt.yticks([])
df.date.hist(color='b', bins=100)
savefig(plt, 'date_distribution')


# In[18]:


df.columns


# In[19]:


col = ['pull_time']
(df[col].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, color='b')
plt.xlabel(f'Game clock (3rd period)')
plt.yticks([])
savefig(plt, 'goalie_pull_game_times_hist')


# In[20]:


cols = ['goal_for_time', 'goal_against_time']
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50,
               alpha=0.5,
               color=['green', 'red'])
plt.xlabel('Game clock (3rd period)')
plt.yticks([])
savefig(plt, '5_on_6_goals')


# In[21]:


cols = ['goal_for_time', 'goal_against_time']
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50,
               alpha=0.5,
               density='normed',
               color=['green', 'red'])
plt.xlabel('Game clock (3rd period)')
plt.yticks([])
savefig(plt, '5_on_6_goals_normed')


# In[22]:


print('Number of goals found:')
(~df[['goal_for_time', 'goal_against_time']].isnull()).sum()


# In[23]:


print('Total goals found:')
(~df[['goal_for_time', 'goal_against_time']].isnull()).sum().sum()


# We want to model the time between goalie pull and goal (i.e. the timedelta).

# In[24]:


cols = ['game_end_timedelta', 'goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               color=['blue', 'red','green'])
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_goalie_pull_outcomes')


# In[25]:


cols = ['goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               color=['red', 'green',])
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_goalie_pull_goal_timedeltas')


# In[26]:


cols = ['goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               density='normed',
               color=['red', 'green',])
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_goalie_pull_goal_timedeltas_normed')


# The mean/median number of seconds until a goal (after pulling the goalie)

# In[27]:


(df[cols].astype('timedelta64[s]')).mean()


# In[28]:


(df[cols].astype('timedelta64[s]')).median()


# In[29]:


(df['game_end_timedelta'].astype('timedelta64[s]') / 60).plot.hist(bins=50, color='b')
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_game_end_timedeltas')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Rough work

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Bugs

# In[247]:


df = load_data()


# Any non 3rd period pulls?

# In[248]:


df[df['pull_period'] != 3]


# Any bugs not in the last 15 minutes?

# In[120]:


mask = df.pull_time < datetime.timedelta(seconds=15*60)
df[mask].shape[0]


# I noticed some goal for timedelats less than 0.. which makes no sense. I'll have to look into that

# In[121]:


mask = df.goal_for_time < datetime.timedelta(seconds=15*60)
df[mask]


# This game is weird... http://www.nhl.com/scores/htmlreports/20052006/PL020591.HTM
# 
# We'll drop this point before modeling.

# In[150]:


df[df.goal_for_timedelta < datetime.timedelta(0)]


# http://www.nhl.com/scores/htmlreports/20032004/PL020907.HTM

# In[35]:


df.game_end_time.astype('timedelta64[s]').plot.hist()


# Games should end at 20 mins. Let's throw out the early times (this must be overtime or something).

# In[40]:


mask = df.game_end_time < datetime.timedelta(seconds=60*20)
mask.sum(), df.shape[0]


# In[41]:


mask = df.game_end_time < datetime.timedelta(seconds=60*15)
mask.sum(), df.shape[0]


# Obviously the game will end at 20:00, this column corresponds to the last row parsed.

# In[ ]:




