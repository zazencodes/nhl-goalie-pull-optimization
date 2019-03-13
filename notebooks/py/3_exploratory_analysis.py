
# coding: utf-8

# In[1]:


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;}</style>')


# In[2]:


get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas')


# # Goalie Pull Bayes Optimize
# 
#  - Exploratory analysis

# ## Explore Parsed Goalie Pull Data

# In[3]:


import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import glob
from tqdm import tqdm_notebook
from colorama import Fore, Style


# In[17]:


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


# In[5]:


ls ../../data/processed/


# In[6]:


ls ../../data/processed/pkl


# In[7]:


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


# In[8]:


df = load_data()
df = clean_df(df)


# In[44]:


# Label the outcomes
df['label'] = ''
label_masks = {
    'goal_for': ~(df.goal_for_time.isnull()),
    'goal_against': ~(df.goal_against_time.isnull()),
    'no_goals': ~(df.game_end_timedelta.isnull()),   
}
for label, mask in label_masks.items():
    df.loc[mask, 'label'] = label
df.loc[df.label == '', 'label'] = float('nan')
df.label.isnull().sum()


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.describe().T


# In[12]:


df.dtypes


# In[13]:


df.isnull().sum() / df.shape[0]


# In[34]:


plt.ylabel('Data Frequency')
plt.yticks([])
df.date.hist(
    color='b',
    bins=100,
    histtype='stepfilled')
savefig(plt, 'date_distribution')


# In[22]:


df.columns


# In[35]:


col = ['pull_time']
(df[col].astype('timedelta64[s]') / 60)    .plot.hist(bins=50,
               color='b',
               histtype='stepfilled')
plt.xlabel(f'Time elapsed (3rd period)')
plt.yticks([])
savefig(plt, 'goalie_pull_game_times_hist')


# We're interested in knowing about the outcome, given the pull time. This way we can look at the odds of scoring as a function of game time elapsed.

# In[75]:


df.head()


# In[81]:


# ax = plt.subplot(111)
# ax.set_prop_cycle(color=['red', 'green', 'orange'])

df['pull_time_seconds'] = df['pull_time'].astype('timedelta64[s]') / 60

iterables = zip(['orange', 'red', 'green'],
                ['no_goals', 'goal_against', 'goal_for'])

for c, label in iterables:
    (df[df.label==label]['pull_time_seconds']
         .plot.hist(bins=60,
                    alpha=0.6,
                    color=c,
                    histtype='stepfilled',
                    label=label))

plt.xlabel(f'Time elapsed (3rd period)')
plt.yticks([])
plt.legend()

savefig(plt, 'goalie_pull_outcomes_game_times_hist')
del df['pull_time_seconds']


# In[36]:


cols = ['goal_for_time', 'goal_against_time']
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50,
               alpha=0.5,
               color=['green', 'red'],
               histtype='stepfilled')
plt.xlabel('Time elapsed (3rd period)')
plt.yticks([])
savefig(plt, '5_on_6_goals')


# In[37]:


cols = ['goal_for_time', 'goal_against_time']
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50,
               alpha=0.5,
               density='normed',
               color=['green', 'red'],
               histtype='stepfilled')
plt.xlabel('Time elapsed (3rd period)')
plt.yticks([])
savefig(plt, '5_on_6_goals_normed')


# In[38]:


print('Number of goals found:')
(~df[['goal_for_time', 'goal_against_time']].isnull()).sum()


# In[39]:


print('Total goals found:')
(~df[['goal_for_time', 'goal_against_time']].isnull()).sum().sum()


# We also want to model the time between goalie pull and goal (i.e. the timedelta).

# In[27]:


cols = ['game_end_timedelta', 'goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               color=['blue', 'red','green'],
               histtype='stepfilled')
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_goalie_pull_outcomes')


# In[28]:


cols = ['goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               color=['red', 'green',],
               histtype='stepfilled')
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_goalie_pull_goal_timedeltas')


# In[29]:


cols = ['goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               density='normed',
               color=['red', 'green'],
               histtype='stepfilled')
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_goalie_pull_goal_timedeltas_normed')


# The mean/median number of seconds until a goal (after pulling the goalie)

# In[30]:


(df[cols].astype('timedelta64[s]')).mean()


# In[31]:


(df[cols].astype('timedelta64[s]')).median()


# In[33]:


(df['game_end_timedelta'].astype('timedelta64[s]') / 60).plot.hist(bins=50, color='b', histtype='stepfilled')
plt.xlabel('Time since goalie pull (mins)')
plt.yticks([])
savefig(plt, '5_on_6_game_end_timedeltas')


# ### Rough work

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
