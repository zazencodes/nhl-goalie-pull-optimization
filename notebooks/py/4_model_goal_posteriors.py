
# coding: utf-8

# In[42]:


# %load jupyter_default.py
import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import glob
from tqdm import tqdm_notebook
from colorama import Fore, Style

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelpad'] = 20
plt.rcParams['legend.fancybox'] = True
plt.style.use('ggplot')

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)

def savefig(plt, name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)

get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')

from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;}</style>')


# In[43]:


get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


# # Goalie Pull Bayes Optimize
# 
#  - Solve for the posterior probabilties at time t:    
# $
# P(\text{goal for}|\text{goalie pulled}; t)\\
# P(\text{goal against}|\text{goalie pulled}; t)
# $
# 
# The idea is to figure out the **risk reward of pulling a goalie as a function of:**
#  - **t = Time remaining in the game.** For instance, if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for? What is the probability it will result in a goal against?
#  - **t = Time since goalie pull.** For instance, after the goalie has been pulled for 1 minute, what is the chance of seeing a goal for or goal against?
#  
# We'll model these independently.
# 
# Another complication is that games can end without either team scoring. To handle this we could correct our calculated posteriors based on the fraction of games that end this way. Or, better yet, we can calculate $P(\text{game end}|\text{goalie pulled}; t)$ in our model as well.

# ## Train Bayesian Model
# 
# We can model $P(t)$ using a discrete **poisson distribution i.e. if modeling the probability by second. We could also assume a Gamma posterior for the continuous case.**
# 
# We'll solve for this posterior distribution computationally using markov chain monte carlo and the `pymc3` library. 
# 
# Ideally we could use a **uniform prior over the domain of times (last 5mins)**. Note: this is OK since we throw out goalie pulls greater than 5 minutes from the end of the game (due to high likelihood of false positives when parsing goalie pulls from the raw game table).

# In[5]:


import pymc3 as pm


# ### Load the training data

# In[6]:


def load_data():
    files = glob.glob('../../data/processed/pkl/*.pkl')
    files = sorted(files)
    print(files)
    return pd.concat((pd.read_pickle(f) for f in files))

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


# In[7]:


df = load_data()
df = clean_df(df)


# ### Rough work

# #### Data loading

# In[44]:


def load_training_samples(df, cols, dtype='timedelta64[s]') -> np.ndarray:
    '''
    Return buckets of training data.
    '''
    out = []
    for col in cols:
        d = df[col].dropna().astype(dtype).values
        out.append(d)
        print(f'Loaded {len(d)} samples for col {col}')
        
        
    out = np.array(out)
    print(f'shape = {out.shape}')
    return out


# Let's start by modeling the 5 on 6 goal times in 3rd period, where time is a continuous (or rather, discretized by second) and measured in minutes.

# In[45]:


features = ['goal_for_time', 'goal_against_time']
training_samples = load_training_samples(df, features)


# In[46]:


training_samples[0].shape


# In[47]:


training_samples[0][:10]


# To get the proper probabilities, we should weight the 

# #### Modeling

# In[48]:


# with pm.Model() as model:
#     prior_goal_for = pm.Uniform('prior_goal_for', 15, 20)
#     prior_goal_against = pm.Uniform('prior_goal_against', 15, 20)
#     obs_goal_for = pm.Gamma('obs_goal_for', observed=training_samples[0])

# need to set up priors for all the parameters of the gamma!...
# THINK ABOUT IT


# In[60]:


from scipy.stats import poisson


# In[61]:


get_ipython().run_line_magic('pinfo', 'poisson')


# ```
# pmf(k, mu, loc=0)   
#     Probability mass function.
# ```

# In[63]:


x = np.arange(0, 20, 1)
y = [poisson.pmf(_x, 1, 1)
     for _x in x]
plt.plot(x, y)


# In[49]:


def bayes_model(training_samples):
    
    with pm.Model() as model:

        # Posteriors for the mu parameter of the poisson distribution
        # Note that mu = mean(Poisson)
        mu_goal_for = pm.Uniform('mu_goal_for', 15*60, 20*60)
        mu_goal_against = pm.Uniform('mu_goal_against', 15*60, 20*60)

        # Observations
        obs_goal_for = pm.Poisson('obs_goal_for', mu_goal_for, observed=training_samples[0])
        obs_goal_against = pm.Poisson('obs_goal_against', mu_goal_against, observed=training_samples[1])
        
        # Posteriors for the goal probabilities
        p_goal_for = pm.Poisson('p_goal_for', mu_goal_for)
        p_goal_against = pm.Poisson('p_goal_against', mu_goal_against)

        # Fit model
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        
    return model, trace

# N = 10
# test_training_samples = np.array([training_samples[0][:N],
#                                   training_samples[1][:N]])
# model, trace, burned_trace = bayes_model(test_training_samples)
# model


# In[50]:


model, trace = bayes_model(training_samples)
model


# In[51]:


N_burn = 10000
burned_trace = trace[N_burn:]


# In[52]:


get_ipython().run_line_magic('pinfo', 'pm.plots.traceplot')


# In[53]:


pm.plots.traceplot(trace=trace, varnames=['p_goal_for', 'p_goal_against'])


# What do red and blue represent? 

# In[60]:


pm.plots.plot_posterior(trace=trace['p_goal_for'])
pm.plots.plot_posterior(trace=trace['p_goal_against'])


# The HDR is really interesting! For the above case (normally distributed data), the HDR is pretty much equivalent to the SD based confience interval. However it generalizes to more complicated distributions 
# 
# https://stats.stackexchange.com/questions/148439/what-is-a-highest-density-region-hdr
# e.g. 
# 
# ![](https://i.stack.imgur.com/Dy89t.png)

# In[62]:


ALPHA = 0.6

plt.hist(burned_trace['mu_goal_for'], bins=50,
         color='green', label='mu_goal_for',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['mu_goal_against'], bins=50,
         color='red', label='mu_goal_against',
         histtype='stepfilled', alpha=ALPHA)
plt.ylabel('MCMC counts')
plt.xlabel('$\mu$ (seconds)')
plt.legend();


# In[63]:


plt.plot(trace['mu_goal_for'], label='mu_goal_for')
plt.plot(trace['mu_goal_against'], label='mu_goal_against')
plt.ylabel('$\mu$ (seconds)')
plt.xlabel('MCMC step')

plt.axvline(N_burn, color='black', lw=2, label='Burn threshold')

plt.legend();


# Include both those plots in blog ^

# In[64]:


from scipy.special import factorial
poisson = lambda mu, k: mu**k * np.exp(-mu) / factorial(k)
poisson(0.5, np.array([1, 4, 5, 2]))


# In[65]:


from scipy.stats import poisson


# In[66]:


get_ipython().run_line_magic('pinfo', 'poisson.pmf')


# In[67]:


poisson.pmf(3, 1)


# In[68]:


poisson.pmf(np.array([1, 4, 3]), 1)


# In[69]:


p = poisson.pmf
# poisson = lambda k, mu: mu**k * np.exp(-mu) / factorial(k)

x = np.arange(16, 22, 1)

mu_goal_for = burned_trace['mu_goal_for'].mean() / 60
y_goal_for = p(x, mu_goal_for)

mu_goal_against = burned_trace['mu_goal_against'].mean() / 60
y_goal_against = p(x, mu_goal_against)

plt.plot(x, y_goal_for, label=r'$P(\rm{goal\;for};\mu_{avg})$', color='green')
plt.plot(x, y_goal_against, label=r'$P(\rm{goal\;against};\mu_{avg})$', color='red')


# In[70]:


p = poisson.pmf
# poisson = lambda k, mu: mu**k * np.exp(-mu) / factorial(k)

x = np.arange(16*60, 22*60, 1)

mu_goal_for = burned_trace['mu_goal_for'].mean()
y_goal_for = p(x, mu_goal_for)

mu_goal_against = burned_trace['mu_goal_against'].mean()
y_goal_against = p(x, mu_goal_against)

plt.plot(x, y_goal_for, label=r'$P(\rm{goal\;for};\mu_{avg})$', color='green')
plt.plot(x, y_goal_against, label=r'$P(\rm{goal\;against};\mu_{avg})$', color='red')


# In[72]:


ALPHA = 0.6
LW = 3

# plt.hist(burned_trace['p_goal_for'] / 60, bins=50,
#          color='green', label=r'$P(\rm{goal\;for}\;|\;\rm{goalie\;pulled})$',
#          histtype='stepfilled', alpha=ALPHA)

# plt.hist(burned_trace['p_goal_against'] / 60, bins=50,
#          color='red', label=r'$P(\rm{goal\;against}\;|\;\rm{goalie\;pulled})$',
#          histtype='stepfilled', alpha=ALPHA)

''' Plot the MCMC samples '''

plt.hist(burned_trace['p_goal_for'] / 60, bins=50,
         color='green', label='p_goal_for samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['p_goal_against'] / 60, bins=50,
         color='red', label='p_goal_against samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

''' Plot the poisson distributions '''

p = poisson.pmf
x = np.arange(16*60, 22*60, 1)
mu_goal_for = burned_trace['mu_goal_for'].mean()
mu_goal_against = burned_trace['mu_goal_against'].mean()
y_goal_for = p(x, mu_goal_for)
y_goal_against = p(x, mu_goal_against)

# Convert into minutes and rescale to fit chart
x = x / 60
scale_frac = 0.7
y_goal_for = y_goal_for / y_goal_for.max() * scale_frac
y_goal_against = y_goal_against / y_goal_against.max() * scale_frac

plt.plot(x, y_goal_for, label=r'$P(\rm{goal\;for};\mu_{MCMC})$', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'$P(\rm{goal\;against};\mu_{MCMC})$', color='red', lw=LW)

plt.ylabel('Counts')
# plt.yticks([])
plt.xlabel('Game clock (3rd period)')
plt.legend();


# (Do not include this plot ^ in blog, but re-use source code)

# In reality, the probability of an empty net goal should be zero after 20 minutes (since the period is over). We would also need to normalize the probabilities such that   
# 
# $
# \sum_t \big{[} P(\mathrm{goal\;for}; \mu, t) + P(\mathrm{goal\;against}; \mu, t) + P(\mathrm{game\;end}) \big{]} = 1
# $
# 
# Since this was just a toy model to get us warmed up with `pymc`, let's just leave this and move on to a more interesting problem.

# ---

# I wonder if we can answer the question: what are the odds of scoring a goal based on when the goalie is pulled?
# 
# It's probably best to decide that based on the "time since goalie pull" metric and the time remaining in the game. For the chart above, the goal for probability is clearly shifted to the left - however this does not mean that pulling a goalie at the 19 minute mark will have lower odds of a good outcome than pulling at the 18 minute mark. This chart is just a litlihood of scoring given the goalie pull times.
# 
# What we should do is label the goalie pull times with the eventual outcome, then model that.

# ---
# 
# Let's focus on our original question
# 
# What is the **risk reward of pulling a goalie as a function of:**
#  - ~~**t = Time remaining in the game.** For instance, if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for? What is the probability it will result in a goal against?~~
#  - **t = Time since goalie pull.** For instance, after the goalie has been pulled for 1 minute, what is the chance of seeing a goal for or goal against?
