
# coding: utf-8

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
plt.rcParams['legend.fancybox'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'Marion'

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)

plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.labelpad'] = 20
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['xtick.major.pad'] = 15
plt.rcParams['xtick.minor.pad'] = 15
plt.rcParams['ytick.major.pad'] = 10
plt.rcParams['ytick.minor.pad'] = 10

def savefig(name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)


get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas, numpy')


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Alex Galea" -v -p pymc3,matplotlib,seaborn,pandas')


# ## Bayesian Modeling Discussion
# 
# We can model the probability of an outcome $y$ as $P_t(y)$ using a **Gamma distribution** where time $t$ is defined below.
# 
# The PMF is given by
# 
# $$
# P_t(\alpha, \beta) = \frac{\beta^\alpha t^{\alpha-1}e^{-\beta t}}{\Gamma (\alpha)}
# $$
# 
# where $t$ is the time metric and $\alpha$ & $\beta$ are solved for with MCMC.
# 
# Based on a set of goalie pull observations $X$ from 2003-2019 NHL games, we'll solve for the posterior distribution $P_t(y|X)$, the probability of the outcome $y$, given the observations. This is done computationally using markov chain monte carlo and the `pymc3` library.
# 
# The outcomes we're interested in are $y = \big\{\mathrm{goal\;for}, \mathrm{goal\;against}, \mathrm{no\;goal}\big\}$. 
# 
# We'll use a **uniform prior** over the domain of times (last 5mins). Note: when gathering the observations, we throw out goalie pulls greater than 5 minutes from the end of the game (due to high likelihood of false positives when parsing goalie pulls from the raw game table).
# 
# Once we find the posteriors discussed above, we can study the risk reward of pulling a goalie. We'll compare posteriors to find the odds of scoring a goal (and the odds of getting scored on) over time $t$ where:
#  - **t = Time elapsed** e.g. if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for?
#  - **t = Time since goalie pull** e.g. after the goalie has been pulled for 1 minute, what is the chance of getting a goal?

HALF_SPACE = u'\u2000'


import pymc3 as pm


# ### Load the training data

ls ../../data/processed/pkl/


def load_data():
    files = glob.glob('../../data/processed/pkl/*.pkl')
    files = sorted(files)
    print(files)
    return pd.concat((pd.read_pickle(f) for f in files), sort=False, ignore_index=True)

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


df = load_data()
df = clean_df(df)


def load_training_samples(
    df,
    cols,
    masks=[],
    dtype='timedelta64[s]'
) -> np.ndarray:
    '''
    Return buckets of training data.
    '''
    if not masks:
        masks = [None] * len(cols)
    out = []
    for col, m in zip(cols, masks):
        if m is None:
            d = df[col].dropna().astype(dtype).values
        else:
            d = df[col][m].dropna().astype(dtype).values
        out.append(d)
        print(f'Loaded {len(d)} samples for col {col}')

    out = np.array(out)
    print(f'Training data shape = {out.shape}')
    return out


# ## Model 1 - Time elapsed

# ### Load data

# Load time of pull for eventual outcomes:
feature_names = ['goal_for', 'goal_against', 'no_goals']

# Logic for loading the data
features = ['pull_time', 'pull_time', 'pull_time']
masks = [
    ~(df.goal_for_time.isnull()),
    ~(df.goal_against_time.isnull()),
    ~(df.game_end_timedelta.isnull()),
]
training_samples = load_training_samples(df, features, masks)


# We actuall want to fit the mirror version of the Gamma distributioin (about the y-axis). In order to do this with PyMC3, we will transform the data.

MAX_TIME_SECONDS = 20*60
training_samples[0] = MAX_TIME_SECONDS - training_samples[0]
training_samples[1] = MAX_TIME_SECONDS - training_samples[1]
training_samples[2] = MAX_TIME_SECONDS - training_samples[2]


(training_samples[0][:10],
training_samples[1][:10],
training_samples[2][:10],)


feature_names


# ### PyMC3 Model

from typing import Tuple
from pymc3.backends.base import MultiTrace

def bayes_model(training_samples) -> Tuple[pm.model.Model, MultiTrace]:
    """
    Solve for posterior distributions using pymc3
    """
    with pm.Model() as model:

        # Priors
        beta_range = (0, 100)
        alpha_range = (0.001, 10)
        beta_goal_for = pm.Uniform(
            'beta_goal_for', *beta_range
        )
        beta_goal_against = pm.Uniform(
            'beta_goal_against', *beta_range
        )
        beta_no_goal = pm.Uniform(
            'beta_no_goal', *beta_range
        )
        alpha_goal_for = pm.Uniform(
            'alpha_goal_for', *alpha_range
        )
        alpha_goal_against = pm.Uniform(
            'alpha_goal_against', *alpha_range
        )
        alpha_no_goal = pm.Uniform(
            'alpha_no_goal', *alpha_range
        )
        
        # Observations to train the model on
        obs_goal_for = pm.Gamma(
            'obs_goal_for',
            alpha=alpha_goal_for,
            beta=beta_goal_for,
            observed=training_samples[0],
        )
        obs_goal_against = pm.Gamma(
            'obs_goal_against',
            alpha=alpha_goal_against,
            beta=beta_goal_against,
            observed=training_samples[1],
        )
        obs_no_goal = pm.Gamma(
            'obs_no_goal',
            alpha=alpha_no_goal,
            beta=beta_no_goal,
            observed=training_samples[2],
        )
        
        # Outcome probabilities
        p_goal_for = pm.Deterministic(
            'p_goal_for', MAX_TIME_SECONDS - pm.Gamma(
                'gamma_goal_for',
                alpha=alpha_goal_for,
                beta=beta_goal_for,
            )
        )
        p_goal_against = pm.Deterministic(
            'p_goal_against', MAX_TIME_SECONDS - pm.Gamma(
                'gamma_goal_against',
                alpha=alpha_goal_against,
                beta=beta_goal_against,
            )
        )
        p_no_goal = pm.Deterministic(
            'p_no_goal', MAX_TIME_SECONDS - pm.Gamma(
                'gamma_no_goal',
                alpha=alpha_no_goal,
                beta=beta_no_goal,
            )
        )
        
        # Fit model
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        
    return model, trace

model, trace = bayes_model(training_samples)
model


# ### Save / Load the model
# 
# Note: using pickle on untrusted files is dangerous. Use at your own risk. No warranty as per licence.

SAVE_MODEL = True
LOAD_MODEL = True
MODEL_PATH = ('../../models/gamma_mcmc_{}.pymc3.pkl'
    .format(datetime.datetime.now().strftime('%Y-%m-%d'))
)
TRACE_PATH = ('../../models/gamma_mcmc_trace_{}.pymc3.dat'
    .format(datetime.datetime.now().strftime('%Y-%m-%d'))
)

import pickle

if SAVE_MODEL:
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'trace': trace}, f)

MODEL_PATH = "../../models/gamma_mcmc_2019-07-07.pymc3.pkl"

if LOAD_MODEL:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        
    model = data['model']
    trace = data['trace']
    del data


# ### MCMC Samples

N_burn = 10000
burned_trace = trace[N_burn:]


from typing import Tuple
from scipy.stats import gamma

def gamma_posterior(
    alpha=None,
    beta=None,
    norm_factors=None,
) -> Tuple[np.ndarray]:

    p = gamma.pdf
    x = np.arange(0*60, 5*60, 1)
    if alpha is None or beta is None:
        return (x / 60,)

    y_goal_for = p(x, alpha[0], scale=1/beta[0])
    y_goal_against = p(x, alpha[1], scale=1/beta[1])
    y_no_goal = p(x, alpha[2], scale=1/beta[2])
    
    if norm_factors is not None:
        y_goal_for = y_goal_for * norm_factors[0]
        y_goal_against = y_goal_against * norm_factors[1]
        y_no_goal = y_no_goal * norm_factors[2]
    
    # Transform x
    x = MAX_TIME_SECONDS - x
    
    # Convert into minutes
    x = x / 60

    return x, y_goal_for, y_goal_against, y_no_goal


ALPHA = 0.6
LW = 3

''' Plot MCMC samples '''

plt.hist(burned_trace['p_goal_for']/60, bins=50,
         color='green', label='p_goal_for samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['p_goal_against']/60, bins=50,
         color='red', label='p_goal_against samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['p_no_goal']/60, bins=50,
         color='orange', label='p_no_goal samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

''' Plot modeled distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha=[
        burned_trace['alpha_goal_for'].mean(),
        burned_trace['alpha_goal_against'].mean(),
        burned_trace['alpha_no_goal'].mean(),
    ],
    beta=[
        burned_trace['beta_goal_for'].mean(),
        burned_trace['beta_goal_against'].mean(),
        burned_trace['beta_no_goal'].mean(),
    ],
)

# Rescale height by arbitrary amount to fit chart
scale_frac = 0.5
y_goal_for = y_goal_for / y_goal_for.max() * scale_frac
y_goal_against = y_goal_against / y_goal_against.max() * scale_frac
scale_frac = 0.65
y_no_goal = y_no_goal / y_no_goal.max() * scale_frac

plt.plot(x, y_goal_for, label=r'$P(\rm{goal\;for})$', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'$P(\rm{goal\;against})$', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'$P(\rm{no\;goal})$', color='orange', lw=LW)

''' Clean up the chart '''

plt.ylabel('Counts')
ax = plt.gca()
ax.set_yticklabels([])

plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.legend()

plt.text(x=11.6, y=0.770,
    s='Goalie Pull Outcome Models',
    fontsize=24, color='black', weight='bold')

plt.text(x=11.6, y=0.71,
    s='MCMC gamma posterior models and samples\nfor the three possilbe goalie pull outcomes.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.5, y=0.71,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_mcmc_samples')

plt.show()


fig = pm.traceplot(trace);
savefig('time_elapsed_gamma_mcmc_traceplot')


plt.plot(trace['alpha_goal_for']/60, label='Goal For', color='green', lw=1)
plt.plot(trace['alpha_goal_against']/60, label='Goal Against', color='red', lw=1)
plt.plot(trace['alpha_no_goal']/60, label='No Goal', color='orange', lw=1)
plt.ylabel(r'$\alpha$')
plt.xlabel('MCMC step')

plt.axvline(N_burn, color='black', lw=2, label='Burn threshold')

plt.legend()

offset = 0.00
plt.text(x=-3700, y=0.103-offset,
    s='Alpha MCMC Samples',
    fontsize=24, color='black', weight='bold')

plt.text(x=-3700, y=0.1005-offset,
    s='MCMC trace for the gamma model alpha parameter.',
    fontsize=14, color='black', style='italic')

plt.text(x=68000, y=0.1005-offset,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_alpha_steps')

plt.show()


plt.plot(trace['beta_goal_for']/60, label='Goal For', color='green', lw=1)
plt.plot(trace['beta_goal_against']/60, label='Goal Against', color='red', lw=1)
plt.plot(trace['beta_no_goal']/60, label='No Goal', color='orange', lw=1)
plt.ylabel(r'$\beta$')
plt.xlabel('MCMC step')

plt.axvline(N_burn, color='black', lw=2, label='Burn threshold')

plt.legend()

offset = 0.0000
plt.text(x=-3700, y=0.000936-offset,
    s='Beta MCMC Samples',
    fontsize=24, color='black', weight='bold')

plt.text(x=-3700, y=0.00092-offset,
    s='MCMC trace for the gamma model beta parameter.',
    fontsize=14, color='black', style='italic')


plt.text(x=68000, y=0.00092-offset,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_beta_steps')

plt.show()


ALPHA = 0.6

plt.hist(burned_trace['alpha_goal_for']/60, bins=50,
         color='green', label='alpha_goal_for',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['alpha_goal_against']/60, bins=50,
         color='red', label='alpha_goal_against',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['alpha_no_goal']/60, bins=50,
         color='orange', label='alpha_no_goal',
         histtype='stepfilled', alpha=ALPHA)

plt.ylabel('MCMC counts')
plt.xlabel(r'$\alpha$')
plt.legend()

savefig('time_elapsed_gamma_alpha_samples')
plt.show()


ALPHA = 0.6

plt.hist(burned_trace['beta_goal_for']/60, bins=50,
         color='green', label='beta_goal_for',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['beta_goal_against']/60, bins=50,
         color='red', label='beta_goal_against',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['beta_no_goal']/60, bins=50,
         color='orange', label='beta_no_goal',
         histtype='stepfilled', alpha=ALPHA)

plt.ylabel('MCMC counts')
plt.xlabel(r'$\beta$')
plt.legend()

savefig('time_elapsed_gamma_beta_samples')
plt.show()


# ### Normalization

alpha_mcmc = [
    burned_trace['alpha_goal_for'].mean(),
    burned_trace['alpha_goal_against'].mean(),
    burned_trace['alpha_no_goal'].mean(),
]

beta_mcmc = [
    burned_trace['beta_goal_for'].mean(),
    burned_trace['beta_goal_against'].mean(),
    burned_trace['beta_no_goal'].mean(),
]

print(f'MCMC values for alpha: {alpha_mcmc}')
print(f'MCMC values for beta: {beta_mcmc}')


mcmc_normalizing_factors = np.array([
    training_samples[0].shape[0],
     training_samples[1].shape[0],
     training_samples[2].shape[0]
])
mcmc_normalizing_factors = mcmc_normalizing_factors / mcmc_normalizing_factors.sum()

print(f'MCMC normalizing factors =\n{mcmc_normalizing_factors}')


x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(alpha=alpha_mcmc, beta=beta_mcmc)

y_goal_for = y_goal_for * mcmc_normalizing_factors[0]
y_goal_against = y_goal_against * mcmc_normalizing_factors[1]
y_no_goal = y_no_goal * mcmc_normalizing_factors[2]

cutoff_renormed_factor = 2 - (y_goal_for.sum() + y_goal_against.sum() + y_no_goal.sum())
model_normalizing_factors = mcmc_normalizing_factors * cutoff_renormed_factor

print(f'Posterior distribution normalizing factors =\n{model_normalizing_factors}')


# Here's what the properly weighted samlpes look like:

ALPHA = 0.6
LW = 3
BINS = 60

''' Plot the MCMC samples '''

plt.hist(np.random.choice(
            burned_trace['p_goal_for'] / 60,
            size=int(burned_trace['p_goal_for'].shape[0] * mcmc_normalizing_factors[0])
         ),
         bins=BINS, color='green', label='p_goal_for samples',
         histtype='stepfilled', alpha=ALPHA, zorder=3)

plt.hist(np.random.choice(
            burned_trace['p_goal_against'] / 60,
            size=int(burned_trace['p_goal_against'].shape[0] * mcmc_normalizing_factors[1])
         ),
         bins=BINS,
         color='red', label='p_goal_against samples',
         histtype='stepfilled', alpha=ALPHA, zorder=2)

plt.hist(np.random.choice(
            burned_trace['p_no_goal'] / 60,
            size=int(burned_trace['p_no_goal'].shape[0] * mcmc_normalizing_factors[2])
         ),
         bins=BINS,
         color='orange', label='p_no_goal samples',
         histtype='stepfilled', alpha=ALPHA)

plt.ylabel('Counts (normed)')

plt.ylim(0, 1400)
plt.xlim(12.5, 20.5)

ax = plt.gca()
ax.set_yticklabels([])

plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.legend()

plt.text(x=12.5, y=1550,
    s='Goalie Pull Outcome Models (Normed)',
    fontsize=24, color='black', weight='bold')

plt.text(x=12.5, y=1450,
    s='Normalized MCMC gamma posterior samples.\nBar heights reflect the relative probability of each outcome at time t.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.75, y=1440,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')

savefig('time_elapsed_gamma_normed_mcmc_samples')

plt.show()


# ### Normalized Posteriors
# 
# Re-normalize for cutoff Gamma distributions

import inspect
print(inspect.getsource(gamma_posterior))


ALPHA = 0.6
LW = 3

''' Plot the modeled distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)
plt.plot(x, y_goal_for, label=r'$P(\mathrm{goal\;for}\;|\;X)$', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'$P(\mathrm{goal\;against}\;|\;X)$', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'$P(\mathrm{no\;goal}\;|\;X)$', color='orange', lw=LW)

plt.ylabel('Posterior probability')
ax = plt.gca()
ax.set_yticklabels([])
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.legend()

plt.text(x=14.78, y=0.00672,
    s='Goalie Pull Outcome Models (Normed)',
    fontsize=24, color='black', weight='bold')

plt.text(x=14.78, y=0.00625,
    s='Normalized MCMC gamma posterior models.\nLine heights reflect the relative probability of each outcome at time $t$.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.72, y=0.00625,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_normed')

plt.show()


# ### Interpretation

def convert_to_time_remaining(x):
    _x = 20 - x
    t = datetime.timedelta(seconds=_x*60)
    return str(t)

convert_to_time_remaining(x[np.argmax(y_goal_for)])


print('Time of max posterior probability =\n'
      f'{x[np.argmax(y_goal_for)], x[np.argmax(y_goal_against)], x[np.argmax(y_no_goal)]}')

print()

t_remaining = [convert_to_time_remaining(x[np.argmax(y_goal_for)]),
              convert_to_time_remaining(x[np.argmax(y_goal_against)]),
              convert_to_time_remaining(x[np.argmax(y_no_goal)])]

print(f'Time of max posterior probability =\n{t_remaining}')


# Great, now we have properly normalized probabilties.
# 
# #### Note: comments below may apply to previous models
# 
#  - From normalizing factors, we can see ~13% chance of scoring when pulling the goalie (on average for all times).
#  - Probability of scoring peaks at 1:24 remaining, with other probabilties following after (01:19 for goal against and 0:41 for no goals).
#  
# From now on we'll work from the distributions as our source of truth. These are hard coded below to help with reproducibility.

# Note: make sure this cell is consistent with results above

model_normlizing_factors = [0.13425752, 0.33528483, 0.53281211]
alpha_mcmc = [4.535463169388722, 3.8818129700995634, 2.3765137077097944]
beta_mcmc = [0.0420821857715431, 0.03668029689191766, 0.033411186867245306]


# ### Cumulative sum
# 
# Calculating the CDF will allow us to make some interesting observations on the results.

x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

plt.plot(x, np.cumsum(y_goal_for)[::-1], label=r'$cumsum [ P(\mathrm{goal\;for}\;|\;X) ]$', color='green', lw=LW)
plt.plot(x, np.cumsum(y_goal_against)[::-1], label=r'$cumsum [ P(\mathrm{goal\;against}\;|\;X) ]$', color='red', lw=LW)
plt.plot(x, np.cumsum(y_no_goal)[::-1], label=r'$cumsum [ P(\mathrm{no\;goal}\;|\;X) ]$', color='orange', lw=LW)

plt.ylabel('Posterior CDF')
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.legend()

ax = plt.gca()
ax.yaxis.tick_right()

plt.text(x=14.78, y=0.623,
    s='Goalie Pull Outcome Model CDF',
    fontsize=24, color='black', weight='bold')

plt.text(x=14.78, y=0.58,
    s='Cumulative sum of posterior outcome probabilities. Line heights on far right side\nsum to 1 and represent the average probability of each outcome.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.72, y=0.57,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_cdf')

plt.show()


mean_posteriors = [
    y_goal_for.sum(),
    y_goal_against.sum(),
    y_no_goal.sum(),
]

print(f'Time of max posterior probabilities =\n{mean_posteriors}')


# The end of game values have been normalized sum up to one, but this ratio changes over time. We can visualize this with the risk-reward ratio (see below).
# 
# ### Re-normalize
# 
# To better compare these probability distributions, we can normalize each bin to 1 using a function $c(t)$, as follows:
# 
# $$
# c(t) \cdot \big[ P(goal\;for; t) + (P(goal\;against; t) + P(no\;goal; t)\big] = 1 \\
# \vdots \\
# c(t) = \big[ P(goal\;for; t) + (P(goal\;against; t) + P(no\;goal; t)\big]^{-1}
# $$
# 
# This will allow us to re-weight the posteriors later, so we can compare them better and yield a different interpretation.
# 
# Essentially, we'll be able to interpret the resulting distribution as the chance of each outcome at time $t$. This stands in contrast to the probability distributions above, where the total area under the curves sum to 1.

alpha = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)


plt.plot(x, alpha, lw=LW)
plt.ylabel('$c(t)$')
# plt.yticks([])
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.show()


ALPHA = 0.6
LW = 3 

''' Plot the gamma distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

# Alpha has same shape as x, y above
alpha = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)

y_goal_for = alpha * y_goal_for
y_goal_against = alpha * y_goal_against
y_no_goal = alpha * y_no_goal

max_index = np.where(y_goal_for == y_goal_for[~(np.isnan(y_goal_for))].max())[0][0]
plt.axvline(x[max_index], label='Goal For Max Likelihood', color='black', lw=2, alpha=0.5)

plt.plot(x, y_goal_for, label='Goal For', color='green', lw=LW)
plt.plot(x, y_goal_against, label='Goal Against', color='red', lw=LW)
plt.plot(x, y_no_goal, label='No Goal', color='orange', lw=LW)

plt.ylabel('Chance of outcome')
# plt.yticks([])
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.legend()

plt.text(x=14.78, y=1.18,
    s='Goalie Pull Outcome Probabilities',
    fontsize=24, color='black', weight='bold')

plt.text(x=14.78, y=1.1,
    s='The chance of each outcome if pulling the goalie at time $t$.\nThe point of maximum Goal For likelihood is marked with a vertical black line.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.72, y=1.1,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')

# Plotting below with error bar
savefig('time_elapsed_gamma_outcome_likelihoods')

plt.show()


# ### Adding error bars

# Note how there are very few samples to draw conclusions from for the low and high times.
# 
# e.g. less than 3 minutes:

np.sum(training_samples[0] < 3*60) + np.sum(training_samples[1] < 3*60) + np.sum(training_samples[2] < 3*60)


# compared to more than 3 minutes:

np.sum(training_samples[0] > 3*60) + np.sum(training_samples[1] > 3*60) + np.sum(training_samples[2] > 3*60)


566 / 11000 * 100


# Only about 5% are over 3 minutes.

# We can show this uncertainty visually using error bars. Starting with the parameter ($\alpha$ and $\beta$) MCMC samples...

plt.hist(burned_trace['alpha_goal_for'])
plt.hist(burned_trace['alpha_goal_against'])
plt.hist(burned_trace['alpha_no_goal'])


plt.hist(burned_trace['beta_goal_for'])
plt.hist(burned_trace['beta_goal_against'])
plt.hist(burned_trace['beta_no_goal'])


# We can use the uncertainty on $\alpha$ and $\beta$ to calculate that for $P$:
# 
# $$
# \sigma_P = \sqrt{ \big ( \frac{\partial P}{\partial \alpha} \;\sigma_{\alpha} \big )^{2} + \big ( \frac{\partial P}{\partial \beta} \;\sigma_{\beta} \big )^{2} }
# $$
# 
# where $\sigma_{\mu}$ is the standard deviation of the $\mu$ samples. We are interested in the standard error of the mean:
# 
# $$
# \sigma_{\bar{P}} = \frac{\sigma_P}{\sqrt{N_{samples}}}
# $$

alpha_mcmc_std = [
    burned_trace['alpha_goal_for'].std(),
    burned_trace['alpha_goal_against'].std(),
    burned_trace['alpha_no_goal'].std(),
]


beta_mcmc_std = [
    burned_trace['beta_goal_for'].std(),
    burned_trace['beta_goal_against'].std(),
    burned_trace['beta_no_goal'].std(),
]


alpha_mcmc_std


beta_mcmc_std


num_samples = len(burned_trace)


num_samples


model_normalizing_factors


from scipy.misc import derivative
from scipy.stats import gamma

def calc_posteror_error(alpha, beta, alpha_std, beta_std, num_samples, step=1e-8):
    x = gamma_posterior()[0] * 60 # convert back into seconds (discrete)
    err = np.sqrt(
        alpha_std * np.power(np.array([
                        derivative(lambda _alpha: gamma.pdf(int(t), _alpha, scale=1/beta),
                                    alpha, dx=step)
                        for t in x
                    ]), 2)
        + beta_std * np.power(np.array([
                        derivative(lambda _beta: gamma.pdf(int(t), alpha, scale=1/_beta),
                                    beta, dx=step)
                        for t in x
                    ]), 2)
    ) / np.sqrt(num_samples)
    # Flip error due to gamma transformation
    err = err[::-1]
    return err


# Note the y-axis corresponds to:
#     
# $$
# c(t) \cdot P(\mathrm{goal\;for}\;|\;X;\;t) \\
# c(t) \cdot P(\mathrm{goal\;against}\;|\;X;\;t) \\
# c(t) \cdot P(\mathrm{no\;goal}\;|\;X;\;t)
# $$
# 
# for each line respectively.

# Plotting with error bands

ALPHA = 1
ALPHA_LIGHT = 0.2
LW = 3

''' Plot the gamma distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

# Alpha has same shape as x, y above
c = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)

y_goal_for = c * y_goal_for
y_goal_against = c * y_goal_against
y_no_goal = c * y_no_goal

max_index = np.where(y_goal_for == y_goal_for[~(np.isnan(y_goal_for))].max())[0][0]
plt.axvline(x[max_index], label='Goal For Max Likelihood', color='black', lw=2, alpha=0.5)
plt.plot(x, y_goal_for, label=r'Goal For', color='green', lw=LW, alpha=ALPHA)
plt.plot(x, y_goal_against, label=r'Goal Against', color='red', lw=LW, alpha=ALPHA)
plt.plot(x, y_no_goal, label=r'No Goal', color='orange', lw=LW, alpha=ALPHA)

''' Plot the errors '''
err_p_goal_for = c * calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0], num_samples=num_samples)
err_p_goal_against = c * calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1], num_samples=num_samples)
err_p_no_goal = c * calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2], num_samples=num_samples)
plt.fill_between(x, y_goal_for-err_p_goal_for, y_goal_for+err_p_goal_for,
                 color='green', alpha=ALPHA_LIGHT)
plt.fill_between(x, y_goal_against-err_p_goal_against, y_goal_against+err_p_goal_against,
                 color='red', alpha=ALPHA_LIGHT)
plt.fill_between(x, y_no_goal-err_p_no_goal, y_no_goal+err_p_no_goal,
                 color='orange', alpha=ALPHA_LIGHT)

plt.ylabel('Chance of outcome')
# plt.yticks([])
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.xlim(15.7, 20)
plt.ylim(0, 1)
plt.legend()

plt.text(x=15.67, y=1.12,
    s='Goalie Pull Outcome Probabilities',
    fontsize=24, color='black', weight='bold')

plt.text(x=15.67, y=1.05,
    s='The chance of each outcome if pulling the goalie at time $t$. Error bands\nrepresent statistical uncertainty, which is dominant for early times.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.6, y=1.05,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_outcome_likelihoods_error_bars')

plt.show()


# Plotting with dotted lines for error bars

ALPHA = 0.8
ALPHA_LIGHT = 0.3
LW = 3

''' Plot the gamma distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

# Alpha has same shape as x, y above
c = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)

y_goal_for = c * y_goal_for
y_goal_against = c * y_goal_against
y_no_goal = c * y_no_goal
plt.plot(x, y_goal_for, label=r'goal for', color='green', lw=LW, alpha=ALPHA)
plt.plot(x, y_goal_against, label=r'goal against', color='red', lw=LW, alpha=ALPHA)
plt.plot(x, y_no_goal, label=r'no goal', color='orange', lw=LW, alpha=ALPHA)

''' Plot the errors '''
err_p_goal_for = c * calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0], num_samples=num_samples)
err_p_goal_against = c * calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1], num_samples=num_samples)
err_p_no_goal = c * calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2], num_samples=num_samples)


plt.plot(x, y_goal_for + err_p_goal_for, color='green', ls='--', lw=LW, alpha=ALPHA_LIGHT)
plt.plot(x, y_goal_against + err_p_goal_against, color='red', ls='--', lw=LW, alpha=ALPHA_LIGHT)
plt.plot(x, y_no_goal + err_p_no_goal, color='orange', ls='--', lw=LW, alpha=ALPHA_LIGHT)
plt.plot(x, y_goal_for - err_p_goal_for, color='green', ls='--', lw=LW, alpha=ALPHA_LIGHT)
plt.plot(x, y_goal_against - err_p_goal_against, color='red', ls='--', lw=LW, alpha=ALPHA_LIGHT)
plt.plot(x, y_no_goal - err_p_no_goal, color='orange', lw=LW, ls='--', alpha=ALPHA_LIGHT)

plt.ylabel('Chance of outcome')
# plt.yticks([])
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.xlim(15.1, 20)
plt.ylim(0, 1)
plt.legend()

# savefig('time_elapsed_gamma_outcome_likelihoods_error_bars')

plt.show()


# We can't say anything conclusive due to huge errors on low times, but we are much more confident on late game predictions

# ### Odds of scoring a goal
# Let's go into odds-space and look at the chance of scoring a goal, compared to either outcome. We want to maximze this.

from scipy.misc import derivative
from scipy.stats import gamma

def calc_posteror_odds_error(alpha_mcmc, beta_mcmc, alpha_mcmc_std, beta_mcmc_std, num_samples,
                             y_goal_for, y_goal_against, y_no_goal):
    err_p_goal_for = calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0], num_samples=num_samples)
    err_p_goal_against = calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1], num_samples=num_samples)
    err_p_no_goal = calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2], num_samples=num_samples)

    # Adjust for odds ratio calculation
    err_odds_goal_for = (
        np.power(err_p_goal_for / y_goal_for, 2)
        + np.power(err_p_goal_against / y_goal_against, 2)
        + np.power(err_p_no_goal / y_no_goal, 2)
    )
    err_odds_goal_for = odds_goal_for * np.sqrt(err_odds_goal_for)
    
    return err_odds_goal_for


ALPHA = 0.8
ALPHA_LIGHT = 0.2
LW = 3

''' Odds ratio '''

x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

odds_goal_for = y_goal_for / (y_goal_against + y_no_goal)

''' Error bars '''
err_odds_goal_for = calc_posteror_odds_error(alpha_mcmc, beta_mcmc, alpha_mcmc_std, beta_mcmc_std, num_samples,
                                             y_goal_for, y_goal_against, y_no_goal)

''' Plots '''

plt.plot(x, odds_goal_for,
         label=r'$odds(\mathrm{goal\;for})$',
         color='green', lw=LW, alpha=ALPHA)
plt.fill_between(x, odds_goal_for-err_odds_goal_for, odds_goal_for+err_odds_goal_for,
                 color='green', lw=LW, alpha=ALPHA_LIGHT)

max_index = np.where(odds_goal_for == odds_goal_for[~(np.isnan(odds_goal_for))].max())[0][0]
plt.axvline(x[max_index], label='Max Likelihood', color='black', lw=2, alpha=0.5)

plt.ylabel('Odds of success')
# plt.yticks([])
plt.xlabel('Time elapsed in 3rd period (minutes)')

plt.xlim(15.1, 19.8)
plt.ylim(0, 0.4)

plt.legend()

plt.text(x=15.1, y=0.449,
    s='Goalie Pull Scoring Odds',
    fontsize=24, color='black', weight='bold')

plt.text(x=15.1, y=0.42,
    s='The chance of scoring if pulling the goalie at time $t$. Error bands\nrepresent statistical uncertainty, which is dominant for early times.',
    fontsize=14, color='black', style='italic')

plt.text(x=19.3, y=0.42,
    s='alexgalea.ca',
    fontsize=14, color='black', style='italic')


savefig('time_elapsed_gamma_odds_goal_for')

plt.show()


f'best odds of scoring = {(odds_goal_for-err_odds_goal_for)[~np.isnan(odds_goal_for-err_odds_goal_for)].max()}'


f't=17 odds of scoring (high) = {(odds_goal_for+err_odds_goal_for)[x == 17][0]}'


f't=17 odds of scoring = {(odds_goal_for)[x == 17][0]}'


f't=17 odds of scoring (low) = {(odds_goal_for-err_odds_goal_for)[x == 17][0]}'


0.26119 - 0.303633


# This chart suggests that odds of scoring are highest (~22%) when the goalie is pulled at the 17 minute mark. It also turns out to be oddly phallic.

# ## Model 2 - Time since goalie pull (*note: this model is not being maintained*)
# 
# The work thus far has been to model the outcomes as a function of "time 
# elapsed". Now we'll shift our attention to "time since goalie pull".

import inspect
print(inspect.getsource(load_training_samples))


df.head()


# Load time of pull for eventual outcomes:
feature_names = ['goal_for_timedelta', 'goal_against_timedelta', 'game_end_timedelta']
training_samples = load_training_samples(df=df, cols=feature_names)


# There should be no samples with t=0, otherwise MCMC does not converge.

training_samples[0][np.where(training_samples[0]==0)] = 1
training_samples[1][np.where(training_samples[1]==0)] = 1
training_samples[2][np.where(training_samples[2]==0)] = 1


(training_samples[0][:10],
training_samples[1][:10],
training_samples[2][:10],)


feature_names


from typing import Tuple
from pymc3.backends.base import MultiTrace

def bayes_model(training_samples) -> Tuple[pm.model.Model, MultiTrace]:
    """
    Solve for posterior distributions using pymc3
    """
    with pm.Model() as model:

        # Priors
#         beta_range = (0, 1000)
#         alpha_range = (0.0001, 100)
        beta_range = (0, 100)
        alpha_range = (0.001, 10)
        beta_goal_for = pm.Uniform(
            'beta_goal_for', *beta_range
        )
        beta_goal_against = pm.Uniform(
            'beta_goal_against', *beta_range
        )
        beta_no_goal = pm.Uniform(
            'beta_no_goal', *beta_range
        )
        alpha_goal_for = pm.Uniform(
            'alpha_goal_for', *alpha_range
        )
        alpha_goal_against = pm.Uniform(
            'alpha_goal_against', *alpha_range
        )
        alpha_no_goal = pm.Uniform(
            'alpha_no_goal', *alpha_range
        )
        
        # Observations to train the model on
        obs_goal_for = pm.Gamma(
            'obs_goal_for',
            alpha=alpha_goal_for,
            beta=beta_goal_for,
            observed=training_samples[0],
        )
        obs_goal_against = pm.Gamma(
            'obs_goal_against',
            alpha=alpha_goal_against,
            beta=beta_goal_against,
            observed=training_samples[1],
        )
        obs_no_goal = pm.Gamma(
            'obs_no_goal',
            alpha=alpha_no_goal,
            beta=beta_no_goal,
            observed=training_samples[2],
        )
        
        # Outcome probabilities
        p_goal_for = pm.Gamma(
            'p_goal_for', 
            alpha=alpha_goal_for,
            beta=beta_goal_for,
        )
        p_goal_against = pm.Gamma(
            'p_goal_against',
            alpha=alpha_goal_against,
            beta=beta_goal_against,
        )
        p_no_goal = pm.Gamma(
            'p_no_goal',
            alpha=alpha_no_goal,
            beta=beta_no_goal,
        )
        
        # Fit model
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        
    return model, trace

model, trace = bayes_model(training_samples)
model


N_burn = 10000
burned_trace = trace[N_burn:]


fig = plt.figure()


fig = pm.traceplot(trace);
fig = plt.gcf()
savefig(fig, 'timesince_traceplot_gamma_mcmc')


from typing import Tuple
from scipy.stats import gamma

def gamma_posterior(
    alpha=None,
    beta=None,
    norm_factors=None,
) -> Tuple[np.ndarray]:

    p = gamma.pdf
    x = np.arange(0*60, 5*60, 1)
    if alpha is None or beta is None:
        return (x / 60,)

    y_goal_for = p(x, alpha[0], scale=1/beta[0])
    y_goal_against = p(x, alpha[1], scale=1/beta[1])
    y_no_goal = p(x, alpha[2], scale=1/beta[2])
    
    if norm_factors is not None:
        y_goal_for = y_goal_for * norm_factors[0]
        y_goal_against = y_goal_against * norm_factors[1]
        y_no_goal = y_no_goal * norm_factors[2]
    
    # Convert into minutes
    x = x / 60

    return x, y_goal_for, y_goal_against, y_no_goal


# ### MCMC Samples

ALPHA = 0.6
LW = 3

''' Plot MCMC samples '''

plt.hist(burned_trace['p_goal_for']/60, bins=50,
         color='green', label='p_goal_for samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['p_goal_against']/60, bins=50,
         color='red', label='p_goal_against samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['p_no_goal']/60, bins=50,
         color='orange', label='p_no_goal samples',
         density='normed',
         histtype='stepfilled', alpha=ALPHA)

''' Plot modeled distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha=[
        burned_trace['alpha_goal_for'].mean(),
        burned_trace['alpha_goal_against'].mean(),
        burned_trace['alpha_no_goal'].mean(),
    ],
    beta=[
        burned_trace['beta_goal_for'].mean(),
        burned_trace['beta_goal_against'].mean(),
        burned_trace['beta_no_goal'].mean(),
    ],
)

# Rescale height by arbitrary amount to fit chart
scale_frac = 1
y_goal_for = y_goal_for / y_goal_for.max() * scale_frac
y_goal_against = y_goal_against / y_goal_against.max() * scale_frac
y_no_goal = y_no_goal / y_no_goal.max() * scale_frac

plt.plot(x, y_goal_for, label=r'$P(\rm{goal\;for})$', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'$P(\rm{goal\;against})$', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'$P(\rm{no\;goal})$', color='orange', lw=LW)

''' Clean up the chart '''

plt.ylabel('Counts')
# plt.yticks([])
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.legend()

savefig(plt, 'time_since_gamma_mcmc_samples')

plt.show()


plt.plot(trace['alpha_goal_for']/60, label='mu_goal_for', color='green')
plt.plot(trace['alpha_goal_against']/60, label='mu_goal_against', color='red')
plt.plot(trace['alpha_no_goal']/60, label='mu_no_goal', color='orange')
plt.ylabel(r'$\alpha$')
plt.xlabel('MCMC step')

plt.axvline(N_burn, color='black', lw=2, label='Burn threshold')

plt.legend()

savefig(plt, 'time_since_gamma_alpha_steps')

plt.show()


plt.plot(trace['beta_goal_for']/60, label='mu_goal_for', color='green')
plt.plot(trace['beta_goal_against']/60, label='mu_goal_against', color='red')
plt.plot(trace['beta_no_goal']/60, label='mu_no_goal', color='orange')
plt.ylabel(r'$\beta$')
plt.xlabel('MCMC step')

plt.axvline(N_burn, color='black', lw=2, label='Burn threshold')

plt.legend()

savefig(plt, 'time_since_gamma_beta_steps')

plt.show()


ALPHA = 0.6

plt.hist(burned_trace['alpha_goal_for']/60, bins=50,
         color='green', label='mu_goal_for',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['alpha_goal_against']/60, bins=50,
         color='red', label='mu_goal_against',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['alpha_no_goal']/60, bins=50,
         color='orange', label='mu_no_goal',
         histtype='stepfilled', alpha=ALPHA)

plt.ylabel('MCMC counts')
plt.xlabel(r'$\alpha$')
plt.legend()

savefig(plt, 'time_since_gamma_alpha_samples')
plt.show()


ALPHA = 0.6

plt.hist(burned_trace['beta_goal_for']/60, bins=50,
         color='green', label='mu_goal_for',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['beta_goal_against']/60, bins=50,
         color='red', label='mu_goal_against',
         histtype='stepfilled', alpha=ALPHA)

plt.hist(burned_trace['beta_no_goal']/60, bins=50,
         color='orange', label='mu_no_goal',
         histtype='stepfilled', alpha=ALPHA)

plt.ylabel('MCMC counts')
plt.xlabel(r'$\beta$')
plt.legend()

savefig(plt, 'time_since_gamma_beta_samples')
plt.show()


# ### Normalization

alpha_mcmc = [
    burned_trace['alpha_goal_for'].mean(),
    burned_trace['alpha_goal_against'].mean(),
    burned_trace['alpha_no_goal'].mean(),
]

beta_mcmc = [
    burned_trace['beta_goal_for'].mean(),
    burned_trace['beta_goal_against'].mean(),
    burned_trace['beta_no_goal'].mean(),
]

print(f'MCMC values for alpha: {alpha_mcmc}')
print(f'MCMC values for beta: {beta_mcmc}')


mcmc_normalizing_factors = np.array([
    training_samples[0].shape[0],
     training_samples[1].shape[0],
     training_samples[2].shape[0]
])
mcmc_normalizing_factors = mcmc_normalizing_factors / mcmc_normalizing_factors.sum()

print(f'MCMC normalizing factors =\n{mcmc_normalizing_factors}')


x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(alpha=alpha_mcmc, beta=beta_mcmc)

y_goal_for = y_goal_for * mcmc_normalizing_factors[0]
y_goal_against = y_goal_against * mcmc_normalizing_factors[1]
y_no_goal = y_no_goal * mcmc_normalizing_factors[2]

cutoff_renormed_factor = 2 - (y_goal_for.sum() + y_goal_against.sum() + y_no_goal.sum())
model_normalizing_factors = mcmc_normalizing_factors * cutoff_renormed_factor

print(f'Poisson normalizing factors =\n{model_normalizing_factors}')


# Here's what the properly weighted samlpes look like:

ALPHA = 0.6
LW = 3
BINS = 60

''' Plot the MCMC samples '''

plt.hist(np.random.choice(
            burned_trace['p_goal_for'] / 60,
            size=int(burned_trace['p_goal_for'].shape[0] * mcmc_normalizing_factors[0])
         ),
         bins=BINS, color='green', label='p_goal_for samples',
         histtype='stepfilled', alpha=ALPHA, zorder=3)

plt.hist(np.random.choice(
            burned_trace['p_goal_against'] / 60,
            size=int(burned_trace['p_goal_against'].shape[0] * mcmc_normalizing_factors[1])
         ),
         bins=BINS,
         color='red', label='p_goal_against samples',
         histtype='stepfilled', alpha=ALPHA, zorder=2)

plt.hist(np.random.choice(
            burned_trace['p_no_goal'] / 60,
            size=int(burned_trace['p_no_goal'].shape[0] * mcmc_normalizing_factors[2])
         ),
         bins=BINS,
         color='orange', label='p_no_goal samples',
         histtype='stepfilled', alpha=ALPHA)

plt.ylabel('Sampled frequency (normed)')
plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')
plt.legend();

savefig(plt, 'time_since_normed_gamma_mcmc_samples')

plt.show()


# ### Normalized Posteriors
# 
# Re-normalize for cutoff Poisson distributions

import inspect
print(inspect.getsource(gamma_posterior))


ALPHA = 0.6
LW = 3

''' Plot the modeled distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)
plt.plot(x, y_goal_for, label=r'$P(\mathrm{goal\;for}\;|\;X)$', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'$P(\mathrm{goal\;against}\;|\;X)$', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'$P(\mathrm{no\;goal}\;|\;X)$', color='orange', lw=LW)

plt.ylabel('Posterior probability')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')
plt.legend()

savefig(plt, 'time_since_normed_poisson')

plt.show()


# ### Interpretation

# def convert_to_time_remaining(x):
#     _x = 20 - x
#     t = datetime.timedelta(seconds=_x*60)
#     return str(t)

# convert_to_time_remaining(x[np.argmax(y_goal_for)])


def convert_to_time(x):
    t = datetime.timedelta(seconds=x*60)
    return str(t)


print('Time of max posterior probability =\n'
      f'{x[np.argmax(y_goal_for)], x[np.argmax(y_goal_against)], x[np.argmax(y_no_goal)]}')

print()

t_remaining = [convert_to_time(x[np.argmax(y_goal_for)]),
              convert_to_time(x[np.argmax(y_goal_against)]),
              convert_to_time(x[np.argmax(y_no_goal)])]

print(f'Time of max posterior probability =\n{t_remaining}')


# Great, now we have properly normalized probabilties.
# 
# #### TODO: update notes below
# 
# Notes:
#  - From normalizing factors, we can see ~12% chance of scoring when pulling the goalie on average.
#  - Probability of scoring peaks at 18.55 mins (1:27 remaining), with other probabilties following close after (01:20 for goal against and 01:07 for no goals)
#  
# From now on we'll work from the distributions as our source of truth. These are hard coded below to help with reproducibility.

# TODO: remove this cell (or update it)

# model_normlizing_factors = [
#     0.1292882,
#     0.26528024,
#     0.62489297,
# ]

# mu_mcmc = [
#     1113.8279468130681,
#     1120.1830172722719,
#     1133.9420018554083
# ]


# ### Cumulative sum
# 
# Calculating the CDF will allow us to make some interesting observations on the results.

x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

plt.plot(x, np.cumsum(y_goal_for), label=r'$cumsum [ P(\mathrm{goal\;for}\;|\;X) ]$', color='green', lw=LW)
plt.plot(x, np.cumsum(y_goal_against), label=r'$cumsum [ P(\mathrm{goal\;against}\;|\;X) ]$', color='red', lw=LW)
plt.plot(x, np.cumsum(y_no_goal), label=r'$cumsum [ P(\mathrm{no\;goal}\;|\;X) ]$', color='orange', lw=LW)

plt.ylabel('Posterior CDF')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')
plt.legend()

ax = plt.gca()
ax.yaxis.tick_right()

savefig(plt, 'time_since_gamma_cdf')

plt.show()


# The end of game values have been normalized sum up to one, but this ratio changes over time. We can visualize this with the risk-reward ratio (see below).
# 
# ### Re-normalize
# 
# To better compare these probability distributions, we can normalize each bin to 1 using a function $c(t)$, as follows:
# 
# $$
# c(t) \cdot \big[ P(goal\;for; t) + (P(goal\;against; t) + P(no\;goal; t)\big] = 1 \\
# \vdots \\
# c(t) = \big[ P(goal\;for; t) + (P(goal\;against; t) + P(no\;goal; t)\big]^{-1}
# $$
# 
# This will allow us to re-weight the posteriors later, so we can compare them better and yield a different interpretation.
# 
# Essentially, we'll be able to interpret the resulting distribution as the chance of each outcome at time $t$. This stands in contrast to the probability distributions above, where the total area under the curves sum to 1.

alpha = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)


plt.plot(x, alpha, lw=LW)
plt.ylabel('$c(t)$')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')

# savefig(plt, 'time_elapsed_poisson_cdf')

plt.show()


from scipy.stats import poisson
ALPHA = 0.6
LW = 3 

''' Plot the poisson distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

# Alpha has same shape as x, y above
alpha = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)

y_goal_for = alpha * y_goal_for
y_goal_against = alpha * y_goal_against
y_no_goal = alpha * y_no_goal
plt.plot(x, y_goal_for, label=r'$c \cdot P(\mathrm{goal\;for}\;|\;X)$', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'$c \cdot P(\mathrm{goal\;against}\;|\;X)$', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'$c \cdot P(\mathrm{no\;goal}\;|\;X)$', color='orange', lw=LW)

plt.ylabel('Chance of outcome')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')
plt.legend()

# Plotting below with error bar
# savefig(plt, 'time_elapsed_outcome_chance_timeseries')

plt.show()


# ### Adding error bars

# We can show this uncertainty visually using error bars. Starting with the parameter ($\alpha$ and $\beta$) MCMC samples...

plt.hist(burned_trace['alpha_goal_for'])
plt.hist(burned_trace['alpha_goal_against'])
plt.hist(burned_trace['alpha_no_goal'])


plt.hist(burned_trace['beta_goal_for'])
plt.hist(burned_trace['beta_goal_against'])
plt.hist(burned_trace['beta_no_goal'])


# We can use the uncertainty on $\alpha$ and $\beta$ to calculate that for $P$:
# 
# $$
# \sigma_P = \sqrt{ \big [ \frac{\partial P}{\partial \alpha} \;\sigma_{\alpha} \big ]^{2} + \big [ \frac{\partial P}{\partial \beta} \;\sigma_{\beta} \big ]^{2} }
# $$
# 
# where $\sigma_{\mu}$ is the standard deviation of the $\mu$ samples.

alpha_mcmc_std = [
    burned_trace['alpha_goal_for'].std(),
    burned_trace['alpha_goal_against'].std(),
    burned_trace['alpha_no_goal'].std(),
]


beta_mcmc_std = [
    burned_trace['beta_goal_for'].std(),
    burned_trace['beta_goal_against'].std(),
    burned_trace['beta_no_goal'].std(),
]


alpha_mcmc_std


beta_mcmc_std


model_normalizing_factors


from scipy.misc import derivative
from scipy.stats import gamma
from tqdm import tqdm_notebook

def calc_posteror_error(alpha, beta, alpha_std, beta_std, step=1e-8):
    x = gamma_posterior()[0] * 60 # convert back into seconds (discrete)
    err = np.sqrt(
        alpha_std * np.power(np.array([
                        derivative(lambda _alpha: gamma.pdf(int(t), _alpha, scale=1/beta),
                                    alpha, dx=step)
                        for t in tqdm_notebook(x)
                    ]), 2)
        + beta_std * np.power(np.array([
                        derivative(lambda _beta: gamma.pdf(int(t), alpha, scale=1/_beta),
                                    beta, dx=step)
                        for t in tqdm_notebook(x)
                    ]), 2)
    )
    # Flip error due to gamma transformation
    err = err[::-1]
    return err


ALPHA = 0.6
ALPHA_LIGHT = 0.3
LW = 3

''' Plot the poisson distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

# Alpha has same shape as x, y above
c = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)

y_goal_for = c * y_goal_for
y_goal_against = c * y_goal_against
y_no_goal = c * y_no_goal
plt.plot(x, y_goal_for, label=r'goal for', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'goal against', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'no goal', color='orange', lw=LW)

''' Plot the errors '''
err_p_goal_for = c * calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0])
err_p_goal_against = c * calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1])
err_p_no_goal = c * calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2])
plt.fill_between(x, y_goal_for-err_p_goal_for, y_goal_for+err_p_goal_for,
                 color='green', alpha=ALPHA_LIGHT)
plt.fill_between(x, y_goal_against-err_p_goal_against, y_goal_against+err_p_goal_against,
                 color='red', alpha=ALPHA_LIGHT)
plt.fill_between(x, y_no_goal-err_p_no_goal, y_no_goal+err_p_no_goal,
                 color='orange', alpha=ALPHA_LIGHT)

plt.ylabel('Chance of outcome')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')
plt.xlim(0, 5)
plt.ylim(0, 1)
plt.legend()

savefig(plt, 'time_since_outcome_chances_gamma_full')

plt.show()


ALPHA = 0.6
ALPHA_LIGHT = 0.3
LW = 3

''' Plot the poisson distributions '''
x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

# Alpha has same shape as x, y above
c = np.power(
    np.sum([y_goal_for, y_goal_against, y_no_goal], axis=0),
    -1
)

y_goal_for = c * y_goal_for
y_goal_against = c * y_goal_against
y_no_goal = c * y_no_goal
plt.plot(x, y_goal_for, label=r'goal for', color='green', lw=LW)
plt.plot(x, y_goal_against, label=r'goal against', color='red', lw=LW)
plt.plot(x, y_no_goal, label=r'no goal', color='orange', lw=LW)

''' Plot the errors '''
err_p_goal_for = c * calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0])
err_p_goal_against = c * calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1])
err_p_no_goal = c * calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2])
plt.fill_between(x, y_goal_for-err_p_goal_for, y_goal_for+err_p_goal_for,
                 color='green', alpha=ALPHA_LIGHT)
plt.fill_between(x, y_goal_against-err_p_goal_against, y_goal_against+err_p_goal_against,
                 color='red', alpha=ALPHA_LIGHT)
plt.fill_between(x, y_no_goal-err_p_no_goal, y_no_goal+err_p_no_goal,
                 color='orange', alpha=ALPHA_LIGHT)

plt.ylabel('Chance of outcome')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')
plt.xlim(0, 3)
plt.ylim(0, 1)
plt.legend()

savefig(plt, 'time_since_outcome_chances_gamma')

plt.show()


# ### Odds of scoring a goal
# Let's go into odds-space and look at the chance of scoring a goal, compared to either outcome. We want to maximze this.
# 
# We can use the odds of scoring if the goalie is not pulled (1) as a reference point.

ALPHA = 0.6
ALPHA_LIGHT = 0.3
LW = 3

''' Odds ratio '''

x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

odds_goal_for = y_goal_for / (y_goal_against + y_no_goal)

''' Error bars '''

err_p_goal_for = calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0])
err_p_goal_against = calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1])
err_p_no_goal = calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2])
err_odds_goal_for = (
    np.power(err_p_goal_for / y_goal_for, 2)
    + np.power(err_p_goal_against / y_goal_against, 2)
    + np.power(err_p_no_goal / y_no_goal, 2)
)
err_odds_goal_for = odds_goal_for * np.sqrt(err_odds_goal_for)

''' Plots '''

plt.plot(x, odds_goal_for,
         label=r'$odds(\;\frac{\mathrm{goal\;for}}{\mathrm{loss}}\;)$',
         color='green', lw=LW, alpha=ALPHA)
plt.fill_between(x, odds_goal_for-err_odds_goal_for, odds_goal_for+err_odds_goal_for,
                 color='green', lw=LW, alpha=ALPHA_LIGHT)
# plt.axhline(1, color='black', lw=LW, alpha=ALPHA,
#             label=r'$odds(\;\mathrm{insert\;val}\;)$')

plt.ylabel('Odds of success')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')

plt.xlim(0, 5)
plt.ylim(0, 1.5)

plt.legend()

savefig(plt, 'time_since_gamma_odds_goal_for_full')

plt.show()


ALPHA = 0.6
ALPHA_LIGHT = 0.3
LW = 3

''' Odds ratio '''

x, y_goal_for, y_goal_against, y_no_goal = gamma_posterior(
    alpha_mcmc,
    beta_mcmc,
    norm_factors=model_normalizing_factors
)

odds_goal_for = y_goal_for / (y_goal_against + y_no_goal)

''' Error bars '''

err_p_goal_for = calc_posteror_error(alpha_mcmc[0], beta_mcmc[0], alpha_mcmc_std[0], beta_mcmc_std[0])
err_p_goal_against = calc_posteror_error(alpha_mcmc[1], beta_mcmc[1], alpha_mcmc_std[1], beta_mcmc_std[1])
err_p_no_goal = calc_posteror_error(alpha_mcmc[2], beta_mcmc[2], alpha_mcmc_std[2], beta_mcmc_std[2])
err_odds_goal_for = (
    np.power(err_p_goal_for / y_goal_for, 2)
    + np.power(err_p_goal_against / y_goal_against, 2)
    + np.power(err_p_no_goal / y_no_goal, 2)
)
err_odds_goal_for = odds_goal_for * np.sqrt(err_odds_goal_for)

''' Plots '''

plt.plot(x, odds_goal_for,
         label=r'$odds(\;\frac{\mathrm{goal\;for}}{\mathrm{loss}}\;)$',
         color='green', lw=LW, alpha=ALPHA)
plt.fill_between(x, odds_goal_for-err_odds_goal_for, odds_goal_for+err_odds_goal_for,
                 color='green', lw=LW, alpha=ALPHA_LIGHT)

plt.ylabel('Odds of success')
# plt.yticks([])
plt.xlabel('Time since goalie pull (minutes)')

plt.xlim(0, 3)
plt.ylim(0, 1.5)

plt.legend()

savefig(plt, 'time_since_gamma_odds_goal_for')

plt.show()


(odds_goal_for-err_odds_goal_for)[~np.isnan(odds_goal_for-err_odds_goal_for)].max()


from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;padding-top:50px;padding-bottom:50px}</style>')

