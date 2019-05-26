
# coding: utf-8

from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;}</style>')


get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas')


# # Goalie Pull Bayes Optimize
# 
#  - Exploratory analysis

# ## Explore Parsed Goalie Pull Data

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
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set() # Revert to matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelpad'] = 10
plt.style.use('ggplot')

def savefig(name):
    plt.savefig(f'../../figures/{name}.png', bbox_inches='tight', dpi=300)


ls ../../data/processed/


ls ../../data/processed/pkl


files = sorted(glob.glob('../../data/processed/pkl/*'))
files


def load_data():
    files = [
        '../../data/processed/pkl/20032004_goalie_pulls_2019-03-01.pkl',
 '../../data/processed/pkl/20052006_goalie_pulls_2019-03-01.pkl',
 '../../data/processed/pkl/20062007_goalie_pulls_2019-03-01.pkl',
 '../../data/processed/pkl/20072008_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20082009_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20092010_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20102011_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20112012_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20122013_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20132014_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20142015_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20152016_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20162017_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20172018_goalie_pulls_2019-04-25.pkl',
 '../../data/processed/pkl/20182019_goalie_pulls_2019-04-25.pkl',
            ]
    return pd.concat((pd.read_pickle(f) for f in files), sort=False)

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


df.head()


df.tail()


df.describe().T


df.dtypes


df.isnull().sum() / df.shape[0]


df.columns


plt.ylabel('Goalie Pulls')
plt.yticks([])
df.date.hist(
    color='b',
    bins=500,
    histtype='stepfilled')
savefig('goalie_pulls_2003-2019')


df.groupby('season').size().sort_index(ascending=True).rename('counts').reset_index()


fig, ax = plt.subplots()

# Calculate number of pulls per season
s = df.groupby('season').size().sort_index(ascending=True).rename('counts').reset_index()

# Add data for the missing season
s = (s.append({'season': '20042005', 'counts': 0}, ignore_index=True)
    .sort_values('season', ascending=True).reset_index(drop=True))

s.plot(marker='o', lw=0, ax=ax, color='b', ms=10)
ax.set_ylim(600, 1000)
plt.legend([])
plt.ylabel('Total Goalie Pulls')

# Assign tick names
label_map = {str(i): s for i, s in enumerate(s.season.tolist())}
fig.canvas.draw()
labels = [lab.get_text() for lab in ax.get_xticklabels()]
ax.set_xticklabels([label_map.get(lab, '') for lab in labels])

savefig('goalie_pulls_by_season')


# Plot goalie pulls per season

games_per_season = {}
for folder in sorted(glob.glob('../../data/raw/html/*')):
    files = glob.glob(os.path.join(folder, '*.html'))
    print(folder, len(files))
    games_per_season[os.path.split(folder)[-1]] = len(files)


s['counts_per_game'] = s.season.map(games_per_season)

fig, ax = plt.subplots()

# Calculate number of pulls per season
s = df.groupby('season').size().sort_index(ascending=True).rename('counts').reset_index()

# Add data for the missing season
s = (s.append({'season': '20042005', 'counts': 0}, ignore_index=True)
    .sort_values('season', ascending=True).reset_index(drop=True))

# Convert to counts per game
s['games'] = s['season'].apply(lambda x: games_per_season.get(x, 0))
s['counts'] = (s['counts'] / s['games']).fillna(0)

s.plot(marker='o', lw=0, ax=ax, color='b', ms=10)
ax.set_ylim(0.5, 0.8)
plt.legend([])
plt.ylabel('Average Goalie Pulls Per Game')

# Assign tick names
label_map = {str(i): s for i, s in enumerate(s.season.tolist())}
fig.canvas.draw()
labels = [lab.get_text() for lab in ax.get_xticklabels()]
ax.set_xticklabels([label_map.get(lab, '') for lab in labels])

savefig('goalie_pulls_per_game_by_season')


fig, ax = plt.subplots()
iterables = zip(['orange', 'red', 'green'],
                ['no_goals', 'goal_against', 'goal_for'])

axes = []
for c, label in iterables:
    m = df.label==label
    
    # Calculate the counts
    s = df[m].groupby('season').size().sort_index(ascending=True).rename(label).reset_index()
    
    # Add data for the missing season
    s = (s.append({'season': '20042005', label: -999}, ignore_index=True)
        .sort_values('season', ascending=True).reset_index(drop=True))
    
    s.loc[s.season == '20122013', label] = -999
    s.plot(marker='o', lw=0, ax=ax, ms=10, color=c, label=label)
    plt.legend()

# ax.set_xticklabels(s.season.tolist());
ax.set_ylim(0, 550)
plt.ylabel('Total Counts')

# Assign tick names
label_map = {str(i): s for i, s in enumerate(s.season.tolist())}
fig.canvas.draw()
labels = [lab.get_text() for lab in ax.get_xticklabels()]
ax.set_xticklabels([label_map.get(lab, '') for lab in labels])

savefig('goalie_pull_outcomes_by_season')


# Plot average pull time by season

df['pull_time_remaining'] = (
    df_['pull_time']
    .apply(lambda x: datetime.timedelta(seconds=60*20) - x)
    .astype('timedelta64[s]')
) / 60
sns.boxplot(x='season', y='pull_time_remaining', data=df, color='b')
plt.ylabel('Time Remaining when Goalie Pulled (minutes)')
plt.xlabel('Season')
plt.xticks(rotation=45)
plt.ylim(-0.1, 4)
savefig('goalie_pull_times_by_season')


df.groupby('season').pull_time_remaining.mean()


col = 'pull_time'
(df[col].astype('timedelta64[s]') / 60)    .plot.hist(bins=100,
               color='b',
               histtype='stepfilled')
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.yticks([])
plt.xlim(14, 20)
savefig('goalie_pull_game_times_hist')


# We're interested in knowing about the outcome, given the pull time. This way we can look at the odds of scoring as a function of game time elapsed.

df.head()


# ax = plt.subplot(111)
# ax.set_prop_cycle(color=['red', 'green', 'orange'])

df['pull_time_seconds'] = df['pull_time'].astype('timedelta64[s]') / 60

iterables = zip(['orange', 'red', 'green'],
                ['no_goals', 'goal_against', 'goal_for'])

for c, label in iterables:
    (df[df.label==label]['pull_time_seconds']
         .plot.hist(bins=60,
                    alpha=0.5,
                    color=c,
                    histtype='stepfilled',
                    label=label))

plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.yticks([])
plt.xlim(14, 20)
plt.legend()

savefig('goalie_pull_outcomes_game_times_hist')
del df['pull_time_seconds']


cols = ['goal_for_time', 'goal_against_time']
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=100,
               alpha=0.5,
               color=['green', 'red'],
               histtype='stepfilled')
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.yticks([])
plt.xlim(14, 20)
savefig('5_on_6_goals')


cols = ['goal_for_time', 'goal_against_time']
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=100,
               alpha=0.5,
               density='normed',
               color=['green', 'red'],
               histtype='stepfilled')
plt.xlabel('Time elapsed in 3rd period (minutes)')
plt.ylabel('Frequency (normed)')
plt.yticks([])
plt.xlim(14, 20)
savefig('5_on_6_goals_normed')


print('Number of goals found:')
(~df[['goal_for_time', 'goal_against_time']].isnull()).sum()


print('Total goals found:')
(~df[['goal_for_time', 'goal_against_time']].isnull()).sum().sum()


# We also want to model the time between goalie pull and goal (i.e. the timedelta).

cols = ['game_end_timedelta', 'goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               color=['blue', 'red','green'],
               histtype='stepfilled')
plt.xlabel('Time since goalie pull (minutes)')
plt.yticks([])
# savefig('5_on_6_goalie_pull_outcomes')


cols = ['goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               color=['red', 'green',],
               histtype='stepfilled')
plt.xlabel('Time since goalie pull (minutes)')
plt.yticks([])
# savefig('5_on_6_goalie_pull_goal_timedeltas')


cols = ['goal_against_timedelta', 'goal_for_timedelta', ]
(df[cols].astype('timedelta64[s]') / 60)    .plot.hist(bins=50, alpha=0.5,
               density='normed',
               color=['red', 'green'],
               histtype='stepfilled')
plt.xlabel('Time since goalie pull (minutes)')
plt.yticks([])
# savefig('5_on_6_goalie_pull_goal_timedeltas_normed')


# The mean/median number of seconds until a goal (after pulling the goalie)

(df[cols].astype('timedelta64[s]')).mean()


(df[cols].astype('timedelta64[s]')).median()


(df['game_end_timedelta'].astype('timedelta64[s]') / 60).plot.hist(bins=50, color='b', histtype='stepfilled')
plt.xlabel('Time since goalie pull (minutes)')
plt.yticks([])
# savefig('5_on_6_game_end_timedeltas')


# ### Rough work

# ## Bugs

# ### *2019-02-01*

df = load_data()


# Any non 3rd period pulls?

df[df['pull_period'] != 3]


# Any bugs not in the last 15 minutes?

mask = df.pull_time < datetime.timedelta(seconds=15*60)
df[mask].shape[0]


# I noticed some goal for timedelats less than 0.. which makes no sense. I'll have to look into that

mask = df.goal_for_time < datetime.timedelta(seconds=15*60)
df[mask]


# This game is weird... http://www.nhl.com/scores/htmlreports/20052006/PL020591.HTM
# 
# We'll drop this point before modeling.

df[df.goal_for_timedelta < datetime.timedelta(0)]


# http://www.nhl.com/scores/htmlreports/20032004/PL020907.HTM

df.game_end_time.astype('timedelta64[s]').plot.hist()


# Games should end at 20 mins. Let's throw out the early times (this must be overtime or something).

mask = df.game_end_time < datetime.timedelta(seconds=60*20)
mask.sum(), df.shape[0]


mask = df.game_end_time < datetime.timedelta(seconds=60*15)
mask.sum(), df.shape[0]


# Obviously the game will end at 20:00, this column corresponds to the last row parsed.

# *2019-04-25*

df = load_data()


# Any non 3rd period pulls?

df[df['pull_period'] != 3]


df.pull_period.value_counts().sort_index()


# None of these make sense.. we'll want to drop them

# Any bugs not in the last 15 minutes?

mask = df.pull_time < datetime.timedelta(seconds=15*60)
df[mask].shape[0]


df[mask]


# I noticed some goal for timedelats less than 0.. which makes no sense. I'll have to look into that

mask = df.goal_for_time < datetime.timedelta(seconds=15*60)
df[mask]


# We'll have to drop these as well

df[df.goal_for_timedelta < datetime.timedelta(0)]


df.game_end_time.astype('timedelta64[s]').plot.hist()


# Games should end at 20 mins. Let's throw out the early times (this must be overtime or something).

# Obviously the game will end at 20:00, this column corresponds to the last row parsed.

# Let's look at the spike in goalie pull times around 10 minutes

mask = (df.pull_time <= datetime.timedelta(seconds=15*60))        & (df.pull_time >= datetime.timedelta(seconds=5*60))
df[mask].pull_time.astype('timedelta64[s]').plot.hist()


df[df.pull_time == datetime.timedelta(seconds=10*60)]


# ### *2019-04-26*

plt.ylabel('Goalie Pulls')
plt.yticks([])
df.date.hist(
    color='b',
    bins=100,
    histtype='stepfilled')
# savefig(plt, 'goalie_pulls_2003-2019')


df.groupby('season').size()


# Too few pulls from 2007-2009. I wonder how many HTML files I have for each

get_ipython().system('ls ../../data/raw/html/')


import os, glob
def nested_count(folder_path, level=0):
    for folder in sorted(glob.glob('{}/'.format(os.path.join(folder_path, '*')))):
        print('{:}{:}: {:,}'.format('    '*level, os.path.split(os.path.split(folder)[-2])[-1], len(glob.glob(os.path.join(folder, '*')))))
        nested_count(folder, level+1)
nested_count('../../data/')


# I will need to look at the parsing for 2007-2009. Something is up

df_ = df[(df.season == '20072008') | (df.season == '20082009')]


len(df_)


df_.label.value_counts()


# It turns out that some tables just aren't being parsed properly... I need to use html.parser instead of lxml in some cases

# ### *2019-04-30*

fig, ax = plt.subplots()
s = df.groupby('season').size().sort_index(ascending=True).rename('counts').reset_index()
ax = s.plot(marker='o', lw=0, ax=ax, color='b', ms=10)
ax.set_xticklabels(s.season.tolist());
# ax.set_ylim(600, 1100)
plt.legend([])
plt.ylabel('Total Goalie Pulls')
# savefig('goalie_pulls_by_season')


label_map


xticks


fig, ax = plt.subplots()
iterables = zip(['orange', 'red', 'green'],
                ['no_goals', 'goal_against', 'goal_for'])

axes = []
for c, label in iterables:
    m = df.label==label
#     m = m & (df.season != '20122013')
    df_ = df[m].copy()
    df_.loc[(df.season == '20122013'), label] = 0
    s = df_.groupby('season').size().sort_index(ascending=True).rename(label).reset_index()
    s.plot(marker='o', lw=0, ax=ax, ms=10, color=c, label=label)
    plt.legend()

    
ax = plt.gca()
ax.set_ylim(0, 600)
ax.set_xlim(0, 15)

label_map = {str(i): season for i, season in enumerate(df.season.drop_duplicates().sort_values(ascending=True).tolist())}
xticks = [label_map.get(str(round(tick)), '') for tick in ax.get_xticks().tolist()]
ax.set_xticklabels(xticks)
plt.ylabel('Goalie Pulls')

# savefig('goalie_pull_outcomes_by_season')





# Something is up with 2009, where I don't seem to collect enough data.

# The issue is my labels. That should be the 2012/2013 season, which was shortened by a lockout

ax


[t.get_text() for t in ax.get_xticklabels()]


for tick in ax.get_xticklabels():
    tick.set_text(label_map.get(tick.get_text(), ''))


tick = ax.get_xticklabels()[0]


tick.set_text('')


label_map


label_map = {str(i): season for i, season in enumerate(df.groupby('season').size().sort_index(ascending=True).index.tolist())}
# xticklabels = [label_map.get(t.get_text(), '') for t in ax.get_xticklabels()]
# ax.set_xticklabels(xticklabels)


# print([t.get_text() for t in ax.get_xticklabels()])

# ticks = [t.get_text() for t in ax.get_xticklabels()]
# ax.set_xticklabels(ticks)


fig, ax = plt.subplots()
iterables = zip(['orange', 'red', 'green'],
                ['no_goals', 'goal_against', 'goal_for'])

axes = []
for c, label in iterables:
    m = df.label==label
    s = df[m].groupby('season').size().sort_index(ascending=True).rename(label).reset_index()
    axes.append(s.plot(marker='o', lw=0, ax=ax, ms=10, color=c, label=label))
    plt.legend()

ax.set_xticklabels(s.season.tolist());
ax.set_ylim(0, 600)
plt.ylabel('Total Counts')
# savefig('goalie_pull_outcomes_by_season')


# fig, ax = plt.subplots()
# iterables = zip(['orange', 'red', 'green'],
#                 ['no_goals', 'goal_against', 'goal_for'])

# axes = []
# for c, label in iterables:
#     m = df.label==label
#     s = df[m].groupby('season').size().sort_index(ascending=True).rename(label).reset_index()
#     s.plot(marker='o', lw=0, ax=ax, ms=10, color=c, label=label)
#     plt.legend()



# ax.set_ylim(0, 600)
label_map = {str(i): season for i, season in enumerate(df.groupby('season').size().sort_index(ascending=True).index.tolist())}
# xticklabels = [label_map.get(t.get_text(), '') for t in ax.get_xticklabels()]
# ax.set_xticklabels(xticklabels)


# print([t.get_text() for t in ax.get_xticklabels()])

# ticks = [t.get_text() for t in ax.get_xticklabels()]
# ax.set_xticklabels(ticks)
# plt.ylabel('Total Counts')
# # savefig('goalie_pull_outcomes_by_season')


label_map


[t.get_text() for t in ax.get_xticklabels()]


ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()])

