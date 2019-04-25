#!/usr/bin/env python
# coding: utf-8

from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;}</style>')


get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas')


# # Goalie Pull Bayes Optimize
# 
#  - Parse the goalie pull stats we need
#  
# ### This script is for the legacy format: *2006/2007* and earlier

# ## Parse HTML Stats Table

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import re
import datetime
import time
from collections import OrderedDict
import glob
from tqdm import tqdm_notebook
from colorama import Fore, Style


# ### Helper functions

def is_int(x):
    try:
        int(x)
        return True
    except:
        return False

def parse_time(s):
    _s = s.copy()
    _s = _s.apply(lambda x:
        datetime.datetime.strptime(x, '%M:%S')
    )
    _s = _s.apply(lambda x:
        datetime.timedelta(
        hours=x.hour,
        minutes=x.minute,
        seconds=x.second,
        microseconds=x.microsecond
    ))
    return _s

def parse_date(s):
    _s = s.copy()
    
    # invalid parsing will be set as NaT
    _s = pd.to_datetime(_s, errors='coerce')

    '''
     _s = _s.apply(lambda x: (re.search(',(.*)', x.text.splitlines()[2])
                             .group(1)
                             .strip()))
     _s = _s.apply(lambda x: datetime.datetime\
                             .strptime(x.strip(), '%B %d, %Y'))
    '''
    
    return _s

def goalie_pull_timedelta(s1, s2):
    out = []
    for d1, d2 in zip(s1.values, s2.values):
        try:
            o = d1 - d2
        except:
            o = float('nan')
        out.append(o)
    return out


# ### Main Functions

def get_game_df(
    soup,
    local_html=True,
) -> pd.DataFrame:
    '''
    Returns the parsed HTML stats table for some game and returns
    a DataFrame.
    
    local_html : bool
        Set True if using local HTML file. Otherwise set False
        for requested HTML from nhl.com
    '''
    
    # Parse main table
    table = soup.find_all('pre')[0]
    raw_table = table.text

    if local_html:
        lines = [line for line in
             raw_table.split('\r\n')[0].split('\n')
             if line.strip()]
    else:
        lines = [line for line in
             raw_table.split('\r\n')
             if line.strip()]

    # Get row split indices
    j = 0
    switch = False
    idx = []
    for i, char in enumerate(lines[1]):
        if char == ' ':
            switch = True
        elif (char == '-') and switch:
            switch = False
            idx.append([j, i])
            j = i
    idx.append([idx[-1][1], None])
    idx = [slice(*x) for x in idx]
    
    # Split data and load to df
    data = []
    for i, line in enumerate(lines):
        if i == 1:
            continue
        d = np.empty((7,), dtype=object) 
        for j, _slice in enumerate(idx):
            d[j] = line[_slice].strip()
        data.append(d)
    
    df = pd.DataFrame(data[1:], columns=[col.strip().lower()
                                         for col in data[0]])
    
    # Check for missing columns
    for col in ['#', 'per', 'time', 'event', 'team', 'type', 'description']:
        if col not in df.columns:
            print(f'Missing column: {col}')
    
    # Basic clean up
    players_on_ice = {}
    prev_num = None
    for i, row in df.iterrows():
        if not is_int(row['#']):
            if ((prev_num not in players_on_ice.keys()
                and (row is not None))
                ):
                players_on_ice[prev_num] = ''
            players_on_ice[prev_num] += ''.join(row.values)
            continue
        prev_num = row['#']
    
    good_rows = df['#'].apply(is_int)
    df = df.loc[good_rows]
    df['#'] = df['#'].astype(int)
    df['per'] = df['per'].astype(int)
    df['time'] = parse_time(df['time'])
    
    return df, players_on_ice


def goalie_pull_game_search(
    game_df: pd.DataFrame, 
    players_on_ice: dict,
    verbose=False,
) -> list:
    '''
    Search through a DataFrame stats table for goalie pulls, and
    finds empty net goals / games that end with a goalie pulled.
    
    players_on_ice : dict
        A mapping of the goal row # to the players who scored
        that goal. In the HTML stats table, these are assumed
        to be the two rows following the goal.
    '''
    out = []
    pull_threshold = datetime.timedelta(minutes=15)

    o = {}
    on_ice_for_goal = ''
    pull_switch = False
    goal_switch = False

    i_final_row = game_df.index[-1]

    for i, row in game_df.iterrows():
        if verbose:
            print(f'Looking at row {i}, # = {row["#"]}')
        if 'pull' in row.description.lower():
            print(f'Found goalie pull')

            # Filter on pull events in the last 10 minutes
            if (row.time < pull_threshold
               or row.per != 3):
                print('Too early in game, not counting it')
                continue

            # Append the new pull data
            pull_switch = True
            o = {}
            o['team_name'] = row.team
            o['pull_period'] = row.per
            o['pull_time'] = row.time
            pull_goalie = (re.search(r'([^\d,]+)', row.description)
                            .group(1)
                            .strip()
                            .lower())
            pull_team = row.team
            if verbose:
                print(f'team, period, time = \n{o}')
                print(f'pull goalie = {pull_goalie}, pull team = {pull_team}')

        if i == i_final_row:
            # Check if game ended 5 on 6
            if o:
                print(Fore.RED + f'Game end with no goalie in net!' + Style.RESET_ALL)
                game_end_time = row.time
                if game_end_time > pull_threshold:
                    # Only count if game_end_time is expected.
                    # This can be lower than pull_threshold if
                    # the game goes to overtime.
                    out.append(o.copy())

        # Search for an event (goal for / against / end of game)
        if pull_switch:
            print('Searching for a goal or end of game')
            if row.event.lower() == 'goal':
                print(f'Found goal after a pull, setting goal against time as {row.time}')
                if row.team == pull_team:
                    o['goal_for_time'] = row.time
                else:
                    o['goal_against_time'] = row.time

                if verbose:
                    print(f'Checking if {pull_goalie} was on the '
                          f'ice ({players_on_ice[str(row["#"])].lower()})')                  
                  
                if row.time == o['pull_time']:
                    print(f'Goal time equal to pull time {row.time}. '
                          'Ignoring it and searching for .')
                    continue
                  
                if pull_goalie in players_on_ice[str(row['#'])].lower():
                    print('Goalie was back in the net')
                    pass
                else:
                    # It's a true empty net goal
                    print(Fore.RED + f'Found empty net goal:\n{o}' + Style.RESET_ALL)
                    out.append(o.copy())
                o = {}
                pull_switch = False

    return out

                  
def get_game_meta(soup) -> dict:
    '''
    Get game metadata like the date and game number.
    '''
    try:
        center_el = soup.find_all('center')[1].text
        game_number = (center_el.split('\n')[1]
                         .lower()
                         .replace('game', '').replace(',', '')
                         .strip().lstrip('0'))
        date = (center_el.split('\n')[2].strip())
        return {'game_number': game_number, 'date': date}
    except:
        print('Failed to get meta data :(')
        return {}


def parse_game(
    soup: BeautifulSoup,
    cols: list,
    season='',
    verbose=False
) -> list:
    '''
    Search for empty net goals for, against and game ends,
    for a given game. Input the HTML as a BeautifulSoup object.
    '''
    out = []

    # Get the game stats df    
    game_df, players_on_ice = get_game_df(soup)
    
    # Get the game metadata
    meta_data = get_game_meta(soup)
    if season:
        meta_data['season'] = season
                  
    # Check for goalie pulls
    goalie_pull_idx = game_df.description.str.lower().str.contains('pull')
    if not goalie_pull_idx.sum():
        return []    
    
    # For each goalie pull, determine the outcome
    game_5on6_data = goalie_pull_game_search(
        game_df, players_on_ice, verbose=verbose
    )
    
    if not game_5on6_data:
        return []
    
    out = []
    for row in game_5on6_data:
        row.update(meta_data)
        out.append([row.get(col, float('nan')) for col in cols])

    return out


def make_final_df(
    data: list,
    cols: list,
) -> pd.DataFrame:
    '''
    Return table with columns:
     - season
     - game number
     - team name
     - date
     - period of pull
     - time of pull
     - time of goal for
     - time of goal against
     - time of game end
     - timedelta of goal for (since pull)
     - timedelta of goal against (since pull)
     - timedelta of game end (since pull)
     
    Each row represents a goalie pull event.
    Goalie pull events that don't result in
    one of the above are ignored.
    '''
    df = pd.DataFrame(data, columns=cols)
    df['date'] = parse_date(df.date.fillna('').astype(str))
    df['goal_for_timedelta'] = goalie_pull_timedelta(df.goal_for_time, df.pull_time)
    df['goal_against_timedelta'] = goalie_pull_timedelta(df.goal_against_time, df.pull_time)
    
    game_end = parse_time(pd.Series('20:00', index=df.index))
    df['game_end_timedelta'] = goalie_pull_timedelta(game_end, df.pull_time)
    # Only output game_end_timedelta when no goal was found
    df.loc[~(df.goal_for_time.isnull() & df.goal_against_time.isnull()),
           'game_end_timedelta'] \
        = float('nan')
    
    return df


def parse_game_range(
    seasons: list,
    test: bool = False,
) -> pd.DataFrame:
    '''
    Parse every game for a given season.
    
    Folder structure is:
    ../../data/raw/html/{season}/{game_number}.html
    
    '''
    
    root_data_path = '../../data/raw/html'
    if not os.path.exists(root_data_path):
        print(f'Root data path not found ({root_data_path})')
        return None
    
    cols = [
        'season',
        'game_number',
        'team_name',
        'date',
        'pull_period',
        'pull_time',
        'goal_for_time',
        'goal_against_time',
        'goal_for_timedelta',
        'goal_against_timedelta',
        'game_end_timedelta',
    ]    
    data = []
    for season in seasons:        
        search_string = os.path.join(root_data_path, season, '*.html')
        html_files = glob.glob(search_string)
        print(f'Found {len(html_files)} files')
        i = 0
        for file in tqdm_notebook(html_files):
            i += 1
            if test:
                return_condition = i%20 == 0
                if return_condition: 
                    print(f'Testing mode - stopping script at {return_condition}')
                    return data, make_final_df(data, cols)
            
            print(f'Processing file {file}')
            try:
                with open(file, 'r') as f:
                    page_text = f.read()
                    soup = BeautifulSoup(page_text, 'lxml')
            except Exception as e:
                print(f'Unable to read/parse file {file}')
                print(str(e))
                continue
            
            try:
                d = parse_game(soup, cols, season)
                if not d:
                    continue
            except Exception as e:
                print(f'Unable to parse game for file {file}')
                print(str(e))
                continue

            data += d
            
    df = make_final_df(data, cols)
    return df


def test_parse_game_range(season, game_number):
    '''
    Parse a specific game with verbose output.
    '''
    
    cols = [
        'season',
        'game_number',
        'team_name',
        'date',
        'pull_period',
        'pull_time',
        'goal_for_time',
        'goal_against_time',
        'game_end_time',
        'goal_for_timedelta',
        'goal_against_timedelta',
        'game_end_timedelta',
    ]    
    data = []
    
    file = f'../../data/raw/html/{season}/{game_number}.html'
    print(f'Processing file {file}')
    with open(file, 'r') as f:
        try:
            page_text = f.read()
            soup = BeautifulSoup(page_text, 'lxml')
        except Exception as e:
            print(f'Unable to read/parse file {file}')
            raise e

    try:
        d = parse_game(soup, cols, season, verbose=True)
        if not d:
            print('No return from parse_game')
    except Exception as e:
        print(f'Unable to parse game for file {file}')
        raise e

    data = d
    print(Fore.RED + f'DONE:\ndata =\n{data}' + Style.RESET_ALL)
    df = make_final_df(data, cols)
    return df


test_parse_game_range('20032004', 710)


seasons = ['20032004']

data, df_goalie_pull = parse_game_range(seasons, test=True)


df_goalie_pull


DATE_STR = datetime.datetime.now().strftime('%Y-%m-%d')


season = '20032004'
df_goalie_pull = parse_game_range([season])
df_goalie_pull.to_csv(f'../data/csv/{season}_goalie_pulls_{DATE_STR}.csv', index=False)
df_goalie_pull.to_pickle(f'../data/pkl/{season}_goalie_pulls_{DATE_STR}.pkl')


season = '20052006'
df_goalie_pull = parse_game_range([season])
df_goalie_pull.to_csv(f'../data/csv/{season}_goalie_pulls_{DATE_STR}.csv', index=False)
df_goalie_pull.to_pickle(f'../data/pkl/{season}_goalie_pulls_{DATE_STR}.pkl')

















season = '20062007'
df_goalie_pull = parse_game_range([season])
df_goalie_pull.to_csv(f'../data/csv/{season}_goalie_pulls_{DATE_STR}.csv', index=False)
df_goalie_pull.to_pickle(f'../data/pkl/{season}_goalie_pulls_{DATE_STR}.pkl')











# ### Bugs

test_parse_game_range('20052006', 591)


# This game is just a weird one.. can't really tell what's going on, but it does look like goalie was pulled at the end. The time scale is off.. suggesting the game went to overtime?
# 
# http://www.nhl.com/scores/htmlreports/20052006/PL020591.HTM

# Here's one where there was a negative goal for timedelta. I fixed the bug by setting `o = {}` when a new pull is found. (Now it works as expected)

test_parse_game_range('20032004', 907)







