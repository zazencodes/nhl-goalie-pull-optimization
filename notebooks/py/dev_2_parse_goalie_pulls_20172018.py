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
# ### This script is for the newest format: *2017/2018* and later (something changed this year)

# Sample games:
# 
# http://www.nhl.com/scores/htmlreports/20032004/PL020001.HTM   
# file:///Users/alex/Documents/nhl-goalie-pull-optimization/data/raw/html/20072008/980.html   
# file:///Users/alex/Documents/nhl-goalie-pull-optimization/data/raw/html/20072008/980.html   

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
        datetime.datetime.strptime(x[:5], '%M:%S')
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


# ### Load data sample

with open('../../data/raw/html/20172018/980.html', 'r') as f:
    soup = BeautifulSoup(f.read(), 'lxml')


df = get_game_df(soup)


# Old functions are not going to work for us.. will need to re-create everything here
# 
# ### get_game_df

len(soup.find_all('table'))


table = soup.find_all('table')[0]
raw_table = table.text


table.find_all('tr')[400]


' '.join(table.attrs['class'])


team_names = []
for row_data in soup.find_all('td', {'class': 'heading + bborder'}):
    if 'on ice' in row_data.text.lower():
        team_names.append(row_data.text.split()[0])
    if len(team_names) == 2:
        break

if len(team_names) != 2:
    team_names = ['', '']


team_names


def test():

    data = []

    for row in soup.find_all('tr', {'class': 'evenColor'}):
        d = []
        for i, row_data in enumerate(row.find_all('td')):
            if 'class' in row_data.attrs:
                classes = ' '.join(row_data.attrs['class'])
                if 'bborder' not in classes:
                    continue
            else:
                continue

            if i == 3:
#                 print(row)
                return row_data

            row_data_text = row_data.text.strip()
            if i >= 6:
                row_data_text = ' '.join(re.findall('[a-zA-Z]+', row_data_text))

            d.append(row_data_text)

        if len(d) >= 8:
            data.append(d[:8])
            
el = test()


str(el.contents[0])


data = []

for row in soup.find_all('tr', {'class': 'evenColor'}):
    d = []
    for i, row_data in enumerate(row.find_all('td')):
        if 'class' in row_data.attrs:
            classes = ' '.join(row_data.attrs['class'])
            if 'bborder' not in classes:
                continue
        else:
            continue
            
        row_data_text = row_data.text.strip()
        if i >= 6:
            row_data_text = ' '.join(re.findall('[a-zA-Z]+', row_data_text))
            
        d.append(row_data_text)

    if len(d) >= 8:
        data.append(d[:8])


data[:3]


def parse_time(s):
    _s = s.copy()
    _s = _s.apply(        
        lambda x:
        datetime.datetime.strptime(x.split(' ')[0], '%M:%S')
    )
    _s = _s.apply(lambda x:
        datetime.timedelta(
        hours=x.hour,
        minutes=x.minute,
        seconds=x.second,
        microseconds=x.microsecond
    ))
    return _s

def parse_team(s):
    _s = s.copy()
    _s = _s.str.split().apply(lambda x: x[0])
    return _s


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

    if not local_html:
        raise NotImplementedError('Only local HTML supported')

    # Parse the main table
    data = []
    for row in soup.find_all('tr', {'class': 'evenColor'}):
        d = []
        for i, row_data in enumerate(row.find_all('td')):
            if 'class' in row_data.attrs:
                classes = ' '.join(row_data.attrs['class'])
                if 'bborder' not in classes:
                    continue
            else:
                continue

            if i == 3:
                # Parse the time
                row_data_text = ' '.join([str(s) for s in row_data.contents])

            elif i >= 6:
                # Parse the players on ice
                row_data_text = ' '.join(re.findall('[a-zA-Z]+', row_data.text.strip()))

            else:
                # Standard parse
                row_data_text = row_data.text.strip()
                
            d.append(row_data_text)

        if len(d) >= 8:
            data.append(d[:8])
    
    cols = ['#', 'per', 'type', 'time', 'event', 'description', 'visitor_on_ice', 'home_on_ice']
    df = pd.DataFrame(data, columns=cols)
    
    good_rows = df['#'].apply(is_int)
    df = df.loc[good_rows]
    df['#'] = df['#'].astype(int)
    df['per'] = df['per'].astype(int)
    df['time'] = parse_time(df['time'])
    df['team'] = parse_team(df['description'])
    
    # Get the team info
    team_names = []
    for row_data in soup.find_all('td', {'class': 'heading + bborder'}):
        if 'on ice' in row_data.text.lower():
            team_names.append(row_data.text.split()[0])
        if len(team_names) == 2:
            break
    if len(team_names) != 2:
        team_names = ['', '']
    team_info = {'visitor': team_names[0], 'home': team_names[1]}
    
    return df, team_info


df, team_info = get_game_df(soup)


soup.text[:1000]


# Parse the main table
data = []
for row in soup.find_all('tr', {'class': 'evenColor'}):
    d = []
    for i, row_data in enumerate(row.find_all('td')):
        if 'class' in row_data.attrs:
            classes = ' '.join(row_data.attrs['class'])
            if 'bborder' not in classes:
                continue
        else:
            continue
            
        if i == 3:
            # Parse the time
            row_data_text = ' '.join([str(s) for s in row_data.contents])

        elif i >= 6:
            # Parse the players on ice
            row_data_text = ' '.join(re.findall('[a-zA-Z]+', row_data.text.strip()))

        else:
            # Standard parse
            row_data_text = row_data.text.strip()

        d.append(row_data_text)

    print(d)
        
    if len(d) >= 8:
        data.append(d[:8])


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

    if not local_html:
        raise NotImplementedError('Only local HTML supported')

    # Parse the main table
    data = []
    for row in soup.find_all('tr', {'class': 'evenColor'}):
        d = []
        for i, row_data in enumerate(row.find_all('td')):
            if 'class' in row_data.attrs:
                classes = ' '.join(row_data.attrs['class'])
                if 'bborder' not in classes:
                    continue
            else:
                continue

            if i == 3:
                # Parse the time
                row_data_text = ' '.join([str(s) for s in row_data.contents])

            elif i >= 6:
                # Parse the players on ice
                row_data_text = ' '.join(re.findall('[a-zA-Z]+', row_data.text.strip()))

            else:
                # Standard parse
                row_data_text = row_data.text.strip()
                
            d.append(row_data_text)

        if len(d) >= 8:
            data.append(d[:8])
    
    cols = ['#', 'per', 'type', 'time', 'event', 'description', 'visitor_on_ice', 'home_on_ice']
    df = pd.DataFrame(data, columns=cols)
    
    good_rows = df['#'].apply(is_int)
    df = df.loc[good_rows]
    df['#'] = df['#'].astype(int)
    df['per'] = df['per'].astype(int)
    df['time'] = parse_time(df['time'])
#     df['team'] = parse_team(df['description'])
    
    # Get the team info
    team_names = []
    for row_data in soup.find_all('td', {'class': 'heading + bborder'}):
        if 'on ice' in row_data.text.lower():
            team_names.append(row_data.text.split()[0])
        if len(team_names) == 2:
            break
    if len(team_names) != 2:
        team_names = ['', '']
    team_info = {'visitor': team_names[0], 'home': team_names[1]}
    
    return df, team_info

df, team_info = get_game_df(soup)


(df['description'] + '.').str.split().apply(lambda x: x[0])












































def goalie_pull_game_search(
    game_df: pd.DataFrame,
    team_info: dict,
    verbose=False,
) -> list:
    '''
    Search through a DataFrame stats table for goalie pulls, and
    finds empty net goals / games that end with a goalie pulled.
    
    team_info : dict
        Visitor and away team names.
    '''
    out = []
    pull_threshold = datetime.timedelta(minutes=15)

    o = {}
    prev_row = None
    prev_row_players = ['', '']
    pull_team = ''
    pull_switch = False
    goal_switch = False
    
    i_final_row = game_df.index[-1]

    for i, row in game_df.iterrows():
        if verbose:
            print(f'Looking at row {i}, # = {row["#"]}')
        if prev_row is None:
            prev_row = row.copy()
                 
        visitor_g_pull = 'G' not in row.visitor_on_ice
        home_g_pull = 'G' not in row.home_on_ice
        if visitor_g_pull or home_g_pull:
            if not pull_switch:
                # The goalie was just pulled
                pull_switch = True
                o = {}
                if visitor_g_pull:
                    pull_team = 'visitor'
                elif home_g_pull:
                    pull_team = 'home'
                else:
                    raise ValueError('Home or away team must have goalie pulled')
                o['team_name'] = team_info.get(pull_team, '')
                o['pull_period'] = row.per
                o['pull_time'] = (row.time + prev_row.time) / 2
                if verbose:
                    print(f'team, period, time = \n{o}')

        if i == i_final_row:
            # Check if game ended 5 on 6
            if o:
                print(Fore.RED + f'Game end with no goalie in net!' + Style.RESET_ALL)
                o['game_end_time'] = row.time
                out.append(o.copy())
                  
        # Search for an event (goal for / against / end of game)
        if pull_switch:
            print('Searching for a goal or end of game')
            if row.event.lower() == 'goal':
                print(f'Found goal after a pull, setting goal against time as {row.time}')
                if row.team == o['team_name']:
                    o['goal_for_time'] = row.time
                else:
                    o['goal_against_time'] = row.time
                  
                if row.time == o['pull_time']:
                    print(f'Goal time equal to pull time {row.time}. '
                          'Ignoring it and searching again.')
                    continue
                
                if ((pull_team == 'visitor' and visitor_g_pull)
                    or (pull_team == 'home' and home_g_pull)):
                    # It's a true empty net goal
                    print(Fore.RED + f'Found empty net goal:\n{o}' + Style.RESET_ALL)
                    out.append(o.copy())
                else:
                    print('Goalie was back in the net')
                    pass
                o = {}
                pull_switch = False

        prev_row = row.copy()

    return out


pulls = goalie_pull_game_search(df, team_info, verbose=True)


pulls


center_el = soup.find_all('table', {'id': 'GameInfo'})[0]


center_el


game_number = center_el.find_all('tr')[6].text.split()[-1].lstrip('0')


game_number


date = center_el.find_all('tr')[3].text.strip()


date








def get_game_meta(soup) -> dict:
    '''
    Get game metadata like the date and game number.
    '''
    try:
        center_el = soup.find_all('table', {'id': 'GameInfo'})[0]
        game_number = center_el.find_all('tr')[6].text.split()[-1].lstrip('0')
        date = center_el.find_all('tr')[3].text.strip()
        return {'game_number': game_number, 'date': date}
    except:
        print('Failed to get meta data :(')
        return {}
    
get_game_meta(soup)














def parse_game(
    soup: BeautifulSoup,
    cols: list = None,
    season='',
    verbose=False
) -> list:
    '''
    Search for empty net goals for, against and game ends,
    for a given game. Input the HTML as a BeautifulSoup object.
    '''
    out = []
                  
    # Default columns
    if cols is None:
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

    # Get the game stats df    
    game_df, team_info = get_game_df(soup)
    
    # Get the game metadata
    meta_data = get_game_meta(soup)
    if season:
        meta_data['season'] = season
                  
    # Check for goalie pulls
    visitor_goalie_pull_idx = ~(game_df.visitor_on_ice.str.contains('G'))
    home_goalie_pull_idx = ~(game_df.home_on_ice.str.contains('G'))
    if not (visitor_goalie_pull_idx.sum() + home_goalie_pull_idx.sum()):
        return []    
    
    # For each goalie pull, determine the outcome
    game_5on6_data = goalie_pull_game_search(
        game_df, team_info, verbose=verbose
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
     - time of goal end
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
    df['game_end_timedelta'] = goalie_pull_timedelta(df.game_end_time, df.pull_time)
    
    return df


def parse_game_range(
    seasons: list,
    test: bool = False,
) -> pd.DataFrame:
    '''
    Parse every game for a given season.
    
    Folder structure is:
    ../data/raw/html/{season}/{game_number}.html
    
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
        'game_end_time',
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
    df = make_final_df(data, cols)
    return df


test_parse_game_range('20072008', 980)


seasons = ['20032004']

data, df_goalie_pull = parse_game_range(seasons, test=True)


df_goalie_pull


df_goalie_pull = parse_game_range(seasons)


df_goalie_pull


df_goalie_pull.to_csv('../data/csv/20032004_goalie_pulls_2019-02-13.csv', index=False)


df_goalie_pull.to_pickle('../data/pkl/20032004_goalie_pulls_2019-02-13.pkl')




























