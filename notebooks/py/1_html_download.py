#!/usr/bin/env python
# coding: utf-8

from IPython.display import HTML
HTML('<style>div.text_cell_render{font-size:130%;}</style>')
get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'pandas')


# # Goalie Pull Bayes Optimize
# 
#  - Save raw HTML from nhl.com

# ## Get Training Data

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


# ### Download HTML

def random_wait(mu) -> float:
    ''' Positive stochastic var with average of mu '''
    return np.random.beta(3, 3) * mu * 2

def init_sess(sess=None):
    if sess is not None:
        sess.close()
    _sess = requests.Session()
    return _sess

def get_page(sess, url, tries=0) -> str:
    try:
        if tries > 3:
            print(f'Scrape failed at URL = {url}')
            return None

        print(f'Requesting HTML for URL = {url}')
        page = sess.get(url)
        print(f'Got {page.status_code} status code')
        
        if page.status_code == 404:
            print('Bad status code, returning no page')
            return None

        if page.status_code not in (200, 404):
            print('Bad status code, waiting 10 seconds and trying again')
            time.sleep(10)
            sess = init_sess(sess)
            return get_page(sess, url, tries+1)

        return page.text

    except Exception as e:
        print(f'Exception: {str(e)}')
        print('Sleeping, then trying again...')
        time.sleep(10)
        sess = init_sess(sess)
        return get_page(sess, url, tries+1)
        
        
def download_game_range(
    url_template,
    seasons,
    games,
    no_break=False
    ) -> None:
    
    root_data_path = '../../data/raw/html'
    if not os.path.exists(root_data_path):
        os.makedirs(root_data_path)
        print(f'Making dirs {root_data_path}')
        
    request_delay = 3

    print(f'Starting data pull at {datetime.datetime.now()}')

    sess = init_sess()
    for season in seasons:
        data_path = os.path.join(root_data_path, season)
        if not os.path.exists(data_path):
            print(f'Making dirs {data_path}')
            os.makedirs(data_path)
        
        for game_num in games: 
            time.sleep(random_wait(request_delay))

            page_text = get_page(
                sess,
                url_template.format(season, game_num)
            )
            if page_text is None:
                if no_break:
                    print('Bad response, trying next page')
                    continue
                print(f'Season = {season}')
                print(f'Max game = {game_num - 1}')
                break

            f_name = os.path.join(data_path, f'{game_num}.html')
            print(f'Writing HTML to file {f_name}')
            with open(f_name, 'w') as f:
                f.write(page_text)
                
        print(f'Done season {season}')
        if season != seasons[-1]:
            print('Waiting 10 minutes...')
            time.sleep(10*60)

    print(f'Ending data pull at {datetime.datetime.now()}')


# url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
# seasons = ['20032004']
# games = list(range(1, 5000))

# download_game_range(url_tempalte, seasons, games)


#  - For 2003/2004 we got up to http://www.nhl.com/scores/htmlreports/20032004/PL021231.HTM

# url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
# seasons = ['20052006']
# games = list(range(1, 5000))

# download_game_range(url_tempalte, seasons, games)


# url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
# seasons = ['20062007']
# games = list(range(1, 5000))

# download_game_range(url_tempalte, seasons, games)


url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
seasons = ['20032004']
games = list(range(1, 3000))

download_game_range(url_tempalte, seasons, games, no_break=True)








# ### Legacy format
# Pull the old format games. 

url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
seasons = ['20032004']
games = list(range(1, 3000))

download_game_range(url_tempalte, seasons, games, no_break=True)


url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
seasons = ['20042005']
games = list(range(1, 3000))

download_game_range(url_tempalte, seasons, games, no_break=True)


url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
seasons = ['20052006']
games = list(range(1, 3000))

download_game_range(url_tempalte, seasons, games, no_break=True)


url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
seasons = ['20062007']
games = list(range(1, 3000))

download_game_range(url_tempalte, seasons, games, no_break=True)


# We accidentaly saved all the 404 pages... not sure why as these should have been skipped. Let's clean them up

def clean_folder_404s(season):
    f_pattern = '../../data/raw/html/{}/*.html'.format(season)
    files = sorted(glob.glob(f_pattern))
    print(f'Found {len(files)} files')
    for file in files:
        with open(file, 'r') as f:
            text = f.read()
        if not text.strip():
            print(f'Deleting {file}')
            os.remove(file)

clean_folder_404s('20032004')
clean_folder_404s('20042005')
clean_folder_404s('20052006')
clean_folder_404s('20062007')








# ### Modern format

url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
seasons = ['20072008', '20082009', '20092010',
           '20102011', '20112012', '20122013',
           '20132014', '20142015', '20152016',
          '20162017', '20172018']
games = list(range(1, 3000))

download_game_range(url_tempalte, seasons, games)


# Pulled this data using `src/html_download/app.py`



