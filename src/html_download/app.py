# -*- coding: utf-8 -*-
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






"""
Set global variables
"""

# SEASONS = ['20072008', '20082009', '20092010',
#            '20102011', '20112012', '20122013',
#            '20132014', '20142015', '20152016',
#           '20162017', '20172018', '20182019']
SEASONS = ['20182019']
REQUEST_DELAY = 3
MAX_NUM_GAMES = 3000
ROOT_DATA_PATH = '../../data/raw/html'



def main():
    """
    Download game sheet HTML files from nhl.com
    """
    url_tempalte = 'http://www.nhl.com/scores/htmlreports/{:}/PL02{:04d}.HTM'
    games = list(range(1, MAX_NUM_GAMES))
    download_game_range(url_tempalte, games)

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
    games,
    no_break=False
    ) -> None:

    if not os.path.exists(ROOT_DATA_PATH):
        os.makedirs(ROOT_DATA_PATH)
        print(f'Making dirs {ROOT_DATA_PATH}')

    request_delay = REQUEST_DELAY
    print(f'Starting data pull at {datetime.datetime.now()}')
    sess = init_sess()
    for season in SEASONS:
        data_path = os.path.join(ROOT_DATA_PATH, season)
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
        if season != SEASONS[-1]:
            print('Waiting 10 minutes...')
            time.sleep(10*60)
    print(f'Ending data pull at {datetime.datetime.now()}')


if __name__ == '__main__':
    main()
