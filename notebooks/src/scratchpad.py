out = []
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
game_df = get_game_df(soup)    