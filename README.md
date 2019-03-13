# Bayes Optimized NHL Goalie Pull Times

 1. Save raw HTML from https://www.nhl.com
 2. Parse the goalie pull stats
 3. Exploratory analysis
 4. Solve for the posterior probabilties at time t:
```
P( goal for     | goalie pulled )
P( goal against | goalie pulled )
P( no goal      | goalie pulled )
```

The idea is to figure out the risk reward of pulling a goalie as a function of the time remaining in the game. For instance, if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for? What is the probability it will result in a goal against?

For more detailed information, check out the source files in `notebooks/src`.

### Useful Links
 - https://github.com/dword4/nhlapi
 - http://www.nhl.com/scores/htmlreports/20032004/PL020970.HTM


### Project Tree

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   └── processed      <- The final, canonical data sets for modeling.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks
│   ├── src            <- Jupyter Notebooks. Naming convention is
│   │                     (where # and initials are optional):
│   │                     [#]_[2-4 word description]_[initials].ipynb
│   │                     e.g. 1_exploratory_analysis_ag.ipynb
│   │
│   ├── py             <- Script version of notebooks.
│   ├── html           <- HTML version of notebooks.
│   └── archive        <- Datestamped archive notebooks. Naming convention is:
│                         [#]_[2-4 word description]_[DS-initials]_[ISO 8601 date].ipynb
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Deliverable reports as HTML, PDF, LaTeX, etc.
│
├── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
└── src                <- Source code for use in this project.
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py

```



