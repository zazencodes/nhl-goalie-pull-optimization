# NHL Goalie-Pull-Time Optimization

We seek out the optimal goalie pull times by looking at historical data and modeling the odds of scoring as a function of the time when goalie was pulled in the 3rd period. In doing so we look for insight into the question "when is the right time to pull your goalie?"


## Blog post: <TODO insert link> 

## Source code

The notebooks have been saved as `.py` and `.html` files in addition to the normal `.ipynb` files. The notebooks can be found in `notbooks/src`. These are labelled as follows:
 1. Save raw HTML from https://www.nhl.com
 2. Parse the goalie pull stats
 3. Exploratory analysis
 4. Maximize the odds of scoring as function of goalie-pull time:
```
P ( Goal For     | Goalie Pulled ; t )
P ( Goal Against | Goalie Pulled ; t )
P ( No Goal      | Goalie Pulled ; t )
```

We determine the risk-reward of pulling a goalie as a function of the time remaining in the game. For instance, if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for? What is the probability it will result in a goal against?



## Project Tree

```
├── LICENSE
├── README.md
│
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   └── processed      <- The final, canonical data sets for modeling.
│
├── models             <- Trained and serialized models
│
├── notebooks
│   ├── src            <- Jupyter Notebooks. Naming convention is
│   │                     (where # and initials are optional):
│   │                     [#]_[2-4 word description]_[initials].ipynb
│   │                     e.g. 1_exploratory_analysis_ag.ipynb
│   │
│   ├── py             <- Script version of notebooks.
│   ├── html           <- HTML version of notebooks.
│
├── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.

```



