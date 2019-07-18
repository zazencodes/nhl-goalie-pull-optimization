# NHL Goalie-Pull-Time Optimization

Using historical NHL goalie pulls data, we model the probability of scoring goals for and having goals scored against. We then solve for the optimal goalie pull time as the maximum liklihood of the scoring odds ratio.

## Blog post: <TODO insert link> 

## Source code

The source code consists of jupyter notebooks, which have been saved as `.py` and `.html` files in addition to the normal `.ipynb` file. The notebooks can be found in `notbooks/src` and are labelled as follows:

Notebook Number | Description
--- | ---
**1** | Saving raw HTML from nhl.com
**2** | Parsing goalie pull statistics
**3** | Exploring the data
**4** | Modeling with pymc3

We model the following conditional probabilities as a function of time elapsed in the 3rd period:
```
P ( Goal For     | Goalie Pulled ; t )
P ( Goal Against | Goalie Pulled ; t )
P ( No Goal      | Goalie Pulled ; t )
```

We determine the risk-reward of pulling a goalie as a function of the time remaining in the game. For instance, if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for? What is the chance it will result in a goal against?

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



