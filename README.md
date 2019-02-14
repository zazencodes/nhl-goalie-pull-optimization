# Bayes Optimized NHL Goalie Pull Times

 1. Save raw HTML from nhl.com
 2. Parse the goalie pull stats we need
 3. Exploratory analysis
 4. Solve for the posterior probabilties at time t:
```
P(goal for | goalie pulled)
P(goal against | goalie pulled)
```

The idea is to figure out the risk reward of pulling a goalie as a function of the time remaining in the game. For instance, if there's 3 minutes left, what is the chance that pulling the goalie will result in a goal for? What is the probability it will result in a goal against?

### Useful Links
 - https://github.com/dword4/nhlapi
 - http://www.nhl.com/scores/htmlreports/20032004/PL020970.HTM
 - [First game of 2003 season](http://www.nhl.com/scores/htmlreports/20032004/PL020001.HTM)!


