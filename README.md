# 4S_Pitching
Shape | Spot | Slot | Sequence

This project is an attempt to quantify and predict pitching performance by means of four independent models which evaluate four (mostly) independent qualities of a pitcher. As of this writing, I am mostly satisifed with my work on Shape+, Spot+, and Slot+, and am in the process of developing Sequence+.

# Shape+
The physical characteristics of a pitch as it approaches the plate. Velocity, movement, and approach angles. 

The effects of pitch shape are felt most heavily when the batter swings. Pitches with better shape will induce swings and misses, foul balls, and poor contact at a higher rate. An XGBoost model uses the above predictors to model the likelihood of the following eight possible results of a swing:
 - swing and miss
 - foul ball
 - ball in play
     - weak
     - under
     - topped
     - flare/burner
     - solid
     - barrel

These probabilites are then converted into an expected run value for the pitch. These values are aggregated on the pitcher-pitch type level and normalized to a mean of 100 and standard deviation of 10. They are again normalized on the pitcher level, such that a 120 Shape+ changeup and a 120 Shape+ full repertoire both represent 97th percentile grades.

This model is trained on all pitch data from 2021 to 2022, meaning pitchers from those seasons cannot be graded.

# Spot+
The pitcher's ability to throw the ball in the correct spots.

This is not a measure of command, because I am not attempting to guess where the pitcher might have been aiming when he released each pitch. I am simply grading the location of each pitch based on its expected run value in that situation (count, handedness). Happily, I had already created these run value heatmaps for my SEAGER project, so it was a simple matter to adapt them for this purpose. 

Given that pitch location is a notoriously high-variance statistic, I used a highly informative Bayesian model (normal prior) and updated the posterior with each pitch thrown. For pitchers with a full season's worth of pitches, the variance is only 1-2 runs, and Spot+ has a 0.4 R-squared with its future self. For pitchers with only a few hundred pitches, that predictiveness drops to about 33%. The model does not grade each pitch type separately, but rather gives a single grade to each pitcher for each season. As this model requires no training, grades can be given for any season with appropriate data.

# Slot+
The deceptive effects of the pitcher's arm slot and release point.

Inspired by Max Bay's Dynamic Dead Zone, this model quantifies the run value of pitches based solely on their movement relative to other pitches released out of the same slot - not raw movement. As expected, the highest graded pitches include fastballs with more ride than expected, offspeed pitches with more drop, and breaking balls with more break. Interestingly, sweepers are rewarded more for the combination of lift and horizontal break than they are for break alone.

# Sequence+
The pitcher's ability to mix and match different pitch types and locations in proper sequence.

In the early stages of development. Thus far I have quantified the value of release point variance, which is significant on the pitch-type level (more tightly grouped is better). I plan to expand this with analogs of Driveline's Mix+/Match+ repertoire statistics.

# Results
The following are same season and next season R-squared values for other public models and for 4S+ (min. 1000 pitches, roughly 1/3 full season).

SIERA:
Model | Same Season | Next Season
--- | --- | --- 
Stuff+ | 0.46 | 0.17
botStf | 0.43 | 0.25
4S+    | 0.44 | 0.23

ERA:
Model | Same Season | Next Season
--- | --- | --- 
Stuff+ | 0.11 | 0.07
botStf | 0.16 | 0.10
4S+    | 0.18 | 0.06

Of course, 4S+ is really just 3S+ at this point.

The current weights (descriptive/predictive) for 3S+ are:
- Shape+ - 38% / 87% 
- Spot+  - 59% / 0%
- Slot+  -  4% / 13%

# Insights
4S+ has Josh Hader as the best per-inning pitcher in baseball, and Johnny Cueto as the worst. It expects Hurston Waldrep to improve the most from 2024, and Nick Hernandez to regress the most.
