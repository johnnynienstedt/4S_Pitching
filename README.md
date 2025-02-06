# 4S_Pitching
Shape | Spot | Slot | Sequence

This project is an attempt to quantify and predict pitching performance by means of four independent models which evaluate four (mostly) independent qualities of a pitcher.

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

This is not a measure of command, because I am not attempting to guess where the pitcher might have been aiming when he released each pitch. I am simply grading the location of each pitch based on its expected run value in that situation (count, pitch type, handedness). Happily, I had already created these run value heatmaps for my SEAGER project, so it was a simple matter to adapt them for this purpose. 

Given that pitch location is a notoriously high-variance statistic, I used a highly informative Bayesian model (normal prior) and updated the posterior with each pitch thrown. For pitchers with a full season's worth of pitches, the variance is only 1-2 runs, and Spot+ has a 0.4 R-squared with its future self. For pitchers with only a few hundred pitches, that predictiveness drops to about 33%. The model does not grade each pitch type separately, but rather gives a single grade to each pitcher for each season. As this model requires no training, grades can be given for any season with appropriate data.

I recently added release point variance as a simple analog for command. This is far less descriptive than location RV, but much more predictive.

# Slot+
The deceptive effects of the pitcher's arm slot and release point.

Inspired by Max Bay's Dynamic Dead Zone, this model quantifies the run value of pitches based solely on their movement relative to other pitches released out of the same slot - not raw movement. As expected, the highest graded pitches include fastballs with more ride than expected, offspeed pitches with more drop, and breaking balls with more break. Interestingly, sweepers are rewarded more for the combination of lift and horizontal break than they are for break alone.

# Sequence+
The pitcher's ability to mix and match different pitch types and locations in proper sequence.

The first part of this model is a simple pitch type sequence matrix which assigns the run value of throwing Pitch Y directly after Pitch X. I did not actually expect this to provide predictive value, but it did, so I left it in.

The second and more robust part of this model assesses the frequency that pitchers execute three optimal sequences. They are:
- Match: same release angle, same location
- Split: same release angle, different location
- Freeze: different release angle, same location

# Results
The following are same season and next season R-squared values for other public models and for 4S+ (min. 1000 pitches, roughly 1/3 full season).

SIERA:
Model | Same Season | Next Season
--- | --- | --- 
Stuff+ | 0.46 | 0.17
botStf | 0.43 | 0.25
4S+    | 0.44 | 0.24

ERA:
Model | Same Season | Next Season
--- | --- | --- 
Stuff+ | 0.11 | 0.07
botStf | 0.16 | 0.10
4S+    | 0.17 | 0.07

The current weights (descriptive/predictive) for 4S+ are:
- Shape+:   37% / 71% 
- Spot+:    52% /  0%
- Slot+:     4% / 10%
- Sequence+: 6% / 19%

# Insights
4S+ has Josh Hader as the best per-inning pitcher in baseball, and Johnny Cueto as the worst. It expects Alexis Diaz to improve the most from 2024, and Robert Garcia to regress the most (in terms of SIERA).

# Run the code
All of the code here is self contained, although you will likely need to install some dependencies. 

If you would like to reproduce my resutls, follow this path:
1. Run get_data.py to scrape and clean data from the 2021-2024 seasons (save classified_pitch_data to csv)
2. Run each of the shape, spot, slot, and sequence scripts (save each of their grades to csv)
3. Run 4S.py to ensure the model is returning the same correlations (save final grades to csv)
4. Run viz.py and analyze as many pitchers as you'd like!

The Spot and Slot scripts contain some visualizations of their own which display, respectively, posterior location RV distributions and slot deviation heatmaps.
