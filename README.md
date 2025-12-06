# 4S_Pitching
Shape | Spot | Slot | Sequence

This project is an attempt to quantify and predict pitching performance by means of four independent models which evaluate four (mostly) independent qualities of a pitcher. These models were trained on data from 2021-2025, using K-fold training where necessary to prevent contamination.

# Shape+
The physical characteristics of a pitch as it approaches the plate. Velocity (represented by time to plate), movement (represented by acceleration, rotated to the reference frame of the pitch), and approach angles (represented by release point, to remove location bias). 

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

# Spot+
The pitcher's ability to throw the ball in the correct spots.

This is not a measure of command, because I am not attempting to guess where the pitcher might have been aiming when he released each pitch. I am simply grading the location of each pitch based on its expected run value in that situation (count, pitch type, handedness). Happily, I had already created these run value heatmaps for my SEAGER project, so it was a simple matter to adapt them for this purpose. 

Given that pitch location is a notoriously high-variance statistic, I used a strong Bayesian prior (normal) and updated the posterior with each pitch thrown. For pitchers with a full season's worth of pitches, the variance is only 1-2 runs, and Spot+ has a 0.7 R-squared with its future self. For pitchers with only a few hundred pitches, that predictiveness drops to about 40% (and it's only that high becuase it sets everyone with that few pitches very close to zero). 

The second part of this model is release angle variance, which comes in three flavors: release variance (all pitches) repertoire variance (variance between averages of each pitch type), and pitch-by-pitch variance (variance within pitch types). The first two are positive when maximized, and the last when minimized. That is, have a diverse arsenal, but repeat your release point consistently within pitch types.

# Slot+
The deceptive effects of the pitcher's arm slot and release point.

Inspired by Max Bay's Dynamic Dead Zone, this model quantifies the run value of pitches based solely on their movement relative to similar pitches released out of the same slot - not raw movement. As expected, the highest graded pitches include fastballs with more ride than expected, offspeed pitches with more drop, and breaking balls with more break.

The second part of this model is a simple calculation of slot rarity: the percentage of pitches in MLB released within 5 degrees of that slot.

# Sequence+
The pitcher's ability to mix and match different pitch types and locations in proper sequence.

The first part of this model is a simple pitch type sequence matrix which assigns the run value of throwing Pitch Y directly after Pitch X. I did not actually expect this to provide predictive value, but it did, so I left it in.

The second and more robust part of this model assesses the frequency that pitchers execute three optimal sequences. They are:
- Match: same release angle, same location
- Split: same release angle, different location
- Freeze: different release angle, same location

# Results
The table below shows same-season and next-season R-squared values for SIERA (min. 1000 pitches, roughly 1/3 full season):

Model | Same Season | Next Season
--- | --- | --- 
Stuff+ | 0.46 | 0.17
botStf | 0.43 | 0.25
4S     | 0.46 | 0.30

The current weights (descriptive/predictive) for 4S+ are:
- Shape+:    54% / 81% 
- Spot+:     36% / 0%
- Slot+:     4% / 5%
- Sequence+: 6% / 14%

# Insights (post-2024)
4S has Josh Hader as the best per-inning pitcher in baseball, and Johnny Cueto as the worst. It expects Alexis Diaz to improve the most from 2024, and Robert Garcia to regress the most (in terms of SIERA).
