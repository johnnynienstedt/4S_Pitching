# 4S_Pitching
Shape | Spot | Slot | Sequence

This project is an attempt to quantify and predict pitching performance by means of four independent models which evaluate four (mostly) independent qualities of a pitcher. As of this writing, I am satisifed with my work on Shape+ and Spot+, and am in the process of developing Slot+ and Sequence+.

# Shape+
The physical characteristics of a pitch. Velocity, movement, and approach angles. 

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

These probabilites are then converted into an expected run value for the pitch. These values are aggregated on the pitcher-pitch type level and normalized to a mean of 100 and standard deviation of 10. They are again normalized on the pitcher level, such that a 130 Shape+ changeup and a 130 Shape+ full repertoire represent the same deviation from the mean.

One of the major limitations of this model is that it is trained on all pitch data from 2021 to 2023, meaning pitchers from those seasons cannot be graded.

# Spot+
The pitcher's ability to throw the ball in the correct spots.

This is not a measure of command, because I am not attempting to guess where the pitcher might have been aiming when he released each pitch. I am simply grading the location of each pitch based on its expected run value in that situation (count, handedness). Happily, I had already created these run value heatmaps for my SEAGER project, so it was a simple matter to adapt them for this purpose. 

Given that pitch location is a notoriously high-variance statistic, I used a highly informative Bayesian model (normal prior) and updated the posterior with each pitch thrown. For pitchers with a full season's worth of pitches, the variance is only 1-2 runs, and Spot+ has a 0.4 R-squared with its future self. For pitchers with only a few hundred pitches, that predictiveness drops to about 33%. The model does not grade each pitch type separately, but rather give a single grade to each pitcher for each season. As this model requires no training, grades can be given for any season with appropriate data.

# Slot+
The deceptive effects of the pitcher's arm slot (and potentially other factors)

This is currently in the early stages of development. Inspired by Max Bay's Dynamic Dead Zone, I am attempting to expand that work by quantifying the run value of pitches based solely on their movement relative to other pitches released out of the same slot - not raw movement. So far I have not seen any meaningful correlation to performance.

In the future, I will likely also attempt to quantify the value of a unique/rare arm slot, as well as the ability to hide the baseball before release.

# Sequence+
The pitcher's ability to mix and match different pitch types and locations in proper sequence

TBD. Potentially some sort of Pitch 1 -> Pitch 2 matrix of run values.

# Results
The following are same-season R-squared values for other public models and for 4S (min. 1000 pitches, roughly 1/3 full season).

K%:
- Stuff+ - 0.39
- botStf - 0.49
- Shape+ - 0.30

BB%:
- Location+ - 0.57
- botCmd    - 0.48
- Spot+     - 0.53

SIERA:
- Pitching+ - 0.46
- botOvr    - 0.43
- 4S+       - 0.38

Of course, 4S+ is really just 2S+ at this point.
