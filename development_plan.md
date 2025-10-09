# Automatic Cuepoints

## Current State
- Have some initial UI widget.
- Beatpoint grids work but not super well.

## To-Do
- Go through code + refresh.
  - Cuepoint engine is ABC.
  - Heuristics take a list of first beat timestamps, cuepoint timestamps, and returns cuepoint timestamps. Seem like hacks for the most part.
- Think about phrase detection / pattern detection, how that works in current framework, if that needs to be overwritten.
  - I think the model does a good (okay) job of separating sections of a track based on sonic values, but it doesn't have any understanding of song structure, learned features.

## Process
- Intro – ALWAYS add the first one.
- Chorus vocals, introduced bassline.
- Ignore instrumental changes when same verse is continuining.
- Increase / decrease in tension
- Big bassline / loudness change.
- Lower in loudness.
- Intro

## Doing Today
- Need to do a refresher of the code.
- Pattern matching.
  - This will be helpful to make sure markings are accurate between different things. Basically this will be dot product of two matrices.
- Accurate vocal marking so that we can do XXX.

## Reseach
- Vocal Detection Packages
  - Spleeter by Deezer (stem separation)
    - Takes 30-90 second per track.
    - Let's start with this because this is the fastest.
    - Setup a notebook.
  - Demucs by Meta is also an option.
  - librosa (shitty version)
    - Need to develop custom detection features / experiment.

## Assumptions
- Beatgrid is correct (this is something we can work on later but should be fine in interim).

## Need
- Vocal Detection (would be nice to have some way of telling when vocal is coming in - helpful for loops)
- Actually accurate phrasing - my approach takes into account frequency bands but there's no level of normalization, also understanding of phrasing seems poor
  - What does this mean?
- Pattern matching in the track (can do some sparse vector similarity based on sampling the different frequency bands")
- Speed up beatmatching 
- Labeling by verse / chorus / breakdown / etc (i like the idea of doing a state machine here to help with this) – sequence classification + learned models
- Vocal detection is definitely needed - can sample volume values too here
- Serato Integration
- DJay Pro Integration (open as a to-do for future integrations)

## Most Common Mistakes Right Now (use Garage as example)
- Intro sequences are overmarked (heuristic about vocal start for intro sections).
- Disincentivize on intros / outros.