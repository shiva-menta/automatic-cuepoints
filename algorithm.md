# Recurrence Engine Algorithm (High-Level)

## Tenets
- CONTEXT MATTERS. Recurrence matrix does a great job at capturing similarities in the track - gives you more of a certainty that you're marking similar portions the same way.

## Steps
- P0
  - Find all off-main diagonals (recurrence matrix image is helpful for visualization, but doesn't really do much in terms of helping with code). Better to find nearest neighbors directly (use Python) and then try to factor in from there.
    - Inter-section similarlity.
  - Find some way to detect square-like shapes (not fully-filled squares, but even just outlines that are easy to see visually).
    - Intra-section similarity.
    - These indicate sub-sections, not necessarily full section (need context to convert these into full sections).

- P1
  - Need to add tolerance for outro drops, or segment transitions as a whole (can make some likelihood prediction about length of the section using other sections that have already been calculated - or just based on energy calculations).
  - Patterns in recurrence matrix are good indicators of similar sections (section self-similarity characteristics).
  - Factor in RMS into formula to help find explicit ends of sections, section labeling, etc.

## Other Heuristics
- Enforce no 1-2-3 measure transitions unless extreme self similarity or diagonal (otherwise just merge to the best option).
- Don't need to label chorus differently than refrain.
- Sections aren't usually directly identical, but have enough similarity (e.g. think about choruses that have diagonals in last half, but no self similarity in the first half).
- Add a heuristic to enforce that all non chorus / build-up / drop sections that are adjacent are merged together (helps for intro, verses that change instrumentation).

## Approaches
- Pure Code + Matrix Manip.
- OpenCV (Visual style approach)
- Try conversion to image then passing to multimodal model and see how well that works as a baseline.
- Try building LSTM model with small dataset and see how it performs.

## Modes
- Vocal Splitting Mode
  - Try to develop some quick heuristic to determine if vocal splitting would even be helpful (some way of likelihood of vocals being in track using VAD?). Doesn't need to be accurate, just needs to be fast enough.
  - Vocal splitting seems necessary to perform on vocal tracks (instrumentation isn't enough).
  - Can try using audio transcription + pulling lyrics from Genius.
- No Vocal Splitting Mode (based on computation)

## Other Thoughts
- Overlapping manually is the same as self-similarity - no need for this.
- Maybe can think of doing some best guess splitting + voting algorithm for where patterns might exist.
- If we are unsure about some section, you should just merge the sections.
- I think to make this work effectively, we realistically need some sort of compound algorithm that uses multiple techniques:
  - Recurrence analysis.
  - Maybe novelty curve detection (if it shares any LARGE) spikes - could use this as hints for self-similarity.
  - Measure by measure similarity scores.
- Few strategies to look into today.
  - Improve longest diagonal with a tolerance measure (look at Leetcode algo).
  - How to detect square-like shapes in an image? This will be super high yield IMO.
  - How to handle cases that have square-like diagonal patterns (intra-section similarity) - not considering these? How to choose the right one?
  - If you need to do exhaustive search, how can you narrow down the search space.

## Next
- Start-working on intra-section similarity.
  - Let's try novelty curve approach as an estimator.
    - This seems to be pretty good at finding ALL possible boundaries within main diagonal
  - Corner detection in graph + limit search space with black.

## Process To-Do
- Recurrence Matrix params experimentation.
  - Right now k scales with the square root of input size - this doesn't make sense - it should be roughly linearly since longer track generally means

## Progress
- No-Tolerance Off-Diagonal Approach (0.072 F1 Score)
- Diagonal per Measure (0.1312 F1 Score)
  - Not sure this approach is the smartest - doing a lot of correction.
  - Also getting tripped up on these patterns in sections
- Tolerance on Diagonal Measure
  - 0.05 ({'true_positive': 80, 'false_positive': 166, 'false_negative': 809}, F1 Score: 0.14096916299559473)
  - 0.10 ({'true_positive': 89, 'false_positive': 199, 'false_negative': 800}, F1 Score: 0.15123194562446898)
  - 0.15 ({'true_positive': 93, 'false_positive': 225, 'false_negative': 796}, F1 Score: 0.15410107705053852)
  - 0.20 ({'true_positive': 103, 'false_positive': 261, 'false_negative': 786}, F1 Score: 0.16440542697525937)