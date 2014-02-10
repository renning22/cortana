## Naive Bayes Classifier

Directly applying naive bayes classifer got 0.837 accuracy. - *2014.01.28*

### Use lexcicon and grouping words by category

Like use \_RARE\_ to replace all words occure less than 5 times. Use \_DIGITS\_ to replace all words that are digits, like 2014.

It improved the accuracy to 0.848. - *2014.01.29*

### Use sqrt of each log probability

Such that then difference of log probabilities that are too small(large in abs value) get weakened. It improve the accuracy to 0.872. -  *2014.01.30*

### Treat multiple occurance of words as only once in decoding and training

Improve accuracy to 0.879. - *2014.01.31*

### Use average backoff from all domains

Improve accuracy to 0.880. - *2014.01.31*



--------------------------------------------------------

### ideas failed

1. Adjusting the **\_RARE\_** threshold to other than 5 doesn't help.
2. Adding Chinese surname lexicon didn't improve the model.
3. Trying to use clustering method. Currently use KMeans(k = v/10) got 0.84 of accuracy. `v` is size of the vocalular.
