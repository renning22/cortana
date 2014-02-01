## Naive Bayes Classifier

Directly applying naive bayes classifer got 0.837 accuracy. - *2014.01.28*

### Use lexcicon and grouping words by category

Like use _RARE_ to replace all words occure less than 5 times. Use digits to replace all words that are digits, like 2014.

It improve the accuracy to 0.848. - *2014.01.29*

### Use sqrt of each log probability

Such that then difference of log probabilities that are too small(large in abs value) get weakened. It improve the accuracy to 0.872. -  *2014.01.30*

### See multiple occurance of words as only once

Improve accuracy to 0.875. - *2014.01.31*
