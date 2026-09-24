Disucssion Questions: 

1) *If you only had access to the prior probabilities, would you be more likely to guess that Kennedy or Johnson authored an
unlabeled paper? Why?*

I would guess that the author with more documents would be most likely, so in this case, Johnson. Johnson has a prior of .647 whereas Kennedy has .353 (both rounded up). Johnson is best represented in the data set, so he would be the more likely choice. 

2) *What are your two prior probability estimates? What is the shape of the matrix storing your likelihoods? What happens when
you vary the smoothing hyperparameter alpha?*

My two prior probability estimates are 0.353 for Kennedy (class 0) and 0.647 for Johnson (class 1), reflecting the split of speeches in the training set.
The likelihoods matrix has shape (2, 24390). Two rows per author and 24,390 columns for each unique word across the training vocabulary.

As alpha increases the max likelihood drops and the min likelihood rises. A larger alpha compresses the probability range toward uniform, since it adds a bigger pseudo-count to every word regardless of its real frequency. Smaller alpha stays closer to the true word frequencies, preserving sharper distinctions and leaving unseen words with near-zero probability

3) *What are the predicted authors for each of the unlabeled works?*

My predictions: [np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(0), np.int64(0)]

So -- Johnson, Kennedy, Kennedy, Kennedy, Kennedy, Johnson, Kennedy, Johnson, Kennedy, Kennedy

4) *How do your predictions in Problem 1 compare with the scikit-learn implementation of Naive Bayes?*

The scikit-learn predictions were [1, 0, 0, 0, 0, 1, 0, 1, 1, 0], compared to my own model's [1, 0, 0, 0, 0, 1, 0, 1, 0, 0]. The two models agree on 9 of the 10 predictions.
Disagreeing on text 8, my model was still close to the standard library implimentation, suggesting that it's functioning correctly. 

5) *Report the accuracy and F1-score of both the Naive Bayes classifier you created and the one off-the-shelf from
scikit-learn in your discussion.md file.*

My model - accuracy: 0.8 f1: 0.75
Sklearn - accuracy: 0.9 f1: 0.8888888888888888



6) * For each classifier, what do you notice from the confusion matrix? *

My model - confusion matrix:[[5 0][2 3]]

Sklearn - confusion matrix:[[5 0] [1 4]]

Both models share the same biases towards mistaking Johnson's, but the sklearn model is more accurate
This indicates the difficulty lies in the data itself, rather than a bug in either model. 









