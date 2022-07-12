***
permutation_fi.py

Use: (taken from article, not my own) Train model on train set, get
model score on test set. Score will be baseline. Shuffle one feature
at a time on test set, then feed data to model to get new score. If
feature that was just shuffled is important, the model should suffer
and score should drop significantly. Otherwise, model should not be
impacted and the feature is not important.
***
from sklearn.inspection import permutation_importance

# Good to repeat shuffling process given random nature of experiment,
# Ensure results are statistically significant.
r = permutation_importance(rf_estimator, X_test, y_test,
                           n_repeats=10,
                           random_state=0)

# Load to dataframe
perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'],
                    index=[i for i in X_train.columns])

perm['AVG_Importance'] = r.importances_mean
