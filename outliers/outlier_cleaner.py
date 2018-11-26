#!/usr/bin/python

from sklearn.feature_selection import SelectPercentile, f_classif
from operator import itemgetter

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    zipped = zip(ages, net_worths, abs(net_worths - predictions))

    zipped.sort(key=itemgetter(2))

    new_len = int(len(zipped) * 0.9)

    cleaned_data = zipped[: new_len]

    return cleaned_data

