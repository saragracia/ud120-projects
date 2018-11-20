#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of poi", sum(enron_data[item]["poi"] for item in enron_data)


#What is the total value of the stock belonging to James Prentice?
print "Total stok value James Prentice", enron_data["PRENTICE JAMES"]["total_stock_value"]


#How many email messages do we have from Wesley Colwell to persons of interest?
print "Messages to poi", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

#What is the value of stock options exercised by Jeffrey K Skilling?
print "exercided stock value", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

#Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of total_payments feature)?
subset = {"SKILLING JEFFREY K":enron_data["SKILLING JEFFREY K"]["total_payments"], 
		"LAY KENNETH L":enron_data["LAY KENNETH L"]["total_payments"],
		"FASTOW ANDREW S":enron_data["FASTOW ANDREW S"]["total_payments"]}

print "Who took more money?",max(zip(subset.values(), subset.keys()))


#How many folks in this dataset have a quantified salary? What about a known email address?
print "# quantified salary?", sum(each["salary"] !='NaN' for each in enron_data.itervalues())
print "# quantified email address?", sum(each["email_address"] !='NaN' for each in enron_data.itervalues())

#How many people in the E+F dataset (as it currently exists) have NaN for their total payments? What percentage of people in the dataset as a whole is this?
totalpaymentsNAN = sum(each["total_payments"] =='NaN' for each in enron_data.itervalues())
print totalpaymentsNAN / float(len(enron_data))

#How many POIs in the E+F dataset have NaN for their total payments? What percentage of POIs as a whole is this?
print sum(each["total_payments"] =='NaN' and each["poi"] for each in enron_data.itervalues())

#What is the new number of people of the dataset? What is the new number of folks with NaN for total payments?
print "Number of people", len(enron_data), "total payments NAN", totalpaymentsNAN
