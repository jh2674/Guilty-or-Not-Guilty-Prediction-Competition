import numpy as np
import pandas as pd
import re
import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


## MAP TEXT DATA TO NUMBER
def text2idx(text_list, idx_list):
    map = dict((zip(text_list, idx_list)))
    return map


## SEARCH KEY WITH VALUE IN A DICTIONARY
def idx2text(dict, value):
    return [k for k, v in dict.items() if v == value]


## SEARCH VALUE WITH KEY IN A DICTIONARY
def text2frequency(dict, key):
    if key in dict.keys():
        return dict.get(key)
    else:
        return 0


## CREATE TOP NUM OF DICTIONARY
def topDict(list, top_num):
    wc = Counter(list)
    pc = wc.most_common(top_num)
    return dict(pc)


## CREATE CUSTOM COLUMN NAME
def createName(text, length):
    name_list = []
    for i in range(length):
        name_list.append(text + '_%s' % i)
    return name_list

## FIND OUT Day
def findDay(text):
    date = re.findall(r'\d+\S\d+\S\d+', text)
    month, day, year = (int(x) for x in date[0].split('/'))
    return day

## FIND OUT MONTH
def findMonth(text):
    date = re.findall(r'\d+\S\d+\S\d+', text)
    month, day, year = (int(x) for x in date[0].split('/'))
    return month

## FIND OUT YEAR
def findYear(text):
    date = re.findall(r'\d+\S\d+\S\d+', text)
    month, day, year = (int(x) for x in date[0].split('/'))
    return year

## FIND OUT THE DAY OF A WEEK (INCLUDING DATE VALIDATION AND MODIFICATION)
def findDayofWeek(text):
    date = re.findall(r'\d+\S\d+\S\d+', text)
    month, day, year = (int(x) for x in date[0].split('/'))
    if month == 2:
        if day < 1:
            day = 1
            print("NG")
        else:
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 3200 != 0):
                if day > 29:
                    day = 29
                    print("R")
            else:
                if day > 28:
                    day = 28
                    print("NR")
    elif month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        if day > 31:
            day = 31
        if day < 1:
            day = 1
    else:
        if day > 30:
            day = 30
        if day < 1:
            day = 1
    ans = datetime.date(year, month, day)
    day_week = ans.weekday()
    return day_week

## SEARCH VALUE WITH KEY IN A SPECIAL DICTIONARY
def day2frequency(dict, key):
    if key in dict.keys():
        if key == 0 or key == 1 or key == 2 or key == 3 or key == 4:
            return dict.get(1)
        else:
            return dict.get(key)
    else:
        return 0

def getHour(text):
    m = text[-1]
    if m != "P" and m != "A":
        temp = text
        m = "A"
    else:
        temp = text[:-1]
    if not temp[-1].isdigit():
        temp = "00:00"
    time = re.findall('..:+..', temp)
    hour, minute = (int(x) for x in time[0].split(':'))
    minute = round(minute, -1)
    if minute == 60:
        minute = 0
        hour = hour +1
    if m == "P":
        hour = hour % 12 + 12
    round_time = hour
    return round_time

def getMinute(text):
    m = text[-1]
    if m != "P" and m != "A":
        temp = text
        m = "A"
    else:
        temp = text[:-1]
    if not temp[-1].isdigit():
        temp = "00:00"
    time = re.findall('..:+..', temp)
    hour, minute = (int(x) for x in time[0].split(':'))
    minute = round(minute, -1)
    if minute == 60:
        minute = 0
        hour = hour +1
    if m == "P":
        hour = hour % 12 + 12
    round_time = minute
    return round_time

## LOAD DATA & PROCESS LABELS
trn_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

## STATISTIC
# print("--------------------------------------")
# print("State", trn_raw['State'].value_counts())
# print("--------------------------------------")
# print("License Type", trn_raw['License Type'].value_counts())
# print("--------------------------------------")
# print("County", trn_raw['County'].value_counts())
# print("--------------------------------------")
# print("Precinct", trn_raw['Precinct'].value_counts())
# print("--------------------------------------")
# trn_raw['Precinct'].value_counts().to_csv('precinct.csv', encoding='gbk')

#print(len(trn_raw['State'].value_counts()))


## DATA PREPROCESSING
# raw data processing
trn_raw['Violation'] = trn_raw['Violation'].fillna(method='ffill')
test_raw['Violation'] = test_raw['Violation'].fillna(method='ffill')
trn_raw['Issuing Agency'] = trn_raw['Issuing Agency'].fillna(method='ffill')
test_raw['Issuing Agency'] = test_raw['Issuing Agency'].fillna(method='ffill')
trn_raw['County'] = trn_raw['County'].fillna(method='ffill')
test_raw['County'] = test_raw['County'].fillna(method='ffill')
trn_raw['Precinct'] = trn_raw['Precinct'].fillna(method='ffill')
test_raw['Precinct'] = test_raw['Precinct'].fillna(method='ffill')
trn_raw['License Type'] = trn_raw['License Type'].fillna(method='ffill')
test_raw['License Type'] = test_raw['License Type'].fillna(method='ffill')
trn_raw['Violation Time'] = trn_raw['Violation Time'].fillna(method='ffill')
test_raw['Violation Time'] = test_raw['Violation Time'].fillna(method='ffill')
trn_raw['Judgment Entry Date'] = trn_raw['Judgment Entry Date'].fillna("1/1/1996")
test_raw['Judgment Entry Date'] = test_raw['Judgment Entry Date'].fillna("1/1/1996")


subname1 = ['Payment Amount', 'Reduction Amount', 'Amount Due', 'Penalty Amount', 'Fine Amount', 'Interest Amount']
subname2 = ['Violation', 'State', 'License Type', 'County', 'Issue Month', 'Issue Year', 'Summons Number']
subname3 = ['Judge Year','Judge Month', 'Plate']
name = np.hstack((subname1, subname2,subname3))
#subname1 = ['Payment Amount', 'Reduction Amount', 'Penalty Amount', 'Fine Amount', 'Interest Amount', 'Amount Due']
#subname2 = ['State', 'License Type', 'Day of The Week', 'County', 'Violation']
#name = np.hstack((subname1, subname2))


# empty training set
trn_set = pd.DataFrame(columns=name)
# empty testing set
test_set = pd.DataFrame(columns=name)

# fill with feature vectors
# Money & Summons Number
trn_set['Payment Amount'] = trn_raw['Payment Amount']
test_set['Payment Amount'] = test_raw['Payment Amount']
trn_set['Reduction Amount'] = trn_raw['Reduction Amount']
test_set['Reduction Amount'] = test_raw['Reduction Amount']
trn_set['Penalty Amount'] = trn_raw['Penalty Amount']
test_set['Penalty Amount'] = test_raw['Penalty Amount']
trn_set['Fine Amount'] = trn_raw['Fine Amount']
test_set['Fine Amount'] = test_raw['Fine Amount']
trn_set['Interest Amount'] = trn_raw['Interest Amount']
test_set['Interest Amount'] = test_raw['Interest Amount']
trn_set['Amount Due'] = trn_raw['Amount Due']
test_set['Amount Due'] = test_raw['Amount Due']
trn_set['Summons Number'] = trn_raw['Summons Number']
test_set['Summons Number'] = test_raw['Summons Number']

# License Type
license_type = np.unique(trn_raw['License Type'])
license_type_idx = text2idx(license_type, range(len(license_type)))
trn_set['License Type'] = trn_raw['License Type'].map(license_type_idx)
license_type = np.unique(test_raw['License Type'])
license_type_idx = text2idx(license_type, range(len(license_type)))
test_set['License Type'] = test_raw['License Type'].map(license_type_idx)
'''
licensetype_dict = topdict(trn_raw['License Type'], 5)
trn_set['License Type'] = trn_raw['License Type'].apply(lambda x: text2frequency(licensetype_dict, x))
test_set['License Type'] = test_raw['License Type'].apply(lambda x: text2frequency(licensetype_dict, x))
'''

# Issue Date --- Month
trn_set['Issue Month'] = trn_raw['Issue Date'].apply(lambda x: findMonth(x))
test_set['Issue Month'] = test_raw['Issue Date'].apply(lambda x: findMonth(x))

# Issue Date --- Year
trn_set['Issue Year'] = trn_raw['Issue Date'].apply(lambda x: findYear(x))
test_set['Issue Year'] = test_raw['Issue Date'].apply(lambda x: findYear(x))

# Judgment Entry Date --- Month
trn_set['Judge Month'] = trn_raw['Judgment Entry Date'].apply(lambda x: findMonth(x))
test_set['Judge Month'] = test_raw['Judgment Entry Date'].apply(lambda x: findMonth(x))

# Judgment Entry Date --- Year
trn_set['Judge Year'] = trn_raw['Judgment Entry Date'].apply(lambda x: findYear(x))
test_set['Judge Year'] = test_raw['Judgment Entry Date'].apply(lambda x: findYear(x))

'''
trn_set['Day Judge'] = trn_raw['Judgment Entry Date'].apply(lambda x: findDayofWeek(x))
day_labels = np.unique(trn_set['Day Judge'])
#print(labels)
day_labels_idx = text2idx(day_labels, range(len(day_labels)))
trn_set['Day Judge'] = trn_set['Day Judge'].map(day_labels_idx)
test_set['Day Judge'] = test_raw['Judgment Entry Date'].apply(lambda x: findDayofWeek(x))
test_set['Day Judge'] = test_set['Day Judge'].map(day_labels_idx)
'''

state_dict = topDict(trn_raw['State'], 10)
trn_set['State'] = trn_raw['State'].apply(lambda x: text2frequency(state_dict, x))
test_set['State'] = test_raw['State'].apply(lambda x: text2frequency(state_dict, x))
'''
# State
labels = np.unique(trn_raw['State'])
labels_idx = text2idx(labels, range(len(labels)))
trn_set['State'] = trn_raw['State'].map(labels_idx)
labels = np.unique(test_raw['State'])
labels_idx = text2idx(labels, range(len(labels)))
test_set['State'] = test_raw['State'].map(labels_idx)
'''
# County
county_dict = topDict(trn_raw['County'], 10)
trn_set['County'] = trn_raw['County'].apply(lambda x: text2frequency(county_dict, x))
test_set['County'] = test_raw['County'].apply(lambda x: text2frequency(county_dict, x))
'''
labels_1 = np.unique(trn_set['County'] )
labels_idx_1 = text2idx(labels_1, range(len(labels_1)))
trn_set['County'] = trn_set['County'].map(labels_idx_1)
#labels1 = np.unique(test_set['County'] )
#labels_idx_1 = text2idx(labels, range(len(labels)))
test_set['County'] = test_set['County'].map(labels_idx_1)
'''
'''
agency_labels_trn = np.unique(trn_raw['Issuing Agency'])
violation_labels_idx_trn = text2idx(agency_labels_trn, range(len(agency_labels_trn)))
trn_set['Issuing Agency'] = trn_raw['Issuing Agency'].map(violation_labels_idx_trn)
agency_labels_test = np.unique(test_raw['Issuing Agency'])
violation_labels_idx_test = text2idx(agency_labels_test, range(len(agency_labels_test)))
test_set['Issuing Agency'] = test_raw['Issuing Agency'].map(violation_labels_idx_test)
'''
# Violation
violation_labels_trn = np.unique(trn_raw['Violation'])
violation_labels_idx_trn = text2idx(violation_labels_trn, range(len(violation_labels_trn)))
trn_set['Violation'] = trn_raw['Violation'].map(violation_labels_idx_trn)
violation_labels_test = np.unique(test_raw['Violation'])
violation_labels_idx_test = text2idx(violation_labels_test, range(len(violation_labels_test)))
test_set['Violation'] = test_raw['Violation'].map(violation_labels_idx_test)

# Plate
plate_dict = topDict(trn_raw['Plate'],10)
#print(plate_dict)
#print(trn_raw['Plate'].value_counts())
trn_set['Plate'] = trn_raw['Plate'].apply(lambda x: text2frequency(plate_dict, x))
test_set['Plate'] = test_raw['Plate'].apply(lambda x: text2frequency(plate_dict, x))


# 'County Precinct'

'''
## ONE HOT ENCODE
def text2onehot(list):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list)
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

county_num_trn = len(trn_raw['County'].unique())
precinct_num_trn = len(trn_raw['Precinct'].unique())
#np.savetxt('ptrn.csv', trn_raw['Precinct'].unique(), delimiter=",")
county_df_trn = pd.DataFrame(text2onehot(trn_raw['County']), columns=createName('County', county_num_trn))
precinct_df_trn = pd.DataFrame(text2onehot(trn_raw['Precinct']), columns=createName('Precinct', precinct_num_trn))
trn_set = pd.concat([trn_set, county_df_trn,precinct_df_trn], axis=1)
print(trn_set.shape)

county_num_test = len(test_raw['County'].unique())
precinct_num_test = len(test_raw['Precinct'].unique())
#np.savetxt('ptest.csv', test_raw['Precinct'].unique(), delimiter=",")
county_df_test = pd.DataFrame(text2onehot(test_raw['County']), columns=createName('County', county_num_test))
precinct_df_test = pd.DataFrame(text2onehot(test_raw['Precinct']), columns=createName('Precinct', precinct_num_test))
test_set = pd.concat([test_set, county_df_test, precinct_df_test], axis=1)
print(test_set.shape)
'''

labels = np.unique(trn_raw['Violation Status'])
labels_idx = text2idx(labels, range(len(labels)))
y_trn = trn_raw['Violation Status'].map(labels_idx)
X_trn = trn_set
X_test = test_set


## RANDOMFOREST MODEL
n_estimators = 1000
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=15, oob_score=True, n_jobs=-1)
rf_model = rf.fit(X_trn, y_trn)
y_pred = rf_model.predict(X_test)
print("Training Score:", rf.oob_score_)
#print(rf.feature_importances_)
idx_list = np.array(np.argsort(-rf.feature_importances_))
#print(idx_list)
print("Feature Importance Rank:\r\n", X_trn.columns.values[idx_list])

## TRANSLATE & SAVE PRDICTION
set = []
id = []
for i in range(len(y_pred)):
    id.append(i+1)
    set.append(list(labels_idx.keys())[list(labels_idx.values()).index(y_pred[i])])

name = ['ID', 'Prediction']
y_set = pd.DataFrame(columns=name)
y_set['ID'] = id
y_set['Prediction'] = set
y_set.to_csv('sub2.csv', encoding='gbk')

