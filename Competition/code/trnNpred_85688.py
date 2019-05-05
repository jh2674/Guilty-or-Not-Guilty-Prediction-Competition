import numpy as np
import pandas as pd
import re
import datetime
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

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
    round_time = minute*60
    return round_time

def time2cos(time,type):
    if type == 1:   # hour
        return math.cos(2*math.pi*time/24)
    elif type == 2: # minute
        return math.cos(2*math.pi*time/60)
    elif type == 3: # month
        return math.cos(2*math.pi*time/12)
    else: # day of week
        return math.cos(2*math.pi*time/7)

def time2sin(time,type):
    if type == 1:   # hour
        return math.sin(2*math.pi*time/24)
    elif type == 2: # minute
        return math.sin(2*math.pi*time/60)
    elif type == 3: # month
        return math.sin(2*math.pi*time/12)
    else: # day of week
        return math.sin(2*math.pi*time/7)


# LOAD DATA SET
trn_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

#trn_raw['Violation'].value_counts().to_csv('violation_trn.csv', encoding='gbk')
#test_raw['Violation'].value_counts().to_csv('violation_test.csv', encoding='gbk')

'''
# STATISTIC ANALYSIS
print("--------------------------------------")
violation_set =pd.concat([trn_raw['Violation'],test_raw['Violation']],ignore_index=True)
print(violation_set.value_counts())

print(trn_raw.shape)

row_copy_1 = trn_raw[trn_raw['Violation'] == 'ANGLE PARKING-COMM VEHICLE'] # 1
row_copy_2 = trn_raw[trn_raw['Violation'] == 'DIVIDED HIGHWAY '] # 3
row_copy_3 = trn_raw[trn_raw['Violation'] == 'ELEVATED/DIVIDED HIGHWAY/TUNNL'] # 1
row_copy_4 = trn_raw[trn_raw['Violation'] == 'NO OPERATOR NAM/ADD/PH DISPLAY'] # 4
row_copy_5 = trn_raw[trn_raw['Violation'] == 'NO STANDING EXCP DP'] # 1
row_copy_6 = trn_raw[trn_raw['Violation'] == 'WASH/REPAIR VEHCL-REPAIR ONLY'] #1
row_copy_7 = trn_raw[trn_raw['Violation'] == 'OT PARKING-MISSING/BROKEN METR'] #2
row_copy_8 = trn_raw[trn_raw['Violation'] == 'EXPIRED METER']
row_copy_9 = trn_raw[trn_raw['Violation'] == 'OVERNIGHT TRACTOR TRAILER PKG']
row_copy_10 = trn_raw[trn_raw['Violation'] == 'EXCAVATION-VEHICLE OBSTR TRAFF']
row_copy_11 = trn_raw[trn_raw['Violation'] == 'NO PARKING-EXC. HNDICAP PERMIT']
row_copy_12 = trn_raw[trn_raw['Violation'] == 'NO STOP/STANDNG EXCEPT PAS P/U']
row_copy_13 = trn_raw[trn_raw['Violation'] == 'VEHICLE FOR SALE(DEALERS ONLY)']
violation_rows_copy = pd.concat([row_copy_1, row_copy_3, row_copy_5, row_copy_6], ignore_index=True)
print(violation_rows_copy)
trn_raw = pd.concat([trn_raw,violation_rows_copy], ignore_index=True)

print("--------------------------------------")
# print("License Type", trn_raw['License Type'].value_counts())
# print("--------------------------------------")
# print("County", trn_raw['County'].value_counts())
# print("--------------------------------------")
# print("Precinct", trn_raw['Precinct'].value_counts())
# print("--------------------------------------")
# trn_raw['Precinct'].value_counts().to_csv('precinct.csv', encoding='gbk')
#print(len(trn_raw['State'].value_counts()))
'''
print(trn_raw['Violation Status'].value_counts())
rows_copy1 = trn_raw[trn_raw['Violation Status'] == 'APPEAL MODIFIED']
rows_copy2 = trn_raw[trn_raw['Violation Status'] == 'HEARING WAIVED']
rows_copy = pd.concat([rows_copy1,rows_copy2],ignore_index=True)
trn_raw = pd.concat([trn_raw, rows_copy, rows_copy, rows_copy, rows_copy, rows_copy], ignore_index=True)
print(trn_raw.shape)
#print(trn_raw['Violation Status'].value_counts())


# DATA PREPROCESSING(FILL NAN)
trn_raw['Violation'] = trn_raw['Violation'].fillna(method='ffill')
test_raw['Violation'] = test_raw['Violation'].fillna(method='ffill')
trn_raw['Issuing Agency'] = trn_raw['Issuing Agency'].fillna(method='ffill')
test_raw['Issuing Agency'] = test_raw['Issuing Agency'].fillna(method='ffill')
trn_raw['County'] = trn_raw['County'].fillna("UNKNOWN")
test_raw['County'] = test_raw['County'].fillna("UNKNOWN")
trn_raw['Precinct'] = trn_raw['Precinct'].fillna(method='ffill')
test_raw['Precinct'] = test_raw['Precinct'].fillna(method='ffill')
trn_raw['License Type'] = trn_raw['License Type'].fillna(method='ffill')
test_raw['License Type'] = test_raw['License Type'].fillna(method='ffill')
trn_raw['Violation Time'] = trn_raw['Violation Time'].fillna(method='ffill')
test_raw['Violation Time'] = test_raw['Violation Time'].fillna(method='ffill')
trn_raw['Judgment Entry Date'] = trn_raw['Judgment Entry Date'].fillna("1/1/1000")
test_raw['Judgment Entry Date'] = test_raw['Judgment Entry Date'].fillna("1/1/1000")

#trn_raw = trn_raw[trn_raw['Violation'] != 'VACANT LOT']

# COLUMN NAME
subname1 = ['Payment Amount', 'Reduction Amount', 'Amount Due', 'Penalty Amount', 'Fine Amount', 'Interest Amount']
subname2 = ['Violation', 'State', 'License Type', 'County', 'Issue Month Cos', 'Issue Month Sin', 'Issue Year', 'Summons Number']
subname3 = ['Judge Year', 'Judge Month Cos', 'Judge Month Sin', 'Plate', 'Issuing Agency']
subname4 = ['Payment Rest']
name = np.hstack((subname1, subname2, subname3))

# CREATE EMPTY TRAINING & TESTING SET
trn_set = pd.DataFrame(columns=name)
test_set = pd.DataFrame(columns=name)

# FEATURES EXTRACTION
# FEATURE CATEGORY 1: Money & Summons Number
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
trn_set['Payment Rest'] = trn_raw['Payment Amount']-trn_raw['Amount Due']
test_set['Payment Rest'] = test_raw['Payment Amount']-test_raw['Amount Due']
#trn_set['Precinct'] = trn_raw['Precinct']
#test_set['Precinct'] = test_raw['Precinct']

#  FEATURE CATEGORY 2: License Type
license_type = np.unique(pd.concat([trn_raw['License Type'], test_raw['License Type']], ignore_index=True))
license_type_idx = text2idx(license_type, range(len(license_type)))
trn_set['License Type'] = trn_raw['License Type'].map(license_type_idx)
test_set['License Type'] = test_raw['License Type'].map(license_type_idx)

# FEATURE CATEGORY 3: Issue Date & Judgment Entry Date (Year & Month)
trn_set['Issue Year'] = trn_raw['Issue Date'].apply(lambda x: findYear(x))
test_set['Issue Year'] = test_raw['Issue Date'].apply(lambda x: findYear(x))
#trn_set['Issue Month'] = trn_raw['Issue Date'].apply(lambda x: findMonth(x))
#test_set['Issue Month'] = test_raw['Issue Date'].apply(lambda x: findMonth(x))
issuemonth_set_trn = trn_raw['Issue Date'].apply(lambda x: findMonth(x))
trn_set['Issue Month Cos'] = issuemonth_set_trn.apply(lambda x: time2cos(x, 3))
trn_set['Issue Month Sin'] = issuemonth_set_trn.apply(lambda x: time2sin(x, 3))
issuemonth_set_test = test_raw['Issue Date'].apply(lambda x: findMonth(x))
test_set['Issue Month Cos'] = issuemonth_set_test.apply(lambda x: time2cos(x, 3))
test_set['Issue Month Sin'] = issuemonth_set_test.apply(lambda x: time2sin(x, 3))
#trn_set['Issue Day'] = trn_raw['Issue Date'].apply(lambda x: findDay(x))
#test_set['Issue Day'] = test_raw['Issue Date'].apply(lambda x: findDay(x))


trn_set['Judge Year'] = trn_raw['Judgment Entry Date'].apply(lambda x: findYear(x))
test_set['Judge Year'] = test_raw['Judgment Entry Date'].apply(lambda x: findYear(x))
#trn_set['Judge Month'] = trn_raw['Judgment Entry Date'].apply(lambda x: findMonth(x))
#test_set['Judge Month'] = test_raw['Judgment Entry Date'].apply(lambda x: findMonth(x))
judgemonth_set_trn = trn_raw['Judgment Entry Date'].apply(lambda x: findMonth(x))
trn_set['Judge Month Cos'] = judgemonth_set_trn.apply(lambda x: time2cos(x, 3))
trn_set['Judge Month Sin'] = judgemonth_set_trn.apply(lambda x: time2sin(x, 3))
judgemonth_set_test = test_raw['Judgment Entry Date'].apply(lambda x: findMonth(x))
test_set['Judge Month Cos'] = judgemonth_set_test.apply(lambda x: time2cos(x, 3))
test_set['Judge Month Sin'] = judgemonth_set_test.apply(lambda x: time2sin(x, 3))
#trn_set['Judge Day'] = trn_raw['Judgment Entry Date'].apply(lambda x: findDay(x))
#test_set['Judge Day'] = test_raw['Judgment Entry Date'].apply(lambda x: findDay(x))


# FEATURE CATEGORY 4: State & County
state_labels = np.unique(pd.concat([trn_raw['State'], test_raw['State']], ignore_index=True))
state_idx = text2idx(state_labels, range(len(state_labels)))
trn_set['State'] = trn_raw['State'].map(state_idx)
test_set['State'] = test_raw['State'].map(state_idx)

county_labels = np.unique(pd.concat([trn_raw['County'], test_raw['County']], ignore_index=True))
county_idx = text2idx(county_labels, range(len(county_labels)))
trn_set['County'] = trn_raw['County'].map(county_idx)
test_set['County'] = test_raw['County'].map(county_idx)

# FEATURE CATEGORY 4: Issue Agency
agency_labels = np.unique(pd.concat([trn_raw['Issuing Agency'], test_raw['Issuing Agency']], ignore_index=True))
agency_idx = text2idx(agency_labels, range(len(agency_labels)))
trn_set['Issuing Agency'] = trn_raw['Issuing Agency'].map(agency_idx)
test_set['Issuing Agency'] = test_raw['Issuing Agency'].map(agency_idx)


# FEATURE CATEGORY 5: Violation
violation_labels = np.unique(pd.concat([trn_raw['Violation'],test_raw['Violation']],ignore_index=True))
violation_labels_idx = text2idx(violation_labels, range(len(violation_labels)))
trn_set['Violation'] = trn_raw['Violation'].map(violation_labels_idx)
test_set['Violation'] = test_raw['Violation'].map(violation_labels_idx)

# FEATURE CATEGORY 6: Plate
labels_trn = np.unique(trn_raw['Plate'])
plate_dict_trn = topDict(trn_raw['Plate'],labels_trn.shape[0])
trn_set['Plate'] = trn_raw['Plate'].apply(lambda x: text2frequency(plate_dict_trn, x))
labels_test = np.unique(test_raw['Plate'])
plate_dict_test = topDict(test_raw['Plate'],labels_test.shape[0])
test_set['Plate'] = test_raw['Plate'].apply(lambda x: text2frequency(plate_dict_test, x))


def findHourPeriod(hour):
    if hour >= 0 and hour < 4:
        return 0
    elif hour >= 4 and hour < 8:
        return 1
    elif hour >= 8 and hour < 12:
        return 2
    elif hour >= 12 and hour < 16:
        return 3
    elif hour >= 16 and hour < 20:
        return 4
    else:
        return 5

def findMinutePeriod(minute):
    if minute >= 0 and minute < 20:
        return 0
    elif minute >= 20 and minute < 40:
        return 1
    else:
        return 2
    '''
    elif minute >= 30 and minute < 40:
        return 3
    elif minute >= 40 and minute < 50:
        return 4
    else:
        return 5
    '''


# FEATURE CATEGORY 7: Violation Time(Hour Period)
#trn_set['Hour'] = trn_raw['Violation Time'].apply(lambda x: getHour(x))
#trn_set['Hour Period'] = trn_set['Hour'].apply(lambda x: findHourPeriod(x))
#trn_set['Hour Cos'] = trn_set['Hour'].apply(lambda x: time2cos(x, 1))
#trn_set['Hour Sin'] = trn_set['Hour'].apply(lambda x: time2sin(x, 1))
#test_set['Hour'] = test_raw['Violation Time'].apply(lambda x: getHour(x))
#test_set['Hour Period'] = test_set['Hour'].apply(lambda x: findHourPeriod(x))
#test_set['Hour Cos'] = test_set['Hour'].apply(lambda x: time2cos(x, 1))
#test_set['Hour Sin'] = test_set['Hour'].apply(lambda x: time2sin(x, 1))


'''
minute_set_trn = trn_raw['Violation Time'].apply(lambda x: getMinute(x))
trn_set['Minute Cos'] = minute_set_trn.apply(lambda x: time2cos(x, 2))
trn_set['Minute Sin'] = minute_set_trn.apply(lambda x: time2sin(x, 2))
#trn_set['Minute Period'] = trn_set['Minute Period'].apply(lambda x: findMinutePeriod(x))
minute_set_test = test_raw['Violation Time'].apply(lambda x: getMinute(x))
test_set['Minute Cos'] = minute_set_test.apply(lambda x: time2cos(x, 2))
test_set['Minute Sin'] = minute_set_test.apply(lambda x: time2sin(x, 2))
#test_set['Minute Period'] = test_set['Minute Period'].apply(lambda x: findMinutePeriod(x))
'''

# LABLE PROCESSING
labels = np.unique(trn_raw['Violation Status'])
labels_idx = text2idx(labels, range(len(labels)))
y_trn = trn_raw['Violation Status'].map(labels_idx)
X_trn = trn_set
X_test = test_set


## RANDOMFOREST MODEL
n_estimators = 1000
rf = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', max_depth=15, oob_score=True, n_jobs=-1, max_features='sqrt')
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
