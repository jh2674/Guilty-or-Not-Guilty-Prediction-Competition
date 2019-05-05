import json
import nolds
import numpy as np
#import tsfresh
#from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
#from tsfresh import extract_relevant_features

filepath = 'pos.json'
with open(filepath) as fp:
    line = fp.readline()

    data = json.loads(line)
    cnt = 1
    while line:
        line = fp.readline()
        if len(line) > 5:

            # every line data
            data = json.loads(line)
            # print(line)

            # data reshape
            acc_head_raw = np.array(data['acc_head'])
            acc_right_raw = np.array(data['acc_right'])
            acc_left_raw = np.array(data['acc_left'])
            gyro_head_raw = np.array(data['gyro_head'])
            gyro_right_raw = np.array(data['gyro_right'])
            gyro_left_raw = np.array(data['gyro_left'])
            quat_head_raw = np.array(data['quat_head'])
            quat_right_raw = np.array(data['quat_right'])
            quat_left_raw = np.array(data['quat_left'])

            acc_all_raw = np.hstack((acc_head_raw, acc_right_raw, acc_left_raw))
            gyro_all_raw = np.hstack((gyro_head_raw, gyro_right_raw, gyro_left_raw))
            quat_all_raw = np.hstack((quat_head_raw, quat_right_raw, quat_left_raw))

            all_raw_pos = np.hstack((acc_all_raw, gyro_all_raw, quat_all_raw))
            #all_raw_pos = np.hstack((gyro_all_raw, quat_all_raw))
            #all_raw_pos = np.hstack((acc_all_raw, quat_all_raw))
            #all_raw_pos = np.hstack((acc_all_raw, gyro_all_raw))
            #all_raw_pos = quat_all_raw

            # feature extract
            # FEATURE 1: MEAN
            all_mean = np.mean(all_raw_pos, axis=0)
            # FEATURE 2: MAX
            all_max = np.max(all_raw_pos, axis=0)
            # FEATURE 3: MIN
            all_min = np.min(all_raw_pos, axis=0)
            # FEATURE 4: VAR
            all_var = np.var(all_raw_pos, axis=0)
            # FEATURE 5: MEDIAN
            all_med = np.median(all_raw_pos, axis=0)
            # FEATURE 6: SKEW
            all_skew = skew(all_raw_pos, axis=0)
            # FEATURE 7: KURIOSIS
            all_kuriosis = kurtosis(all_raw_pos, axis=0)
            # FEATURE 8: SAMPLE ENTROPY
            all_se = nolds.sampen(all_raw_pos)
            # FEATURE 9: PCA
            #pca = PCA(n_components=30)
            #pca.fit(all_raw)
            #all_pca = pca.components_[1,:]
            # FEATURE 10: FFT
            quat_head = np.transpose(np.array(data['quat_head']))
            head_fft = np.absolute(np.sqrt(np.sum(np.square(np.fft.fft(quat_head, axis=1)), axis=0)))[1:6]
            quat_left = np.transpose(np.array(data['quat_left']))
            left_fft = np.absolute(np.sqrt(np.sum(np.square(np.fft.fft(quat_left, axis=1)), axis=0)))[1:6]
            quat_right = np.transpose(np.array(data['quat_right']))
            right_fft = np.absolute(np.sqrt(np.sum(np.square(np.fft.fft(quat_right, axis=1)), axis=0)))[1:6]
            all_fft = np.hstack((head_fft, left_fft, right_fft))

            # COMBINE ALL FEATURES
            all_pos = np.hstack((all_mean, all_max, all_min, all_var, all_med, all_skew, all_kuriosis, all_se))
            #all_pos = np.hstack((all_mean, all_max, all_min, all_var, all_med, all_skew, all_kuriosis))
            # feature store
            if cnt == 1:
                raw_vec_pos = all_raw_pos
                feat_vec_pos = all_pos
            else:
                raw_vec_pos = np.vstack((raw_vec_pos, all_raw_pos))
                feat_vec_pos = np.vstack((feat_vec_pos, all_pos))

            cnt += 1

        else:
            break
    #print(raw_vec.shape)
    #print(all_min.shape)

print('pos feature vector:', feat_vec_pos.shape)
    #print('raw data:', raw_vec.shape)




filepath = 'neg.json'
with open(filepath) as fp:
    line = fp.readline()

    data = json.loads(line)
    cnt = 1
    while line:
        line = fp.readline()
        if len(line) > 5:

            # every line data
            data = json.loads(line)
            # print(line)

            # data reshape
            acc_head_raw = np.array(data['acc_head'])
            acc_right_raw = np.array(data['acc_right'])
            acc_left_raw = np.array(data['acc_left'])
            gyro_head_raw = np.array(data['gyro_head'])
            gyro_right_raw = np.array(data['gyro_right'])
            gyro_left_raw = np.array(data['gyro_left'])
            quat_head_raw = np.array(data['quat_head'])
            quat_right_raw = np.array(data['quat_right'])
            quat_left_raw = np.array(data['quat_left'])

            acc_all_raw = np.hstack((acc_head_raw, acc_right_raw, acc_left_raw))
            gyro_all_raw = np.hstack((gyro_head_raw, gyro_right_raw, gyro_left_raw))
            quat_all_raw = np.hstack((quat_head_raw, quat_right_raw, quat_left_raw))

            all_raw_neg = np.hstack((acc_all_raw, gyro_all_raw, quat_all_raw))
            #all_raw_neg = np.hstack((gyro_all_raw, quat_all_raw))
            #all_raw_neg = np.hstack((acc_all_raw, quat_all_raw))
            #all_raw_neg = np.hstack((acc_all_raw, gyro_all_raw))
            #all_raw_neg = quat_all_raw

            # feature extract
            # FEATURE 1: MEAN
            all_mean = np.mean(all_raw_neg, axis=0)
            # FEATURE 2: MAX
            all_max = np.max(all_raw_neg, axis=0)
            # FEATURE 3: MIN
            all_min = np.min(all_raw_neg, axis=0)
            # FEATURE 4: VAR
            all_var = np.var(all_raw_neg, axis=0)
            # FEATURE 5: MEDIAN
            all_med = np.median(all_raw_neg, axis=0)
            # FEATURE 6: SKEW
            all_skew = skew(all_raw_neg, axis=0)
            # FEATURE 7: KURIOSIS
            all_kuriosis = kurtosis(all_raw_neg, axis=0)
            # FEATURE 8: SAMPLE ENTROPY
            all_se = nolds.sampen(all_raw_neg)
            # FEATURE 9: PCA
            #pca = PCA(n_components=30)
            #pca.fit(all_raw)
            #all_pca = pca.components_[1,:]

            # FEATURE 10: FFT
            quat_head = np.transpose(np.array(data['quat_head']))
            head_fft = np.absolute(np.sqrt(np.sum(np.square(np.fft.fft(quat_head, axis=1)), axis=0)))[1:6]
            quat_left = np.transpose(np.array(data['quat_left']))
            left_fft = np.absolute(np.sqrt(np.sum(np.square(np.fft.fft(quat_left, axis=1)), axis=0)))[1:6]
            quat_right = np.transpose(np.array(data['quat_right']))
            right_fft = np.absolute(np.sqrt(np.sum(np.square(np.fft.fft(quat_right, axis=1)), axis=0)))[1:6]
            all_fft = np.hstack((head_fft, left_fft, right_fft))

            # COMBINE ALL FEATURES
            all_neg = np.hstack((all_mean, all_max, all_min, all_var, all_med, all_skew, all_kuriosis, all_se))
            #all_neg = np.hstack((all_mean, all_max, all_min, all_var, all_med, all_skew, all_kuriosis))


            # feature store
            if cnt == 1:
                raw_vec_neg = all_raw_neg
                feat_vec_neg = all_neg
            else:
                raw_vec_neg = np.vstack((raw_vec_neg, all_raw_neg))
                feat_vec_neg = np.vstack((feat_vec_neg, all_neg))

            cnt += 1

        else:
            break

print('neg feature vector:', feat_vec_neg.shape)

feat_vec = np.vstack((feat_vec_pos, feat_vec_neg))
print('feature vector:', feat_vec.shape)

#df = pd.DataFrame(feat_vec, columns=['acc_R_1', 'b', 'c'])


from sklearn.ensemble import RandomForestClassifier

X_trn = feat_vec
y_trn = np.vstack((np.ones([363,1]),np.ones([747,1])*-1))
#print('y_trn:', y_trn.shape)
n_estimators = 200
rf = RandomForestClassifier(n_estimators = n_estimators, max_depth=15, oob_score=True)
rf_model = rf.fit(X_trn, y_trn)
print(rf.oob_score_)
print(rf.feature_importances_)
idx = np.argsort(rf.feature_importances_)
print(idx)
#y_pred = rf_model.predict(X_test)

print(head_fft.shape)