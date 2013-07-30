
import numpy as np
# import dynamicalModels.models
from dynamicalModels.models.MarkovCRF import MarkovCRF
from dynamicalModels.models.MarkovSemiMarkovCRF import MarkovSemiMarkovCRF
from dynamicalModels.models.SemiMarkovCRF import SemiMarkovCRF
from dynamicalModels.models.CRF_Functions import *#msm_preprocess_train, msm_preprocess_test
from pystruct.learners import NSlackSSVM,SubgradientSSVM

import os
import optparse
import time

from pyKinectTools.dataset_readers.MSR_DailyActivities import MSRPlayer
from IPython import embed
from pylab import *

subjects_train = '12345'
subjects_test = '678910'
# subjects_train = '123456789'
# subjects_test = '10'
actions = '125'
feature_type = 'smij'

''' Training data '''
## Load in feature data if pre-trained
filename = '/Users/colin/Desktop/msm_features/{}_a{}_s{}.npz'.format(feature_type, actions, subjects_train)
data = np.load(filename)
features, frame_labels, gesture_labels, n_gestures, n_frame_features, n_frames_per_gesture, BoF_dict_train = data.items()[0][1]

frame_hists_train, gesture_hists_train = preprocess_features(None, features, frame_labels, n_frames_per_gesture)

# Calculate per-frame unary value using an SVM
frame_clf_train = chi2_classifier(kernel='rbf')
frame_clf_train.fit(np.vstack(frame_hists_train), np.hstack(frame_labels))

# Calculate per-gesture unary value using an SVM
gesture_clf_train = chi2_classifier(kernel='rbf')
gesture_clf_train.fit(np.vstack(gesture_hists_train), np.hstack(gesture_labels))

# Calculate HMM transitions for each frame and gesture
n_gestures = len(np.unique(gesture_labels))
frame_prior_train, frame_transition_matrix_train = calculate_hmm_params(frame_labels, n_gestures)
gesture_prior_train, gesture_transition_matrix_train = calculate_hmm_params(gesture_labels, n_gestures)

print "Unary (frame) score:", frame_clf_train.score(np.vstack(frame_hists_train), np.hstack(frame_labels))
print "Unary (gesture) score:", gesture_clf_train.score(np.vstack(gesture_hists_train), np.hstack(gesture_labels))

gesture_transition_matrix_train = np.ones([n_gestures,3])/3.

# Markov CRF
markovCRF = MarkovCRF(n_states=n_gestures, clf=frame_clf_train,
				 prior=frame_prior_train, transition=frame_transition_matrix_train,
				 inference_method='dai')
markov_svm = SubgradientSSVM(markovCRF, verbose=1, C=1., n_jobs=1)
markov_svm.fit(frame_hists_train, frame_labels)
m_predict = markov_svm.predict(frame_hists_train)
print 'Markov w:', markov_svm.w
print 'Markov CRF score: {}%'.format(100*np.sum([np.sum(np.equal(m_predict[i],x)) for i,x in enumerate(frame_labels)])  / np.sum([np.size(x) for x in frame_labels], dtype=np.float))

# semi-Markov CRF
sm_crf = SemiMarkovCRF(n_states=n_gestures,clf=gesture_clf_train,
				 prior=gesture_prior_train, transition_matrix=gesture_transition_matrix_train)
sm_svm = SubgradientSSVM(sm_crf, verbose=1, C=1., n_jobs=1)
sm_svm.fit(frame_hists_train, frame_labels)
sm_predict = sm_svm.predict(frame_hists_train)
print 'Semi-Markov w:', sm_svm.w
print 'Semi-Markov CRF score: {}%'.format(100*np.sum([np.sum(sm_predict[i]==x) for i,x in enumerate(frame_labels)])  / np.sum([np.size(x) for x in frame_labels], dtype=np.float))

# Markov semi-Markov CRF
MarkovSemiMarkovCRF = MarkovSemiMarkovCRF(n_states=n_gestures,
				 markov_prior=frame_prior_train, markov_transition=frame_transition_matrix_train,
				 semi_markov_prior=gesture_prior_train, semi_markov_transition=gesture_transition_matrix_train,
				 markov_clf=frame_clf_train,semi_markov_clf=gesture_clf_train)
msm_svm = SubgradientSSVM(MarkovSemiMarkovCRF, verbose=1, C=1., n_jobs=1)
msm_svm.fit(frame_hists_train, frame_labels)
msm_predict = msm_svm.predict(frame_hists_train)
print 'MsM w:', msm_svm.w
print 'MsM-CRF score: {}%'.format(100*np.sum([np.sum(msm_predict[i]==x) for i,x in enumerate(frame_labels)])  / np.sum([np.size(x) for x in frame_labels], dtype=np.float))

for i in range(len(subjects_train)):
	print 'i', i
	print 'm  ', m_predict[i]
	print 'sm ', sm_predict[i]
	print 'msm', msm_predict[i]
	print 'tru', np.array(frame_labels[i])
	print ""

print ""
print "SVM Weights"
print 'Markov w:', markov_svm.w
print 'Semi-Markov w:', sm_svm.w
print 'MsM w:', msm_svm.w
print ""
print "SCORES"
print 'Markov CRF score: {}%'.format(100*np.sum([np.sum(np.equal(m_predict[i],x)) for i,x in enumerate(frame_labels)])  / np.sum([np.size(x) for x in frame_labels], dtype=np.float))
print 'Semi-Markov CRF score: {}%'.format(100*np.sum([np.sum(sm_predict[i]==x) for i,x in enumerate(frame_labels)])  / np.sum([np.size(x) for x in frame_labels], dtype=np.float))
print 'MsM-CRF score: {}%'.format(100*np.sum([np.sum(msm_predict[i]==x) for i,x in enumerate(frame_labels)])  / np.sum([np.size(x) for x in frame_labels], dtype=np.float))


''' ------------------------------------------------- '''


''' Testing data '''
filename = '/Users/colin/Desktop/msm_features/{}_a{}_s{}.npz'.format(feature_type, actions, subjects_test)
data = np.load(filename)
features_test, frame_labels_test, gesture_labels_test, n_gestures_test, n_frame_features_test, n_frames_per_gesture_test, BoF_dict_test = data.items()[0][1]

frame_hists_test, _ = preprocess_features(None, features_test, frame_labels_test, n_frames_per_gesture_test)

# Evaluate models
m_predict = markov_svm.predict(frame_hists_test)
sm_predict = sm_svm.predict(frame_hists_test)
msm_predict = msm_svm.predict(frame_hists_test)

for i in range(len(subjects_test)-1):
	print 'i', i
	print 'm  ', m_predict[i]
	print 'sm ', sm_predict[i]
	print 'msm', msm_predict[i]
	print 'tru', np.array(frame_labels[i])
	print ""

print ""
print "EXPERIMENT:"
print "TRAIN -- Subjects {:5} -- Actions {}".format(",".join(subjects_train), ",".join(actions))
print "TEST -- Subjects {:5} -- Actions {}".format(",".join(subjects_test), ",".join(actions))
print ""
print "SCORES"
print 'Markov CRF score: {}%'.format(100*np.sum([np.sum(np.equal(m_predict[i],x)) for i,x in enumerate(frame_labels_test)])  / np.sum([np.size(x) for x in frame_labels_test], dtype=np.float))
print 'Semi-Markov CRF score: {}%'.format(100*np.sum([np.sum(sm_predict[i]==x) for i,x in enumerate(frame_labels_test)])  / np.sum([np.size(x) for x in frame_labels_test], dtype=np.float))
print 'MsM-CRF score: {}%'.format(100*np.sum([np.sum(msm_predict[i]==x) for i,x in enumerate(frame_labels_test)])  / np.sum([np.size(x) for x in frame_labels_test], dtype=np.float))


# Plot unaries:
if 0:
	for i in range(5):
		subplot(2,5, i+1)
		plot(frame_unary_train[i])
		subplot(2,5, i+1+5)
		plot(gesture_unary_train[i])

