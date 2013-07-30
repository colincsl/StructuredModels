"""
Main file for viewing data
"""

import os
import optparse
import time
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler
import skimage
import skimage.feature as sf

from pyKinectTools.dataset_readers.MSR_DailyActivities import MSRPlayer
from IPython import embed
from pylab import *

# separate unaries for markov and sm 
# setup metrics for comparison
# look at kitchen dataset?


class chi2_classifier:
	'''
	Train the input histogram data with a ch2-rbf kernel.
	Procedure:
		1) Convert input histogram for use in a chi2 kernel using the chi2 kernel approximation
	which outputs a modified feature vector.
		2) Train an SVM with an RBF kernel

	--Input--
	kernel, svm_c, chi2_sampler: parameters for the svm and the kernel

	Todo: use SGD Classifier
	  Problem: No multi-class probability support
	  Sln: check source version of sklearn? i think it might be in the dev version
	  frame_classifier = SGDClassifier(loss="log", penalty="l2")
	'''

	def __init__(self, kernel='rbf', svm_c=1000., chi2_sampler=1):
		self.chi2 = AdditiveChi2Sampler(chi2_sampler)
		self.clf = SVC(C=svm_c, kernel=kernel, probability=True)

	def fit(self, data, labels):
		'''
		data: a set of histograms for each sample
		labels: the truth for each sample
		'''
		data = [x/np.sum(x, dtype=np.float) for x in data]
		# data = [np.nan_to_num(x) for x in data]
		chi2_hists = self.chi2.fit_transform(data)
		self.clf.fit(chi2_hists, labels)

	def predict_proba(self, data):
		'''
		data: a set of histograms for each sample (not chi2)
		'''
		data = [x/np.sum(x, dtype=np.float) for x in data]
		chi2_hists = self.chi2.fit_transform(data)
		return self.clf.predict_proba(chi2_hists)

	def score(self, data, labels):
		data = [x/np.sum(x, dtype=np.float) for x in data]
		chi2_hists = self.chi2.fit_transform(data)
		return self.clf.score(chi2_hists, labels)


def calculate_hmm_params(label_sequences, n_gesture):
	'''
	Get the prior and transition matricies for a markov (or semi-markov) model.

	--Input--
	label_sequences: list/array of labels for each sequence
	n_gestures: integer for max number of gestures
	--Return--
	prior and transition matrix
	'''

	# Prior: how often are you in each state
	prior = np.histogram(np.hstack(label_sequences), n_gesture, [0,n_gesture])[0].astype(np.float)
	prior /= np.sum(prior)

	# State transitions: how often do you go from state i to j
	transition_matrix = np.zeros([n_gesture,n_gesture], np.float)
	# Go through each frame in each sequence sequence
	for labels in label_sequences:
		for i,lab in enumerate(labels):
			transition_matrix[lab,labels[i]] += 1
	# Normalize
	transition_matrix /= np.maximum(np.sum(transition_matrix, 0, dtype=np.float), .0001)

	return prior, transition_matrix


def preprocess_features(BoF_dict, features, frame_labels, n_frames_per_gesture):
	'''
	Turn features into histograms of words and put them in the correct format
	'''
	# For each frame, create a histogram of the features using the dictionary
	if BoF_dict is not None:
		predicted_features = [BoF_dict.predict(x) for x in features]
		n_clusters = BoF_dict.n_clusters
	else:
		predicted_features = features
		n_clusters = np.max([np.max(x) for x in features])

	n_features = np.shape(features[0])[0]/len(frame_labels[0])
	n_features_per_frame = frame_labels.copy()*0 + n_features

	# create histogram of features
	n_features_tmp = [np.cumsum(x) for x in n_features_per_frame]
	frame_splits = [np.split(x, y)[:-1] for x,y in zip(predicted_features, n_features_tmp)]
	# frame_splits = [np.split(x, y)[:-2] for x,y in zip(predicted_features, n_features_tmp)]
	# if len(x[i])>0
	frame_hists = [np.array([np.histogram(x[i], n_clusters, [0, n_clusters])[0] for i in xrange(len(x))], dtype=np.float) for x in frame_splits]
	for i,h in enumerate(frame_hists):
		for ii,j in enumerate(h):
			if np.all(frame_hists[i][ii]==0):
				frame_hists[i][ii] += 1
				# frame_hists[i][ii] /= np.sum(frame_hists[i][ii], dtype=np.float)

	# Using sum of features in each gesture, create histogram of features (w/ dictionary)
	n_frames_tmp = [np.cumsum(x) for x in n_frames_per_gesture]
	gesture_splits = [np.split(x, y)[:-1] for x,y in zip(frame_hists, n_frames_tmp)]
	gesture_hists = np.array([np.vstack([np.sum(x, 0, dtype=np.float) for x in y]) for y in gesture_splits])

	# Normalize
	frame_hists = np.array([np.array([np.array(x/np.sum(x, dtype=np.float)) for x in y]) for y in frame_hists])
	gesture_hists = np.array([np.array([x/np.sum(x, dtype=np.float) for x in y]) for y in gesture_hists])

	# frame_hists = np.array([nan_to_num(x) for x in frame_hists])
	# gesture_hists = np.array([nan_to_num(x) for x in gesture_hists])

	return frame_hists, gesture_hists


# def msm_preprocess_train(features, n_frames_per_gesture, frame_labels, gesture_labels, n_clusters=100, BoF_dict=None):
# 	'''
# 	x1) BoW Dictionary w/ KMeans
# 	2) Turn features into histograms of words
# 	3) Train unary SVMs for frames and gestures
# 	4) Compute prior and transition matricies for markov model
# 	'''
# 	# Create a dictionary from all of the features
# 	if BoF_dict is None:
# 		BoF_dict = KMeans(n_clusters=n_clusters, verbose=1)
# 		BoF_dict.fit(np.vstack(features))

# 	# Turn features into histograms
# 	frame_hists, gesture_hists = preprocess_features(BoF_dict, features, frame_labels, n_frames_per_gesture)

# 	# Calculate per-frame unary value using an SVM
# 	frame_classifier = chi2_classifier(kernel='rbf')
# 	frame_classifier.fit(np.vstack(frame_hists), np.hstack(frame_labels))

# 	# Calculate per-gesture unary value using an SVM
# 	gesture_classifier = chi2_classifier(kernel='rbf')
# 	gesture_classifier.fit(np.vstack(gesture_hists), np.hstack(gesture_labels))

# 	# Calculate HMM transitions for each frame and gesture
# 	n_gestures = len(np.unique(gesture_labels))
# 	frame_prior, frame_transition_matrix = calculate_hmm_params(frame_labels, n_gestures)
# 	gesture_prior, gesture_transition_matrix = calculate_hmm_params(gesture_labels, n_gestures)

# 	return frame_classifier, gesture_classifier,\
# 			frame_prior, frame_transition_matrix,\
# 			gesture_prior, gesture_transition_matrix,\
# 			frame_hists, gesture_hists


# def msm_preprocess_test(features, n_frames_per_gesture, frame_labels, gesture_labels, BoF_dict, frame_classifier, gesture_classifier):
# 	'''
# 	Turn features into histograms of words
# 	'''
# 	# Turn features into histograms
# 	frame_hists, gesture_hists = preprocess_features(BoF_dict, features, frame_labels, n_frames_per_gesture)

# 	# Calculate per-frame unary value using an SVM
# 	print "Unary (frame) score:", frame_classifier.score(np.vstack(frame_hists), np.hstack(frame_labels))
# 	print "Unary (gesture) score:", gesture_classifier.score(np.vstack(gesture_hists), np.hstack(gesture_labels))

# 	return frame_hists, gesture_hists




# -------------------------outline------------------------------------------
# ---Lingling/Luca---
## 1 - Features
# Calculate STIPs
# BoF_dict_learning_from_stip_file
# calculate_hist_from_stip_file
# ---
## 2 - Calculate Unary and histogram data
# unary_generate_vidfeat
# ---
## 3 - Learn SVM,HMM,sSVM
# train_kernel_svm
# calculate transition frequency:
# 	[prior1,transp1]=hmmtraining_trans(usedidx,trainidx,transpath);
# 	[~,transp2]=hmmtraining_trans_seg(usedidx,trainidx,transpath) ;
# learn parameter model_log{k} using structural svm:
# 	model_log{k}=structural_svm_max(param.struct_c,param.ssvm_max_iter,param);
# ---
## 4 - Test



# learn parameter model_log{k} using structural svm:
# 	model_log{k}=structural_svm_max(param.struct_c,param.ssvm_max_iter,param);

# C&F callback version #: param2 in example_script
	# 1-crf
	# 2-semi-crf
	# 3-msm-crf

# Loss function (Use lossCB (not lossCB1))

# Constraint Fcn
	# finds the most violated Constraint
	# do inference and add loss function

# Function CB: setup feature vector in svm^struct format
# y is current assumed set of labels

# Decode fcn:
	# constraint function without the loss fcn

# Don't use single slack variable! (use many) [optimization value/param -w]


