
import os
import optparse
import time
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC,SVC
import skimage
import skimage.feature as sf

from pyKinectTools.dataset_readers.MSR_DailyActivities import MSRPlayer
from pyKinectTools.utils.SkeletonUtils import msr_to_kinect_skel, get_skel_angles
from IPython import embed
from pylab import *

def smij(skels, n_top_joints=5):
	skel_angles = np.array([get_skel_angles(x) for x in skels])
	n_joint_angles = len(skel_angles[0])
	skel_angles_variance = [np.var(skel_angles[i:i+n_interval], 0) for i in range(0, len(skels), n_interval)]
	joints_ranked = [np.argsort(x)[::-1] for x in skel_angles_variance]

	# Rank joint angles by max variance
	features = []
	for i in range(len(skel_angles_variance)):
		joints_top_idx = joints_ranked[i][:n_top_joints]
		# joints_top_var = skel_angles_variance[i][joints_ranked[i][:n_top_joints]]
		# joint_histogram = np.histogram(np.hstack(joints_top_idx), n_joint_angles)
		features += [joints_top_idx]
	features = np.array(features)	
	
	return features

def im_to_blocks(img, blk_size=[48,64,1]):
	'''
	blk_size must be multiple of the image
	'''
	if len(img.shape) == 2:
		height, width = img.shape
		depth = 1
	else:
		height, width, depth = img.shape

	n_blk_height = height / blk_size[0]
	n_blk_width = width / blk_size[1]

	blocks = np.empty([depth, n_blk_height*n_blk_width, blk_size[0], blk_size[1]])
	for d in range(depth):
		for i in range(n_blk_height):
			for j in range(n_blk_width):
				if depth > 1:
					blocks[d,i*n_blk_width+j] = img[i*blk_size[0]:(i+1)*blk_size[0], j*blk_size[1]:(j+1)*blk_size[1], d]
				else:
					blocks[d,i*n_blk_width+j] = img[i*blk_size[0]:(i+1)*blk_size[0], j*blk_size[1]:(j+1)*blk_size[1]]

	if len(img.shape) == 2:
		blk = blk[0]

	return blocks


''' 1 - Calculate features
Calculate Daisy (dense sift) for each image in each sequence
'''
 # User parameters
n_clusters = 100    # for bag of words
n_interval = 5     # only use every n_interval frames
mask_images = False  # Whether to use features from the whole image or just on the user's body
feature_type = 'smij' # or 'daisy' or 'hog'

# Store features and meta data
features = []
frame_labels = []
gesture_labels = []
n_gestures = []
n_frame_features = []
n_frames_per_gesture = []

# Iterate through the same actions for different users
# note to self: I have up to user 10 on my computer
action_set = [1,2,5]
# subject_set = [1,2,3,4,5,6,7,8,9]
subject_set = [10]
# subject_set = [6,7,8,9,10]
# for subject in [1,2,3,4,5]:

for subject in subject_set:
	# Setup kinect data player
	cam = MSRPlayer(base_dir='/Users/colin/Data/MSR_DailyActivities/Data/', actions=action_set,subjects=[subject],
					get_depth=True, get_color=True, get_skeleton=True)

	features_all = []
	frame_labels_ = []
	gesture_labels_ = []
	n_frame_features_ = []
	n_frames_per_gesture_ = []

	# Go through each action sequence
	for i_action in xrange(len(cam.labels)+1):
		n_total_frames = cam.color_stack.shape[3]
		n_used_frames = np.ceil(float(n_total_frames) / n_interval).astype(np.int)
		features_ = []

		gray_stack = np.dstack([skimage.color.rgb2gray(cam.color_stack[:,:,:,i]) for i in xrange(n_total_frames)])
		# Black out the non-human areas in the image
		if mask_images:
			n_images = np.minimum(gray_stack.shape[2], cam.mask_stack.shape[2])
			gray_stack[:,:,:n_images] *= cam.mask_stack[:,:,:n_images]

		# Use joblib to parallelize the feature calculations
		if feature_type == 'daisy':
			features_ = np.array(joblib.Parallel(n_jobs=-1,verbose=1)(joblib.delayed(sf.daisy)(gray_stack[:,:,i], step=10,radius=20,rings=2) for i in range(0,gray_stack.shape[2],n_interval)))
			feature_dims = features_.shape[-1]
			features_ = features_.reshape([n_used_frames,-1,feature_dims])
		elif feature_type == 'hog':
			gray_stack = gray_stack[:,:,::n_interval]
			gray_blocks = im_to_blocks(gray_stack, [48*2,64*2,1])
			gray_blocks = np.concatenate(gray_blocks,0)
			features_ = np.array(joblib.Parallel(n_jobs=-1,verbose=1)(joblib.delayed(sf.hog)(gray_blocks[i,:,:], orientations=5) for i in range(0,gray_blocks.shape[0])))
			feature_dims = features_.shape[-1]
			features_ = features_.reshape([n_used_frames,-1,feature_dims])
		elif feature_type == 'smij':
			skels = cam.skel_stack
			n_top_joints = 5
			skel_angles = np.array([get_skel_angles(x) for x in skels])
			n_joint_angles = len(skel_angles[0])
			skel_angles_variance = [np.var(skel_angles[i:i+n_interval], 0) for i in range(0, len(skels), n_interval)]
			joints_ranked = [np.argsort(x)[::-1] for x in skel_angles_variance]

			# Rank joint angles by max variance
			for i in range(len(skel_angles_variance)):
				joints_top_idx = joints_ranked[i][:n_top_joints]
				# joints_top_var = skel_angles_variance[i][joints_ranked[i][:n_top_joints]]
				# joint_histogram = np.histogram(np.hstack(joints_top_idx), n_joint_angles)
				features_ += [joints_top_idx]
			features_ = np.array(features_)
			# Histogram of most informative joints: concatenate hist of rank=1, r=2,...
			# hmij = []
			# for i in range(n_top_joints):
			# 	top_idx = [x[i] for x in joints_ranked]
			# 	hist = np.histogram(np.hstack(top_idx), n_joint_angles)
			# 	hmij += [hist]
			# hmij = np.hstack(np.hstack(hmij))

		if len(features_) == 0:
			continue

		if mask_images:
			# Find which features are within the masked region
			valid_elements_count = np.all(np.abs(features_-1./feature_dims)>1e-3, 2).sum(1)
			features_ = np.vstack(features_)
			# Only use data from masked region
			valid_elements = np.all(np.abs(features_-1./feature_dims)>1e-3, 1)
			n_frame_features_ += [valid_elements_count]
			features_ = features_[valid_elements]
		else:
			n_frame_features_ += [[features_.shape[1]]*features_.shape[0]]

		# Store frame and gesture level activity labels
		labels = np.ones(n_used_frames, np.int)*cam.action_label
		features_all += [np.vstack(features_)]
		frame_labels_ += [labels]
		gesture_labels_ += [cam.action_label]
		n_frames_per_gesture_ += [n_used_frames]

		try:
			cam.next_sequence()
		except:
			print 'Done with subject {} action {}'.format(subject, i_action)
			pass

	# Reformat feature/label data
	features += [np.vstack(features_all).copy()]
	frame_labels += [np.hstack(frame_labels_)]
	gesture_labels += [np.hstack(gesture_labels_)]
	n_frame_features += [np.hstack(n_frame_features_)]
	n_frames_per_gesture += [np.hstack(n_frames_per_gesture_)]

# Todo: make labels 0...K
frame_labels = np.array(frame_labels) #make zero base
gesture_labels = np.vstack(gesture_labels) #make zero base
n_frame_features = np.array(n_frame_features)
n_frames_per_gesture = np.vstack(n_frames_per_gesture)

# Reformat labels
old_labels = np.unique(gesture_labels)
n_labels = len(np.unique(gesture_labels))
new_labels = range(n_labels)
convert_labels = {x:y for x,y in zip(old_labels, new_labels)}
frame_labels = np.array([np.array([convert_labels[x] for x in y]) for y in frame_labels])
gesture_labels = np.array([np.array([convert_labels[x] for x in y]) for y in gesture_labels])

n_gestures = n_labels

try:
	BoF_dict = KMeans(n_clusters=n_clusters, verbose=1)
	BoF_dict.fit(np.vstack(features))
except:
	BoF_dict = None
	print 'No dictionary created'

# Save feature data so it doens't have to be recomputed every time
save = [features, frame_labels, gesture_labels, n_gestures, n_frame_features, n_frames_per_gesture, BoF_dict]
action_text = "".join([str(x) for x in action_set])
subject_text = "".join([str(x) for x in subject_set])
np.savez('/Users/colin/Desktop/msm_features/{}_a{}_s{}'.format(feature_type, action_text, subject_text), save)
