import numpy as np
import pystruct.models
from pystruct.models import StructuredModel
from IPython import embed
from pystruct.learners import NSlackSSVM,SubgradientSSVM


class MarkovSemiMarkovCRF(StructuredModel):
	markov_prior = None
	markov_transition = None
	semi_markov_prior = None
	semi_markov_transition = None

	def __init__(self, markov_prior, markov_transition,
				 semi_markov_prior, semi_markov_transition,
				 markov_clf, semi_markov_clf, normalize=False,
	 			n_states=2, n_features=None, inference_method='qpbo',
                 class_weight=None):
		self.size_psi = 4

		self.markov_prior = markov_prior
		self.semi_markov_prior = semi_markov_prior

		self.markov_clf = markov_clf
		self.semi_markov_clf = semi_markov_clf

		# NOTE: a value of 0 is set to .000001 for log numerical reasons
		self.markov_transition = np.maximum(markov_transition, .001)
		self.semi_markov_transition = np.maximum(semi_markov_transition, .001)

		self.normalize = normalize

		# Boiler plate
		self.n_states = n_states
		self.inference_method = inference_method
		self.inference_calls = 0
		if n_features is None:
			# backward compatibilty hack
			n_features = n_states
		self.n_features = n_features

		if class_weight is not None:
			if hasattr(self, 'n_labels'):
				n_things = self.n_labels
			else:
				n_things = n_states

			if len(class_weight) != n_things:
				raise ValueError("class_weight must have length n_states or"
								 " be None")
			self.class_weight = np.array(class_weight)
		else:
			self.class_weight = np.ones(n_states)


	def psi(self, x, y):
		'''
		psi() = [
				Markov:
				 sum_1^T(psi_markov_unary) =
				 sum_1^T(psi_markov_pairwise) = log(transition_matrix_ij)
				Semi-markov
				 sum_1^M(psi_semi_markov_unary)
				 sum_1^M(psi_semi_markov_pairwise) = log(transition_matrix_ij))
				]
		'''

		data = x
		# Get markov unaries
		markov_unary = self.markov_clf.predict_proba(data)

		# Get semi-markov unaries by finding each segment in the training data
		segment_idx = [0] + [i+1 for i,(y0,y1) in enumerate(zip(y[:-1], y[1:])) if y0 != y1] + [len(y)]
		segments = [data[start:end] for start,end in zip(segment_idx[:-1], segment_idx[1:])]
		semi_markov_data = [np.sum(s,0) for s in segments ]
		# semi_markov_data = [[x/np.sum(x, dtype=np.float) for x in y] for y in semi_markov_data]
		# print 'sshape', np.shape(semi_markov_data)
		# embed()
		semi_markov_data = [x/np.sum(x, dtype=np.float) for x in semi_markov_data]
		segment_labels = [y[i] for i in segment_idx[:-1]]

		# Classify each semi-markov histogram to get the sm unary potential
		semi_markov_unary = self.semi_markov_clf.predict_proba(semi_markov_data)
		# semi_markov_unary = 1.-self.semi_markov_clf.predict_proba(semi_markov_unary_hists)

		# print ""
		# print 'y', y#, semi_markov_data
		# print 'segments', segment_idx
		# print 'semi', semi_markov_unary
		# print 't', self.semi_markov_transition
		# print 'x', markov_unary
		psi = np.array([
						np.sum(np.log([markov_unary[i,y_] for i,y_ in enumerate(y)])),
						np.sum(np.log([self.markov_transition[i,j] for i,j in zip(y[:-1], y[1:])])),
						np.sum(np.log([semi_markov_unary[i,y_] for i,y_ in enumerate(segment_labels)])),
						np.sum(np.log([self.semi_markov_transition[i,j] for i,j in zip(segment_labels[:-1], segment_labels[1:])])),
						])

		# print 'psi', psi
		return psi


	def inference(self, x, w, relaxed=None,
					min_seg_length=3, max_seg_length=20,
					n_interval=1, loss=0):
		'''
		min_seg_length :
		max_seg_length :
		n_interval : Eval every n_interval markovs
		normalize : normalize the histograms?

		if using inference augmented with loss...
		'''

		n_classes = self.n_states
		data_length = x.shape[0]
		markov_histograms = x

		# SVM weight notation: w=weight, m=markov, sm=semi-markov, pair=pairwise
		w_m_unary = w[0]
		w_m_pair = w[1]
		w_sm_unary = w[2]
		w_sm_pair = w[3]
		# w_m_unary = 0.
		# w_m_pair = 0.
		# w_sm_unary = w[0]
		# w_sm_pair = w[1]

		# Initialize score/class matricies
		score = np.zeros([data_length,n_classes], dtype=np.float)
		next_boundary = np.zeros([data_length,n_classes], dtype=np.int)
		current_class = np.zeros([data_length,n_classes], dtype=np.int)

		# Update starting segment score with prior
		score[0] += w_m_pair*np.log(self.markov_prior)

		# Get markov unaries
		markov_unary = w_m_unary * np.log(self.markov_clf.predict_proba(markov_histograms))

		# Compute log of transition matrix for pairwise term
		markov_pairwise = w_m_pair*np.log(self.markov_transition)
		semi_markov_pairwise = w_sm_pair*np.log(self.semi_markov_transition)

		# Init the starting position index
		idx_start = 0
		max_seg_length = np.minimum(data_length, max_seg_length)

		# Go through all of the data from start to end
		while idx_start < data_length-1:
			seg_length = np.minimum(data_length-idx_start, max_seg_length) # segment length

			if seg_length <= min_seg_length:
				min_seg_length -= n_interval

			# Get all candidate positions for the end of this gesture (start to end)
			idx_end_candidates = np.arange(idx_start+min_seg_length, idx_start+seg_length, n_interval)

			# Markov unaries for each potential segment
			potential_markov_unary_score = np.vstack([np.sum(markov_unary[idx_start:end+1],0) for end in idx_end_candidates])

			# semi-Markov unaries for each potential segment
			semi_markov_histograms = [np.sum(markov_histograms[idx_start:end+1],0) for end in idx_end_candidates]
			potential_semi_markov_unaries = w_sm_unary*np.log(self.semi_markov_clf.predict_proba(semi_markov_histograms))
			# potential_semi_markov_unaries = w_sm_unary*np.log(1.-self.semi_markov_clf.predict_proba(semi_markov_histograms_chi2))

			if loss is not 0:
				loss_tmp = np.array([np.sum(loss[idx_start:end+1],0) for end in idx_end_candidates])
			else:
				loss_tmp = 0

			# Get score for each state
			for state in range(n_classes):
				new_score = np.array(
						score[idx_start,state]
						+ potential_markov_unary_score
						+ markov_pairwise[state]
						+ potential_semi_markov_unaries
						+ semi_markov_pairwise[state]
						+ loss_tmp
						)

				# Get the best transition for each state
				score_tmp = np.max(new_score, 0)
				idx_tmp = np.argmax(new_score,0)
				next_idx = idx_end_candidates[idx_tmp[np.argmax(score_tmp,0)]]

				# print "n:", new_score, np.max(score_tmp,0), np.argmax(score_tmp,0), next_idx
				score[idx_start,state] = np.max(score_tmp,0)
				current_class[idx_start,state] = np.argmax(score_tmp,0)
				next_boundary[idx_start,state] = next_idx + 1


			if idx_start + n_interval < data_length:
				idx_start += n_interval
			else:
				break

		# if loss is 0:
		# 	embed()

		## Get final segment labelings

		# Init segments [segments] to an arbitraily large size.
		# This denotes starting and ending positions for each segment/class
		# Index 0 is the frame and 1 is the class label
		segments = np.zeros([np.minimum(100,data_length),2], np.int)

		# Get class of the first segment
		segments[0,1] = current_class[0,np.argmax(score[0])]

		# Find the rest of the segments
		i=1
		while segments[i-1,0] < data_length and 0 < i < data_length and i < len(segments):
			# if next_boundary[segments[i-1,0], segments[i-1,1]]==0:
				# break
			try:
				segments[i,0] = next_boundary[segments[i-1,0], segments[i-1,1]]
				if next_boundary[segments[i-1,0], segments[i-1,1]]+1 >= len(segments):
					break
				segments[i,1] = current_class[segments[i,0],segments[i,1]]
			except:
				pass
			i += 1
		# print 'seg', segments

		# Ensure last seqment is at the end of the data
		n_segs = np.sum(segments[:,0]!=0) # the number of valid segments; add 1 for first frame(=0)
		segments[n_segs,0] = data_length

		# Output a dense labeling of each markov and the corresponding label
		path = np.zeros(data_length, np.int)
		for i in xrange(n_segs):
			path[segments[i,0]:segments[i+1,0]] = segments[i,1]

		return path



	def loss(self, y, y_hat):
		'''
		Percent of correct entries in the labeling
		'''
		if type(y) not in [list, np.ndarray]:
			count = 1.
		else:
			count = float(len(y))
		return np.sum((y!=y_hat))/count

	def loss_augmented_inference(self, x, y, w, relaxed=None):
		'''
		Used to find the most violated constraint
		1) Run inference
		2) Calculate loss
		3) Find the best path given psi and the loss
		'''

		# Calculate loss
		data_length = x.shape[0]
		delta_score = np.ones([data_length,self.n_states], np.float)

		for i,y_ in enumerate(y):
			delta_score[i,y_] = 0

		path = self.inference(x, w, loss=delta_score)

		return path





# X_train = np.array([[[.85,.15],[.85,.15],[.85,.15],[.15,.85]],
# 					[[.85,.15],[.85,.15],[.15,.85],[.15,.85]],
# 					[[.85,.15],[.15,.85],[.15,.85],[.15,.85]],
# 					[[.15,.85],[.15,.85],[.15,.85],[.15,.85]],
# 					[[.15,.85],[.15,.85],[.15,.85],[.85,.15]],
# 					[[.15,.85],[.15,.85],[.85,.15],[.85,.15]],
# 					[[.15,.85],[.85,.15],[.85,.15],[.85,.15]]], dtype=np.float)
# y_train = np.array([[0,0,0,1],
# 					[0,0,1,1],
# 					[0,1,1,1],
# 					[1,1,1,1],
# 					[1,1,1,0],
# 					[1,1,0,0],
# 					[1,0,0,0]
# 							], dtype=np.int)

# markov_prior = np.array([.5,.5])
# markov_transition = np.asmatrix(markov_prior).T * np.asmatrix(markov_prior)

# msm_crf = MsM_CRF(n_states=2, n_features=4, markov_prior=markov_prior, markov_transition=markov_transition)
# svm = SubgradientSSVM(msm_crf, verbose=1, C=100, n_jobs=1)

# svm.fit(X_train, y_train)
# # print svm.score(X_train, y_train)
# print svm.predict(X_train)


# X_train = np.array([[[.85,.15],[.85,.15],[.15,.85],[.15,.85]],
# 					[[.85,.15],[.15,.85],[.85,.15],[.15,.85]],
# 					[[.15,.85],[.85,.15],[.15,.85],[.85,.15]],
# 					[[.15,.85],[.15,.85],[.85,.15],[.85,.15]],
# 					[[.85,.15],[.15,.85],[.15,.85],[.15,.85]],
# 					[[.15,.85],[.15,.85],[.15,.85],[.85,.15]]], dtype=np.float)
# y_train = np.array([[1,1,0,0],
# 					[1,0,1,0],
# 					[0,1,0,1],
# 					[0,0,1,1],
# 					[1,0,0,0],
# 					[0,0,0,1],
# 							], dtype=np.int)

# svm = NSlackSSVM(msm_crf, verbose=1, check_constraints=True, C=100, n_jobs=1)

# X_train = np.array([[1,1,0,0],
# 					[1,0,1,0],
# 					[0,1,0,1],
# 					[0,0,1,1],
# 					[1,0,0,0],
# 					[0,0,0,1]], dtype=np.float)*5
# y_train = np.array([1,1,-1,-1,1,-1])

# y_train = np.array([[1,1,-1,-1],
# 					[1,-1,1,-1],
# 					[-1,1,-1,1],
# 					[-1,-1,1,1],
# 					[1,-1,-1,-1],
# 					[-1,-1,-1,1],
# 							], dtype=np.int)

# class BinarySVM(CRF):

	# markov_prior = None
	# markov_transition = None
	# semi_markov_prior = None
	# semi_markov_transition = None

	# def __init__(self, **kwargs):
	# 	super(MsM_CRF, self).__init__(**kwargs)
	# 	self.size_psi = self.n_features

	# def psi(self, x, y):
	# 	return y*x

	# def inference(self, x, w, relaxed=None):
	# 	return np.sign(np.dot(w,x))

	# def loss(self, y, y_hat):
	# 	if type(y) not in [list, np.ndarray]:
	# 		count = 1.
	# 	else:
	# 		count = float(len(y))
	# 	return np.sum((y!=y_hat))/count

	# def loss_augmented_inference(self, x, y, w, relaxed=None):
	# 	'''
	# 	Used to find the most violated constraint
	# 	'''
	# 	y_infer = self.inference(x, w)
	# 	return self.psi(x, y_infer) + self.loss(y, y_infer)


	# def inference(self, x, w, relaxed=None,
	# 				min_seg_length=5, max_seg_length=400,
	# 				n_interval=1, loss = 0):
	# 	'''
	# 	min_seg_length :
	# 	max_seg_length :
	# 	n_interval : Eval every n_interval markovs
	# 	normalize : normalize the histograms?

	# 	if using inference augmented with loss...
	# 	'''

	# 	n_classes = 2
	# 	data_length = x.shape[0]
	# 	markov_histograms = x

	# 	# SVM weight notation: w=weight, m=markov, sm=semi-markov, pair=pairwise
	# 	w_m_unary = w[0]
	# 	w_m_pair = w[1]
	# 	w_sm_unary = w[2]
	# 	w_sm_pair = w[3]

	# 	# Initialize score/class matricies
	# 	score = np.zeros([data_length,n_classes], dtype=np.float)
	# 	next_boundary = np.zeros([data_length,n_classes], dtype=np.int)
	# 	next_class = np.zeros([data_length,n_classes], dtype=np.int)

	# 	# Get per-markov unaries
	# 	markov_unary_hists = self.markov_chi2_sampler.transform(markov_histograms)
	# 	markov_unary = self.markov_clf.predict_proba(markov_unary_hists)

	# 	# generate markov unary score and include the loss function
	# 	markov_unary_score = w_m_unary*np.log(markov_unary) + loss

	# 	# Init the starting position index
	# 	idx_start = data_length - min_seg_length - 1

	# 	# Go through all of the data starting at the end.
	# 	while idx_start >= 0:
	# 		seg_length = np.minimum(data_length-idx_start, max_seg_length) # segment length
	# 		# markov_unary_score_tmp = np.zeros([seg_length,n_classes], dtype=np.float)
	# 		# markov_unary_score_tmp = np.zeros([data_length,n_classes], dtype=np.float)

	# 		# Compute a score for each index in the potential segment
	# 		# for scoreidx in xrange(idx_start, idx_start+seg_length):
	# 			# markov_unary_score_tmp[scoreidx-idx_start] = np.sum(markov_unary[idx_start:scoreidx+1],0)

	# 		# Iterate through all of the potential segments (start from the end)
	# 		idx_end_candidates = range(idx_start+seg_length-1, idx_start+min_seg_length-1, -n_interval)
	# 		markov_unary_score_tmp = np.zeros([len(idx_end_candidates),n_classes], dtype=np.float)
	# 		semi_markov_unaries = []
	# 		for i,idx_end in enumerate(idx_end_candidates):
	# 			markov_unary_score_tmp[i] = np.sum(markov_unary[:idx_end],0)

	# 			semi_markov_histograms = np.sum(markov_histograms[idx_start:idx_end], 0)

	# 			if self.normalize:
	# 				semi_markov_histograms /= float(np.sum(semi_markov_histograms))

	# 			semi_markov_unary_hists = self.semi_markov_chi2_sampler.transform(semi_markov_histograms)
	# 			semi_markov_unaries += [self.semi_markov_clf.predict_proba(semi_markov_unary_hists)]
	# 		semi_markov_unaries = np.log(np.vstack(semi_markov_unaries))

	# 		for state in range(n_classes):

	# 			# This score represents ... for each possible end index
	# 			new_score = np.array(
	# 					#score[idx_end_candidates]
	# 					+ markov_unary_score_tmp
	# 					+ w_m_pair*np.log(self.markov_transition[state])
	# 					+ w_sm_unary*semi_markov_unaries
	# 					+ w_sm_pair*np.log(self.semi_markov_transition[state])
	# 					)

	# 			score_tmp = np.max(new_score, 0)
	# 			idx_tmp = np.argmax(new_score,0)
	# 			score[idx_start,state] = np.max(score_tmp,0)

	# 			# Get the best transition for each state
	# 			next_class[idx_start,state] = np.argmax(score_tmp,0)
	# 			next_boundary[idx_start,state] = idx_end_candidates[idx_tmp[next_class[idx_start,state]]]

	# 		if idx_start - n_interval >= 0:
	# 			idx_start -= n_interval
	# 		else:
	# 			break


	# 	## Get final segment labelings

	# 	# Init segments [segments] to an arbitraily large size.
	# 	# This denotes starting and ending positions for each segment/class
	# 	# segments[:,0] is the frame
	# 	# segments[:,1] is the class
	# 	segments = np.zeros([10,2], np.int)
	# 	segments[0,0] = idx_start

	# 	# Update starting segment score with prior
	# 	# score[:,idx_start] += w_m_pair*self.markov_prior # add prior
	# 	score[idx_start] += w_m_pair*self.markov_prior # add prior
	# 	# Get class of the first segment
	# 	segments[0,1] = np.argmax(score[idx_start])

	# 	# Find the rest of the segments
	# 	i=1
	# 	while segments[i-1,0] < data_length - min_seg_length and i < data_length - min_seg_length:
	# 		try:
	# 			segments[i,0] = next_boundary[segments[i-1,0], segments[i-1,1]]
	# 		except:
	# 			embed()
	# 		segments[i,1] = next_class[segments[i-1,0],segments[i-1,1]]
	# 		i += 1

	# 	# Ensure first segment and last seqment are markov 1 and the last markov
	# 	segments[0,0] = 0
	# 	n_segs = np.sum(segments[:,0]!=0) # the number of valid segments
	# 	segments[n_segs-1,0] = data_length

	# 	print 'segments', segments
	# 	# Output a dense labeling of each markov and the corresponding label
	# 	path = np.zeros(data_length, np.int)
	# 	for i in xrange(n_segs-1):
	# 		# print segments[i,1], segments[i,0],segments[i+1,0]
	# 		path[segments[i,0]:segments[i+1,0]] = segments[i,1]

	# 	print 'score', score
	# 	return path
