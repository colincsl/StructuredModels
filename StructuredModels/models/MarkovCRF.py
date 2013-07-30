import numpy as np
from pystruct.models import StructuredModel
from pystruct.models import CRF
from IPython import embed


class MarkovCRF(CRF):
	prior = None
	transition = None

	def __init__(self, prior, transition, clf, n_states=2, n_features=None, inference_method='qpbo',
				 class_weight=None):
		# super(MarkovCRF, self).__init__(**kwargs)
		self.size_psi = 2
		self.prior = prior
		self.transition = np.maximum(transition, .0001)
		self.clf = clf

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
		Todo: Divide psi by N?

		NOTE: a psi of 0 is set to .01 for log numerical reasons
		psi() = [
				Markov:
				 sum_1^T(psi_frame_unary) =
				 sum_1^T(psi_frame_pairwise) = log(transition_matrix_ij)
				Semi-markov
				 sum_1^M(psi_gesture_unary)
				 sum_1^M(psi_gesture_pairwise) = log(transition_matrix_ij))
				]
		'''

		data = x
		unary = self.clf.predict_proba(data)

		psi = np.array([
						np.sum(np.log([unary[i,y_] for i,y_ in enumerate(y)])),
						np.sum(np.log([self.transition[i,j] for i,j in zip(y[:-1], y[1:])]))
						])

		# print 'psi',psi
		return psi


	# def infer_partial(self, x, w, relaxed=None):
	# 	'''
	# 	Todo: Use log probabilities
	# 	P(s_t|s_t-1, o) = P(o|s_t) * P(s_t-1 | s_t) * P(s_t)
	# 	'''
	# 	n_states = x.shape[1]
	# 	prob = np.zeros([x.shape[0],n_states], dtype=np.float)
	# 	best_states = np.zeros(n_states, dtype=np.int)
	# 	paths = {x:[x] for x in range(n_states)}

	# 	# Get initial probability
	# 	prob[0] = self.prior * x[0]
	# 	# Go through each observation
	# 	for i,val in enumerate(x[1:]):
	# 		for s in range(n_states):
	# 			prob[i+1, s] = np.max(prob[i] * val[s] * np.asarray(self.transition)[:,s])
	# 			best_states[s] = np.argmax(prob[i] * val[s] * np.asarray(self.transition)[:,s])

	# 		# Traceback
	# 		new_paths = {}
	# 		for p,best in zip(paths,best_states):
	# 			new_paths[p] = paths[best] + [p]
	# 		paths = new_paths

	# 	# print prob
	# 	# print "paths", paths
	# 	# print ""
	# 	return prob, paths

	def infer_partial_log(self, x, w, relaxed=None, loss=None):
		'''
		P(s_t|s_t-1, o) = P(o|s_t) * P(s_t-1 | s_t) * P(s_t)
		'''

		data = x
		n_states = self.n_states
		n_data = data.shape[0]

		prob = np.zeros([n_data,self.n_states], dtype=np.float)
		paths = {i:[i] for i in range(n_states)}
		best_states = np.zeros(n_states, dtype=np.int)

		w_unary = w[0]
		w_pairwise = w[1]

		# Compute markov probabilities (from histograms)
		unary_prob = self.clf.predict_proba(data)
		unary_score = w_unary * np.log(unary_prob)
		pairwise_score = w_pairwise * np.log(np.asarray(self.transition))

		# Get initial probability
		prob[0] = np.log(self.prior) + unary_score[0]
		if loss is not None:
			prob[0] += loss[0]

		# Go through each observation
		# for i,_ in enumerate(data):
		for i in xrange(1,n_data):
			for state in range(n_states):
				score_tmp = prob[i-1] + unary_score[i][state] + pairwise_score[state]
				if loss is not None:
					score_tmp += loss[i]

				prob[i, state] = np.max(score_tmp)
				best_states[state] = np.argmax(score_tmp)

			# Traceback
			new_paths = {}
			for p,best in zip(paths,best_states):
				new_paths[p] = paths[best] + [p]
			paths = new_paths

		best_path = np.argmax(prob[-1])
		inferred_path = np.array(paths[best_path])

		return inferred_path

	def inference(self, x, w, relaxed=None, loss=None):
		''' Computed with Viterbi '''
		inferred_path = self.infer_partial_log(x, w, relaxed=None, loss=loss)

		# print "Inferred:", inferred_path, x, w
		return inferred_path

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
		inferred_path = self.infer_partial_log(x, w, relaxed=None)

		# print "Inferred w/ loss:", inferred_path
		return inferred_path





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

# frame_prior = np.array([.5,.5])
# frame_transition_matrix = np.asmatrix(frame_prior).T * np.asmatrix(frame_prior)

# msm_crf = MsM_CRF(n_states=2, n_features=4, frame_prior=frame_prior, frame_transition=frame_transition_matrix)
# svm = SubgradientSSVM(msm_crf, verbose=1, C=100, n_jobs=1)

# svm.fit(X_train, y_train)
# # print svm.score(X_train, y_train)
# print svm.predict(X_train)


# # X_train = np.array([[[.85,.15],[.85,.15],[.15,.85],[.15,.85]],
# # 					[[.85,.15],[.15,.85],[.85,.15],[.15,.85]],
# # 					[[.15,.85],[.85,.15],[.15,.85],[.85,.15]],
# # 					[[.15,.85],[.15,.85],[.85,.15],[.85,.15]],
# # 					[[.85,.15],[.15,.85],[.15,.85],[.15,.85]],
# # 					[[.15,.85],[.15,.85],[.15,.85],[.85,.15]]], dtype=np.float)
# # y_train = np.array([[1,1,0,0],
# # 					[1,0,1,0],
# # 					[0,1,0,1],
# # 					[0,0,1,1],
# # 					[1,0,0,0],
# # 					[0,0,0,1],
# # 							], dtype=np.int)

# # svm = NSlackSSVM(msm_crf, verbose=1, check_constraints=True, C=100, n_jobs=1)

# # X_train = np.array([[1,1,0,0],
# # 					[1,0,1,0],
# # 					[0,1,0,1],
# # 					[0,0,1,1],
# # 					[1,0,0,0],
# # 					[0,0,0,1]], dtype=np.float)*5
# # y_train = np.array([1,1,-1,-1,1,-1])

# # y_train = np.array([[1,1,-1,-1],
# # 					[1,-1,1,-1],
# # 					[-1,1,-1,1],
# # 					[-1,-1,1,1],
# # 					[1,-1,-1,-1],
# # 					[-1,-1,-1,1],
# # 							], dtype=np.int)

# # class BinarySVM(CRF):

# 	# frame_prior = None
# 	# frame_transition = None
# 	# gesture_prior = None
# 	# gesture_transition = None

# 	# def __init__(self, **kwargs):
# 	# 	super(MsM_CRF, self).__init__(**kwargs)
# 	# 	self.size_psi = self.n_features

# 	# def psi(self, x, y):
# 	# 	return y*x

# 	# def inference(self, x, w, relaxed=None):
# 	# 	return np.sign(np.dot(w,x))

# 	# def loss(self, y, y_hat):
# 	# 	if type(y) not in [list, np.ndarray]:
# 	# 		count = 1.
# 	# 	else:
# 	# 		count = float(len(y))
# 	# 	return np.sum((y!=y_hat))/count

# 	# def loss_augmented_inference(self, x, y, w, relaxed=None):
# 	# 	'''
# 	# 	Used to find the most violated constraint
# 	# 	'''
# 	# 	y_infer = self.inference(x, w)
# 	# 	return self.psi(x, y_infer) + self.loss(y, y_infer)

