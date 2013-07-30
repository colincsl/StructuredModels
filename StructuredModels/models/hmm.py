
'''
Implementation of a Hidden Markov Model based on Bishop's Pattern Recognition and Machine Learning book.

'''

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm

# Generate training data?
# Sine waves?
sig1 = np.sin(np.arange(180)*np.pi/180)
sig2 = np.cos((np.arange(180)+30)*np.pi/180)

data_raw = []
for i in range(10):
	data_raw += [np.arange(180)+np.random.random(1)*10]
	data_raw += [np.arange(-180,0)+np.random.random(1)*10 + 180]
data_raw = np.array(data)
data = data_raw.copy()

class GaussianHMM:
	''' Continuous (Gaussian) version of an HMM '''

	def __init__(self, n_states):

		# self.n_states = n_states
		n_states = 3

		# Prior: p(z)
		prior = np.random.random([n_states])
		prior /= np.sum(prior)

		# Transitions: p(z'|z)
		transition_matrix = np.random.random([n_states,n_states])
		transition_matrix /= np.sum(transition_matrix, 1)[:,None]

		# Emissions: p(o|z)
		emission_means = None
		emission_variances = None

	def fit(self, data):
		'''
		Baum-Welch learning

		Expectation: Find posterior of latent variables p(Z|X,theta)
			gamma(z') = p(z'|X,theta) [marginal posterior]
			eta(z,z') = p(z,z'|X,theta) [joint posterior]
		'''

		data_tmp = data.reshape([1,-1]).T

		# Initialize emission matrix by setting mean/variance using kmeans
		kmeans = KMeans(n_clusters=n_states, n_jobs=-1)
		kmeans.fit(data_tmp)
		data_predict = kmeans.predict(data_tmp)

		emission_means = kmeans.cluster_centers_.T[0]
		emission_variances = np.array([np.var(data_tmp[data_predict==x]) for x in range(n_states)])

		'''
		# Uncomment!
		for k in range(n_states):
			prior[k] = gamma[z[1,k]] / np.sum(gamma[1,:])
			transition_matrix[j,k] = np.sum(eta[z[:-1], z[1:]]) /
										np.sum([np.sum(eta[z[:-1], z[1:,l]]) for l in np.arange(self.n_states)])

			emission_means[k] = np.sum(gamma[z[:,k]]*x) / np.sum(gamma[z[:,k]])
			emission_variances[k] = np.sum(gamma[z[:,k]*np.asmatrix((x-emission_means[k]))*np.asmatrix((x-emission_means[k]).T)]) /
									np.sum(gamma[z[:,k]])
		'''

	# def forward(self, data):
	# 	'''
	# 	Alpha is the joint probability of observing all data and z
	# 	alpha(z[n]) = p(x1...xn, z[n])
	# 	'''
	# 	data = data_raw[0]
	# 	length = data.shape[0]
	# 	alpha = np.zeros([n_states, length], np.float)

	# 	# Initialize alpha
	# 	datum_norm = (data[0] - emission_means) / emission_variances
	# 	alpha[:,0] = prior * norm.pdf(datum_norm)

	# 	# Recursively update alpha
	# 	for i,d in enumerate(data[1:]):
	# 		datum_norm = (d - emission_means) / emission_variances
	# 		datum_pdf = norm.pdf(datum_norm)
	# 		for k in xrange(n_states):
	# 			# alpha[k] = np.sum( p(X|Z) * p(Z'|Z) * alpha[k])
	# 			alpha[:,i+1] = np.sum( alpha[:,i] * transition_matrix[:,k]) * datum_pdf[k]

	# 	return alpha[:,-1].sum()

	# def backward(self, data):
	# 	length = data.shape[0]
	# 	beta = np.zeros([n_states, length], np.float)

	# 	# Initialize beta
	# 	beta[:,-1] = 1.

	# 	# Recursively update beta
	# 	for i,d in enumerate(data[-1:1:-1]):
	# 		datum_norm = (d - emission_means) / emission_variances
	# 		datum_pdf = norm.pdf(datum_norm)
	# 		for k in xrange(n_states):
	# 			beta[:,length-i-2] = np.sum( transition_matrix[:,k] * datum_pdf[k] * beta[:,length-i-1])

	# 	datum_norm = (data[0] - emission_means) / emission_variances
	# 	datum_pdf = norm.pdf(datum_norm)

	# 	return (beta[:,0]*prior*datum_pdf).sum()

	def forward_scaled(self, data):
		'''
		Alpha is the joint probability of observing all data and z
		alpha(z[n]) = p(x1...xn, z[n])

		Use scale parameter to prevent underflow
		'''
		data = data_raw[1]
		length = data.shape[0]

		# Convert data to standard deviations and find probabilities
		data_norm = (data[:,None] - emission_means) / np.sqrt(emission_variances)
		emission_pdf = norm.pdf(data_norm).T
		emission_pdf /= emission_pdf.sum(0)

		alpha = np.zeros([n_states, length], np.float)
		scale = np.zeros(length, np.float)

		# Initialize alpha
		# datum_norm = (data[0] - emission_means) / np.sqrt(emission_variances)
		alpha[:,0] = prior * norm.pdf(data_norm[0])
		scale[0] = 1./alpha[:,0].sum()
		alpha[:,0] *= scale[0]

		# Recursively update alpha
		# for i,d in enumerate(data[1:]):
		for t in xrange(1,length):
			for k in xrange(n_states):
			# 	# alpha[k] = np.sum( p(X|Z) * p(Z'|Z) * alpha[k])
				alpha[k,t] = np.sum( alpha[:,t-1] * transition_matrix[:,k]) * emission_pdf[k,t]
			scale[t] = 1./alpha[:,t].sum()
			alpha[:,t] *= scale[t]

		# return alpha[:,-1].sum()

	def backward_scaled(self, data, scale):
		'''
		Beta is the conditional probability of all future data given z
		beta(z[n]) = p(xn...xN | z[n])
		'''

		length = data.shape[0]
		beta = np.zeros([n_states, length], np.float)

		data_norm = (data[:,None] - emission_means) / np.sqrt(emission_variances)
		emission_pdf = norm.pdf(data_norm).T
		emission_pdf /= emission_pdf.sum(0)

		# Initialize beta
		beta[:,-1] = scale[-1]

		# Recursively update beta
		for t in xrange(length-2, -1, -1):
			for k in xrange(n_states):
				beta[k,t] = np.sum( transition_matrix[k,:] * emission_pdf[:,t+1] * beta[:,t+1])
			beta[:,t] *= scale[t]

		# return (beta[:,0]*prior*datum_pdf).sum()

	figure(1)
	title('Alphas')
	for i in range(n_states):
		plot(alpha[i,:])
	figure(2)
	title('Betas')
	for i in range(n_states):
		plot(beta[i,:])


	# def compute_gamma(self):
	# 	# see #4 on http://www.cs.sjsu.edu/faculty/stamp/RUA/HMM.pdf

	# 	gamma = np.zeros([n_states,length])
		gamma = alpha * beta / scale

		eta = np.zeros([n_states, n_states,length])

		for t in range(data.shape[0]-1):
			denom=0
			for i in range(n_states):
				denom += np.sum(alpha[i,t]*transition_matrix[i,:]*emission_pdf[:,t+1]*beta[:,t+1])
			for i in range(n_states):
				for j in range(n_states):
					eta[i,j,t] = np.sum(alpha[i,t]*transition_matrix[i,j]*emission_pdf[j,t+1]*beta[j,t+1]) / denom


	def reestimate_parameters(self):
		'''
		p(X) = \prod_n c_n
		gamma(z) = alpha(z)*beta(z) / p(X)
		'''
		# data_prob = np.prod(scale)
		# gamma = alpha * beta / scale

		# for t in range(length):
		# 	for k in range(n_states):
		# 		for j in range(n_states):
		# 			eta[k,j,t] = alpha[k,t-1] * transition_matrix * emission_pdf[] * beta


		transition_matrix_ = transition_matrix.copy()
		emission_means_ = emission_means.copy()
		emission_variances_ = emission_variances.copy()

		prior[k] = gamma[:,0]
		# for k in range(n_states):
		# 	for j in range(n_states):

		# 		# transition_matrix_[j,k] = np.sum(eta[z[:-1], z[0:]]) /
		# 		# 							np.sum([np.sum(eta[z[:-1], z[0:,l]]) for l in np.arange(self.n_states)])
		# 		transition_matrix[j,k] = np.sum(eta[i,j]) / \
		# 									np.sum([np.sum(eta[i,j]) for l in np.arange(n_states)])


		# 		emission_means[k] = np.sum(gamma[z[:,k]]*data) / np.sum(gamma[z[:,k]])
		# 		emission_variances[k] = np.sum(gamma[z[:,k]*np.asmatrix((data-emission_means[k]))*np.asmatrix((data-emission_means[k]).T)]) / \
		# 								np.sum(gamma[z[:,k]])


	def reestimate_parameters(self):
		# see #5 on http://www.cs.sjsu.edu/faculty/stamp/RUA/HMM.pdf

		# Re-estimate prior
		prior = gamma[:,0]

		# Re-estimate transition_matrix
		for i in range(n_states):
			for j in range(n_states):
				numer = np.sum(eta[i,j])
				denom = np.sum(gamma[i])
				transition_matrix_[i,j] = numer/denom

		emission_means_ = np.sum(gamma*data, 1) / np.sum(gamma,1)
		emission_variances_ = np.sum(gamma * (data-emission_means[:,None])*((data-emission_means[:,None])), 1) / np.sum(gamma,1)

		# for k in range(n_states):
			# emission_variances_[k] = np.sum(gamma[k] * np.asmatrix((data-emission_means[k]))*np.asmatrix((data-emission_means[k]).T)) /
									# np.sum(gamma[k])

		# Re-estimate emission parameters

	def loglikelihood(self):
		'''
		The log likelihood can simply be computed using the scale variable
		'''
		logProb = -np.sum(np.log(scale))


	def transform(self, data):
		'''
		Forward-backward algorithm
		'''

		gamma = alpha*beta


	def decode(self, data):
		'''
		Viterbi
		'''

		# for t in xrange(length):
		pass



