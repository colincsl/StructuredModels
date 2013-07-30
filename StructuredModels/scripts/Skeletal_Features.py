
from scipy.spatial.distance import cdist, pdist

from pyKinectTools.dataset_readers.MSR_DailyActivities import MSRPlayer
from pyKinectTools.utils.SkeletonUtils import msr_to_kinect_skel, get_skel_angles

cam = MSRPlayer(base_dir='/Users/colin/Data/MSR_DailyActivities/Data/', actions=[1,2,5],subjects=[1],
				get_depth=True, get_color=True, get_skeleton=True)
n_interval = 15
n_top_joints = 3

skels = cam.skel_stack
skels = [msr_to_kinect_skel(x) for x in skels]

# Compute joint angles
skel_angles = np.array([get_skel_angles(x) for x in skels])
n_joint_angles = len(skel_angles[0])


''' Joint Angles Similarities [Eshed Ohn-Bar and Mohan M. Trivedi HAUD13] '''

# Find Affinity matrix (size: m*(m-1)/2 where m=#angles)

# Cosine distance
# distance_cos = np.dot(xi, xj) / (np.linalg.norm(xi)*np.linalg.norm(xj))
distance_cos = pdist(skel_angles, metric='cosine')
# distance_cos = cdist(skel_angles, skel_angles, metric='cosine')

# Longest Common Subsequence
# Modified from http://www.algorithmist.com/index.php/Longest_Common_Subsequence
# Robust to time series shifts and gaps

def lcss_length(str1, str2):
	return lcss(str1, str2)

def lcss_decode(str1, str2):
	return lcss(str1, str2, decode=True)

def lcss(str1, str2, decode=False):
	m = len(str1)
	n = len(str2)
	matrix = np.zeros([m+1,n+1], np.int16)
	for i in range(1, m+1):
		for j in range(1, n+1):
			if str1[i-1] == str2[j-1]:
				matrix[i,j] = matrix[i-1,j-1] + 1
			else:
				matrix[i,j] = np.maximum(matrix[i-1,j], matrix[i,j-1])

	if not decode:
		return np.max(matrix)

	def decode(i, j):
		if i is 0 or j is 0:
			return []
		elif str1[i-1] == str2[j-1]:
			return decode(i-1,j-1) + [x[i-1]]
		elif matrix[i-1,j] > matrix[i,j-1]:
			return decode(i-1,j)
		else:
			return decode(i,j-1)

	return decode(m,n)


distance_lcss = 1 - lcss_length(xi,xj)/min(np.linalg.norm(xi, 'l1'), np.linalg.norm(xj), 'l1')


# Create affinity matrix
sigma2 = 0.7**2
affinity_matrix = np.exp(-D/sigma2) / np.sum(-D/sigma2, 1)



''' ----------------- SMIJ ------------------------- '''
# Compute angle variances over time
skel_angles_variance = [np.var(skel_angles[i:i+n_interval], 0) for i in range(0, len(skels), n_interval)]

# plot(skel_angles_variance)
# show()

# Rank joint angles by max variance
joints_ranked = [np.argsort(x)[::-1] for x in skel_angles_variance]
joints_top_idx = [x[:n_top_joints] for x in joints_ranked]
joints_top_var = [skel_angles_variance[i][x[:5]] for i,x in enumerate(joints_ranked)]
joint_histogram = np.histogram(np.hstack(joints_top_idx), n_joint_angles)

# Histogram of most informative joints: concatenate hist of rank=1, r=2,...
hmij = []
for i in range(n_top_joints):
	top_idx = [x[i] for x in joints_ranked]
	hist = np.histogram(np.hstack(top_idx), n_joint_angles)
	hmij += [hist]
hmij = np.hstack(np.hstack(hmij))



# Features:
	# set of the most informative joints in each time segment
	# the temporal evolution of the most informative joints


import skimage.feature as sf
sf.hog(image, orientations=9, visualise=False, normalise=False)








