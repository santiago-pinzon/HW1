import scipy
from scipy.stats import multivariate_normal
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

def ROCCurve(discScores, labels):
	sortedScores = np.sort(discScores)[::-1]
	thresholdList = [min(sortedScores) - eps]
	for ind in range(len(sortedScores) - 1):
		thresholdList.append((sortedScores[ind] + sortedScores[ind+1])/2)
	thresholdList.append(max(sortedScores) + eps)
	num1 = len([test for test in labels if test == 1])
	num0 = len([test for test in labels if test == 0])
	ptn = []
	pfp = []
	ptp = []
	perror = []
	for tau in thresholdList:
		decisions = [disc >= tau for disc in discScores]
		tn = 0
		fp = 0
		tp = 0
		error = 0
		for i, val in enumerate(decisions):
			if decisions[i] == 0 and labels[i] == 0:
				tn = tn + 1
			elif decisions[i] == 1 and labels[i] == 0:
				fp = fp + 1
				error = error + 1
			elif decisions[i] == 1 and labels[i] == 1:
				tp = tp + 1
			else:
				error = error + 1
		ptn.append(tn/num0)
		pfp.append(fp/num0)
		ptp.append(tp/num1)
		perror.append(error/len(labels))
	return ptn, pfp, ptp, perror

# generate set of 1000k
# constants
l1 = 0.35
l0 = 0.65
l0sub = 0.5
DATASIZE = 10000
# pdf constants
m01 = [3, 0]
c01 = [[2, 0], [0, 1]]
m02 = [0, 3]
c02 = [[1, 0], [0, 2]]
m = [2, 2]
c = [[1, 0], [0, 1]]
classes = 2
eps = 2**-52

# rng
rng = default_rng()
pdf1 = multivariate_normal(mean=m, cov=c)  # 1 label
pdf01 = multivariate_normal(mean=m01, cov=c01)  # 0 label < 50
pdf00 = multivariate_normal(mean=m02, cov=c02)  # 0 label > 50

# define datasets for labels
l0Dataset = []
l1Dataset = []

# generate dataset
for i in range(DATASIZE):
	test = rng.uniform()
	if test < l0:
		subtest = rng.uniform()
		if subtest < l0sub:
			l0Dataset.append(rng.multivariate_normal(m01, c01))
		else:
			l0Dataset.append(rng.multivariate_normal(m02, c02))
	else:
		l1Dataset.append(rng.multivariate_normal(m, c))

fulldataset = l0Dataset + l1Dataset

# Calculate Scores
scoresL0 = []
scoresL1 = []
for tup in l0Dataset:
	pTupGiven1 = pdf1.pdf(tup) * 0.35
	pTupGiven0 = pdf01.pdf(tup) * 0.325 + pdf00.pdf(tup) * 0.325
	scoresL0.append([pTupGiven0, pTupGiven1])

for tup in l1Dataset:
	pTupGiven1 = pdf1.pdf(tup) * 0.35
	pTupGiven0 = pdf01.pdf(tup) * 0.325 + pdf00.pdf(tup) * 0.325
	scoresL1.append([pTupGiven0, pTupGiven1])



# calculate empirical and theoretical optimal thresholds
# optIndex = np.argmax(OptimalROC)
# optThresh = thresholdValues[optIndex]
# optVal = ROCVals[optIndex]
#
# thIndex = np.argmin(pError)
# thThresh = thresholdValues[thIndex]
# thVal = ROCVals[thIndex]

# LDA Analysis
# need to take mean of each value
mu1hat = np.mean(np.array(l0Dataset), axis=0, keepdims=True)
mu2hat = np.mean(np.array(l1Dataset), axis=0, keepdims=True)

s1hat = np.cov(np.transpose(np.array(l0Dataset)), rowvar=True)
s2hat = np.cov(np.transpose(np.array(l1Dataset)), rowvar=True)

Sb = (mu1hat - mu2hat) * np.transpose(mu1hat - mu2hat)
Sw = s1hat + s2hat

sbsw = np.matmul(np.linalg.inv(np.array(Sw)), np.array(Sb))
D, V = scipy.linalg.eig(sbsw)
I = np.argsort(np.array(D))[::-1]
w = V[I[0]]

y1 = np.matmul(l0Dataset, np.transpose(np.array(w)))
y2 = np.matmul(l1Dataset, np.transpose(np.array(w)))

if np.mean(y2) <= np.mean(y1):
	w = -1 * w
	y1 = -1 * y1
	y2 = -1 * y2

zero1 = np.zeros(len(y1))
zero2 = np.ones(len(y2))

# generate labels and l2 discriminant
discriminantsLDA = np.append(y1, y2)
labels = np.append(zero1, zero2)

(ptn, pfp, ptp, perror) = ROCCurve(discriminantsLDA, labels)

f1, ((lda, rocLDA), (thresholds, last)) = plt.subplots(2, 2)
lda.plot(y1, zero1, 'b+')
lda.plot(y2, zero2, 'r+')
rocLDA.plot(pfp, ptp, 'og')


# plt.plot(*zip(*ROCVals))
# plt.plot(optVal[0],optVal[1], 'go')
# plt.plot(thVal[0], thVal[1], 'ro')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(*zip(*l0Dataset), 'bo')
ax1.plot(*zip(*l1Dataset), 'ro')

plt.show()


# ROC curve

# theoretical optimal thresh
