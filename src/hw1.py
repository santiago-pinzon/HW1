from scipy.stats import multivariate_normal
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

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
	pTupGiven0 = pdf01.pdf(tup) * 0.325 + pdf00.pdf(tup) * 0.35
	scoresL0.append([pTupGiven0, pTupGiven1])

for tup in l1Dataset:
	pTupGiven1 = pdf1.pdf(tup) * 0.35
	pTupGiven0 = pdf01.pdf(tup) * 0.325 + pdf00.pdf(tup) * 0.35
	scoresL1.append([pTupGiven0, pTupGiven1])

ROCVals = []
OptimalROC = []
thresholdValues = []
pError = []

# Vary lambda and create ROC Curve
for lambda1 in range(2000):
	tau = lambda1 / 100
	thresholdValues.append(tau)
	print("loop " + str(lambda1) + "/2000")
	# expected risk classification
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for p in scoresL0:
		if p[1] / p[0] > tau:
			fp = fp + 1  # classified1.append(tup)
		else:
			tn = tn + 1  # classified0.append(tup)

	for p in scoresL1:
		if p[1] / p[0] > tau:
			tp = tp + 1  # classified1.append(tup)
		else:
			fn = fn + 1  # classified0.append(tup)

	ROCVals.append([fp / (fp + tn), tp / (tp + fn)])
	OptimalROC.append(tp / (tp + fn) - fp / (fp + tn))
	pError.append((fp / len(l0Dataset)) * l0 + (fn / len(l1Dataset)) * l1)

# calculate empirical and theoretical optimal thresholds
optIndex = np.argmax(OptimalROC)
optThresh = thresholdValues[optIndex]
optVal = ROCVals[optIndex]

thIndex = np.argmin(pError)
thThresh = thresholdValues[thIndex]
thVal = ROCVals[thIndex]

# LDA Analysis
# need to take mean of each value
mu1hat = np.mean(np.array(l0Dataset), axis=0, keepdims=True)
mu2hat = np.mean(np.array(l1Dataset), axis=0, keepdims=True)

s1hat = np.cov(np.transpose(np.array(l0Dataset)), rowvar=True)
s2hat = np.cov(np.transpose(np.array(l1Dataset)), rowvar=True)

Sb = (mu1hat - mu2hat) * np.transpose(mu1hat - mu2hat)
Sw = s1hat + s2hat

sbsw = np.linalg.inv(np.array(Sw)) * Sb
V, D = np.linalg.eig(sbsw)
I = np.argsort(np.diag(np.array(D)))[::-1]
w = V[I]

y1 = np.transpose(w) * l0Dataset;
y2 = np.transpose(w) * l1Dataset;
zero1 = np.zeros(len(y1))
zero2 = np.ones(len(y2))

f1, lda = plt.subplots(1, 1)
lda.plot(y1, zero1, 'bo')
lda.plot(y2, zero2, 'ro')

# print(optIndex)
# print(optThresh)
# print(optVal)
#
# print(thIndex)
# print(thThresh)
# print(thVal)


# plt.plot(*zip(*ROCVals))
# plt.plot(optVal[0],optVal[1], 'go')
# plt.plot(thVal[0], thVal[1], 'ro')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(*zip(*l0Dataset), 'bo')
ax1.plot(*zip(*l1Dataset), 'ro')

plt.show()

# ROC curve

# theoretical optimal thresh
