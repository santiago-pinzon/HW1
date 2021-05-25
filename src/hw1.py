from scipy.stats import multivariate_normal
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

# Vary lambda and create ROC Curve
for lambda1 in range(2000):
	tau = lambda1/100
	print("loop " + str(lambda1) + "/1000")
	# expected risk classification
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for p in scoresL0:
		if p[1] / p[0] > tau:
			fp = fp + 1	#classified1.append(tup)
		else:
			tn = tn + 1	#classified0.append(tup)

	for p in scoresL1:
		if p[1] / p[0] > tau:
			tp = tp + 1	#classified1.append(tup)
		else:
			fn = fn + 1	#classified0.append(tup)

	ROCVals.append([fp/(fp + tn), tp/(tp + fn)])


# plot our things
print(ROCVals)

plt.plot(*zip(*ROCVals))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(*zip(*l0Dataset), 'bo')
ax1.plot(*zip(*l1Dataset), 'ro')

plt.show()

# ROC curve

# theoretical optimal thresh
