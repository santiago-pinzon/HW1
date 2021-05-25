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
pfd1 = multivariate_normal(mean=m, cov=c) #1 label
pdf01 = multivariate_normal(mean=m01, cov=c01) #0 label < 50
pdf00 = multivariate_normal(mean=m02, cov=c02) #0 label > 50

# define datasets for labels
l0Dataset = []
l1Dataset = []

#generate dataset
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

#ERM classification
test = pdf01.pdf([3,3])
print(test)


print(l0Dataset)
plt.plot(*zip(*l0Dataset), 'bo')
plt.plot(*zip(*l1Dataset), 'ro')
plt.show()

plt.plot(*zip(*fulldataset), 'go')
plt.show()
# class-conditional Gaussian pdfs

# ROC curve

# theoretical optimal thresh
print("Test")
