import scipy
from scipy.stats import multivariate_normal
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def classify(data, labels, hatred):
    decisions = []
    for dat in data:
        probabilities = [pdf2.pdf(dat) * l2 + pdf03.pdf(dat) * l3 * 0.5 * hatred + pdf13.pdf(dat) * l3 * 0.5 * hatred,
                         pdf1.pdf(dat) * l1 + pdf03.pdf(dat) * l3 * 0.5 * hatred + pdf13.pdf(dat) * l3 * 0.5 * hatred,
                         pdf1.pdf(dat) * l1 + pdf2.pdf(dat) * l2]
        decisions.append(np.argmin(probabilities) + 1)

    l1correct = []
    l2correct = []
    l3correct = []
    l1wrong = []
    l2wrong = []
    l3wrong = []

    # cmat[x][y] = # where true = x + 1 and guess = y + 1 (I guess this would make more sense in matlab ¯\_(ツ)_/¯)
    cmat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    errors = 0

    for i in range(len(decisions)):
        x = labels[i]
        y = decisions[i]
        val = data[i]
        cmat[int(x) - 1][int(y) - 1] = cmat[int(x) - 1][int(y) - 1] + 1
        if x == 1:
            if y == 1:
                l1correct.append(val)
            else:
                l1wrong.append(val)
                errors = errors + 1
        if x == 2:
            if y == 2:
                l2correct.append(val)
            else:
                l2wrong.append(val)
                errors = errors + 1
        if x == 3:
            if y == 3:
                l3correct.append(val)
            else:
                l3wrong.append(val)
                errors = errors + 1

    cmat[int(0)] = np.array(cmat[int(0)]) / (len(l1wrong) + len(l1correct))
    cmat[int(1)] = np.array(cmat[int(1)]) / (len(l2wrong) + len(l2correct))
    cmat[int(2)] = np.array(cmat[int(2)]) / (len(l3wrong) + len(l3correct))
    perror = errors / len(labels)
    print('Confusion matrix')
    print('\t1\t2\t3')
    print('1\t' + str(cmat[0]))
    print('2\t' + str(cmat[1]))
    print('3\t' + str(cmat[2]))
    print('pError: ' + str(perror))

    lamda = [[0,1,hatred], [1,0, hatred], [1,1,0]]
    probs = [[l1,l2,l3], [l1,l2, l3], [l1, l2, l3]]
    minerror = (np.array(lamda) * np.array(probs))
    minerror = (minerror * np.array(cmat))
    print('Expected risk: ' + str(np.sum(minerror)))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    col = ax1.scatter3D(*zip(*data), c=labels, cmap='viridis')
    plt.colorbar(col, ax=ax1, location='left', shrink=0.5)
    ax1.set_title('True Dataset')

    ax2.scatter3D(*zip(*l1correct), c='green', marker='x', label='1 Correct')
    ax2.scatter3D(*zip(*l2correct), c='green', marker='.', label='2 Correct')
    ax2.scatter3D(*zip(*l3correct), c='green', marker='*', label='3 Correct')
    ax2.scatter3D(*zip(*l1wrong), c='red', marker='x', label='1 Wrong')
    ax2.scatter3D(*zip(*l2wrong), c='red', marker='.', label='2 Wrong')
    ax2.scatter3D(*zip(*l3wrong), c='red', marker='*', label='3 Wrong')
    ax2.legend()
    ax2.set_title("Classified Dataset")

    plt.show()


# generate set of 1000k
# constants
l3 = 0.4
l2 = 0.3
l1 = 0.3
l3sub = 0.5

DATASIZE = 10000
# pdf constants
m01 = [1, 1, 5]
c01 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
m02 = [5, 5, 5]
c02 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
m03 = [1, 5, 5]
c03 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
m13 = [5, 1, 5]
c13 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
classes = 3
eps = 2 ** -52

rng = default_rng()
pdf1 = multivariate_normal(mean=m01, cov=c01)  # 1 label
pdf2 = multivariate_normal(mean=m02, cov=c02)  # 2 label
pdf03 = multivariate_normal(mean=m03, cov=c03)  # 3 label < 0.5
pdf13 = multivariate_normal(mean=m13, cov=c13)  # 3 label > 0.5

# define datasets for labels
l1Dataset = []
l2Dataset = []
l3Dataset = []

# generate dataset
for i in range(DATASIZE):
    test = rng.uniform()
    if test < l1:
        l1Dataset.append(rng.multivariate_normal(m01, c01))
    elif test < l1 + l2:
        l2Dataset.append(rng.multivariate_normal(m02, c02))
    else:
        if test < l1 + l2 + l3 / 2:
            l3Dataset.append(rng.multivariate_normal(m03, c03))
        else:
            l3Dataset.append(rng.multivariate_normal(m13, c13))

data = l1Dataset + l2Dataset + l3Dataset
labels = np.append(np.ones(len(l1Dataset)), np.full(len(l2Dataset), 2))
labels = np.append(labels, np.full(len(l3Dataset), 3))

# classify our data
classify(data, labels, 1)
classify(data, labels, 10)
classify(data, labels, 100)