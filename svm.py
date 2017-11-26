# implement of svm by zsz
# 2017-11-11 happy single day!
# naive implement
# :)

from numpy import *
from collections import Iterable
import matplotlib.pyplot as plt

# use this to fetch data, y is label
def fetchData(filename, begX, xSize, begY, ySize):
    raw = loadtxt(filename)
    x = raw[:, begX:begX + xSize]
    y = raw[:, begY:begY + ySize]
    tmp = [[1] if p[0] == 0 else [-1] for p in y]
    y = mat(tmp)
    return x, y

# define svm structure
class Svm:
    def __init__(self, name = 'svm'):
        self.name = name
        self.MAX_STEPS = 1000

    # use this function to fit sample data, C is a trade-off
    def fit(self, x_data, y_data, C=0.5, kernel=None):
        # identify the kernel, linear default
        if kernel == None:
            self.kernel_name = 'linear'
        elif not isinstance(kernel, Iterable):
            raise NameError('kernel should be array-like object')
        elif kernel[0] == 'linear':
            self.kernel_name = 'linear'
        elif kernel[0] == 'rbf':
            self.kernel_name = 'rbf'
            if len(kernel) < 2:
                raise NameError('kernel should be given sigma for rbf model')
            if float(kernel[1]) == 0:
                kernel[1] = 1
            self.sigma = float(kernel[1])
        elif kernel[0] == 'polynomial':
            self.kernel_name = 'polynomial'
            if len(kernel) < 2:
                raise NameError('kernel should be given m for polynomial model')
            self.m = int(kernel[1])
        else:
            raise NameError('no support for such kernel : {}'.format(str(kernel[0])))

        assert len(x_data) == len(y_data)
        self.N = len(x_data)
        self.x_data = mat(x_data.copy())
        self.y_data = mat(y_data.copy())
        self.C = C

        # make variables
        self.alphas = mat(zeros((self.N, 1)))
        self.b = 0
        self.predictY = mat(zeros((self.N, 1)))

        self.calculateK()
        self.train()

        # calculate b by SVs
        b1 = 0
        b2 = 0
        for i in range(self.N):
            alpha = self.alphas[i, 0]
            if alpha > 0 and self.y_data[i, 0] > 0:
                b1 = self.y_data[i, 0] - dot(multiply(self.alphas, self.y_data).T, self.K[:, i])[0, 0]
            if alpha > 0 and self.y_data[i, 0] < 0:
                b2 = self.y_data[i, 0] - dot(multiply(self.alphas, self.y_data).T, self.K[:, i])[0, 0]
        self.b = (b1 + b2)/2

    # SMO implement
    def tune(self, i, j):
        self.predictY[i, 0] = self.predict(self.x_data[i, :])
        self.predictY[j, 0] = self.predict(self.x_data[j, :])
        a1 = self.alphas[i, 0]
        a2 = self.alphas[j, 0]
        y1 = self.y_data[i, 0]
        y2 = self.y_data[j, 0]
        gamma = a1 + y1 * y2 * a2
        # watch out for the boundary
        L = 0
        U = self.C
        if y1 == y2:
            L = max(0, gamma - self.C)
            U = min(self.C, gamma)
        else:
            L = max(0, -gamma)
            U = min(self.C, self.C - gamma)
        b2 = a2 + y2 * (self.predictY[i, 0] - self.y_data[i, 0] - self.predictY[j, 0] + self.y_data[j, 0])/(self.K[i, i] + self.K[j, j] - 2 * self.K[i, j])
        if b2 < L:
            b2 = L
        if b2 > U:
            b2 = U
        b1 = a1 + y1 * y2 * (a2 - b2)
        # update here
        self.alphas[i, 0] = b1
        self.alphas[j, 0] = b2

    # train by data
    def train(self):
        i, done = self.findI(0)
        count = 0
        while not done and count <= self.MAX_STEPS:
            count = count + 1
            index = i
            # choose a random one is a trick to avoid the same update operation
            while index == i:
                index = int(random.uniform(0, self.N))
            self.tune(i, index)
            # here we use a trick to roll up, which can avoid the same update operation
            i, done = self.findI(i)

    # calculate the value of K by one column
    def calculateK_col(self, sample):
        sample = mat(sample)
        col = mat(zeros((self.N, 1)))
        if self.kernel_name == 'linear':
            col = dot(self.x_data, sample.T)
        elif self.kernel_name == 'rbf':
            for i in range(self.N):
                diff = self.x_data[i, :] - sample
                col[i, 0] = exp(-dot(diff, diff.T)[0,0]/(2*self.sigma**2))
        elif self.kernel_name == 'polynomial':
            for i in range(self.N):
                col[i, 0] = (dot(self.x_data[i, :], sample.T)[0,0] + 1)**self.m
        return col


    def calculateK(self):
        self.K = mat(zeros((self.N, self.N)))
        for i in range(self.N):
            self.K[:,i] = self.calculateK_col(self.x_data[i,:])

    def predict(self, x):
        return dot(self.calculateK_col(x).T,multiply(self.alphas, self.y_data))[0,0] + self.b


    def findI(self, last):
        # roll up from last place and return at the first one
        for i in range(last + 1, self.N):
            self.predictY[i, 0] = self.predict(self.x_data[i,:])
            if (self.alphas[i, 0] == 0 and self.y_data[i, 0] * self.predictY[i, 0] <= 1) or\
                (self.alphas[i, 0] == self.C and self.y_data[i, 0] * self.predictY[i, 0] >= 1) or\
                (self.alphas[i, 0] < self.C and self.alphas[i, 0] >0 and abs(self.y_data[i, 0] * self.predictY[i, 0] - 1)>1e-1):
                return i, False
        for i in range(last):
            self.predictY[i, 0] = self.predict(self.x_data[i,:])
            if (self.alphas[i, 0] == 0 and self.y_data[i, 0] * self.predictY[i, 0] <= 1) or\
                (self.alphas[i, 0] == self.C and self.y_data[i, 0] * self.predictY[i, 0] >= 1) or\
                (self.alphas[i, 0] < self.C and self.alphas[i, 0] >0 and abs(self.y_data[i, 0] * self.predictY[i, 0] - 1)>1e-1):
                return i, False
        return 0, True


def main():
    x_data, y_data = fetchData('sample.txt', 0, 2, 3, 1)
    svm = Svm()
    svm.fit(x_data, y_data, kernel=['rbf', 1])
    count = 0
    # give a quantity of the empirical risk
    for i in range(len(x_data)):
        if sign(svm.predict(x_data[i])) == y_data[i, 0]:
            count = count + 1
    print('Empirical risk is {}'.format(float(count)/len(x_data)))

    # here we only use the two first dimension to simplify
    x = linspace(-10,10,100)
    y = linspace(-10,10,100)
    X, Y = meshgrid(x, y)
    blue = x_data[:50, :]
    red = x_data[50:, :]

    # compute the decision boundary
    Edge = []
    for i in range(len(x)):
        for j in range(len(y)):
            if abs(svm.predict([x[i], y[j]])) < 1e-1:
                Edge.append([x[i], y[j]])
    Edge = mat(Edge)

    # draw it up
    plt.scatter(list(blue[:,0]), list(blue[:,1]),c='b')
    plt.scatter(list(red[:,0]), list(red[:,1]), c='r')
    plt.scatter(list(Edge[:, 0]), list(Edge[:, 1]),c='g')
    plt.show()


if __name__ == '__main__':
    main()