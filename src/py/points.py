import numpy as np

def horizontal_strips(h):

    nY = int(np.ceil(2/h))
    Y = np.linspace(-1, 1, nY)

    Xs, Ys = [], []

    for y in Y:
        xmax = np.cos(np.arcsin(y))
        nX = int(np.ceil(2*xmax/h))
        X = np.linspace(-xmax, xmax, nX)
        Xs.append(X)
        Ys.append(y*np.ones(nX))

    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)

    return X, Y

# fig = plt.figure()
# fig.add_subplot(121).scatter(Xs, Ys)
# fig.show()

def fibonacci_spiral(nPoints):

    phi = (1 + np.sqrt(5))/2
    nPoints = 100
    X = np.zeros(nPoints)
    Y = np.zeros(nPoints)

    a, da = 0, 2*np.pi*(phi - 1)/phi
    r, dr = 0, 1/(nPoints + 1)

    for i in range(nPoints):
        X[i], Y[i] = r*np.cos(a), r*np.sin(a)
        a = np.mod(a + da, 2*np.pi)
        r += dr

    return X, Y

# fig.add_subplot(122).scatter(X, Y)
# fig.show()


def uniform_random(nPoints):
    nPoints = 50
    R = np.sqrt(np.random.random(nPoints))
    Theta = np.random.uniform(0, 2*np.pi, nPoints)
    X = R*np.cos(Theta)
    Y = R*np.sin(Theta)
    return X, Y
