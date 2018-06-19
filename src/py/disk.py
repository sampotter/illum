import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

cat = np.concatenate

nR, nTheta = 10, 20

Rs = np.cos(np.linspace(0, np.pi/2, 2*nR))
# Rs = 1 - np.linspace(0, 1, 2*nR)**2.5;

Rs_even = Rs[0::2]
Rs_odd = Rs[1:-1:2]
# Rs_odd /= Rs_odd[0]

Thetas = np.linspace(0, 2*np.pi, nTheta, endpoint=False)
Thetas_even = Thetas[0::2]
Thetas_odd = Thetas[1::2]

[R_even, Theta_even] = np.meshgrid(Rs_even, Thetas_even)
R_even = R_even.flatten()
Theta_even = Theta_even.flatten()

[R_odd, Theta_odd] = np.meshgrid(Rs_odd, Thetas_odd)
R_odd = R_odd.flatten()
Theta_odd = Theta_odd.flatten()

R = cat([[0], R_even, R_odd])
Theta = cat([[0], Theta_even, Theta_odd])

X = R*np.cos(Theta)
Y = R*np.sin(Theta)
Points = np.column_stack([X, Y])

tri = scipy.spatial.Delaunay(Points)

fig = plt.figure()
fig.add_subplot(121).triplot(X, Y, tri.simplices.copy())
fig.add_subplot(122).scatter(X, Y)
plt.show()

