import numpy as np
import matplotlib.pyplot as plt

xk = np.array([0.9, 0.9])
pk = np.array([0.8, 0.8])
P = 1.0

X, Y = np.meshgrid(np.linspace(0,1.5,100), np.linspace(0,1.5,100))
Z = 0.5*((X+Y)**2) - xk[0]*X - xk[1]*Y - pk[0]*X - pk[1]*Y
rhs = P - 0.5*(np.sum(xk**2) + np.sum(pk**2))

plt.contourf(X, Y, Z <= rhs, levels=1)
plt.colorbar()
plt.scatter(xk[0], xk[1], color='red', label='xk')
plt.scatter(pk[0], pk[1], color='blue', label='pk')
plt.legend()
plt.show()
