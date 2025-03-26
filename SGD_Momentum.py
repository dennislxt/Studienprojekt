import torch as t
from matplotlib import pyplot  as plt
import numpy as np

x_plot = np.linspace(-6.5,6.5,200)
y_plot = x_plot ** 2 * np.sin(x_plot)

x = t.tensor([-5.0] , requires_grad=True) # Startwert

optimierer = t.optim.SGD([x], lr=0.01,momentum=0.95)

x_werte = [-5]
loss_werte = []

for i in range(100):
    optimierer.zero_grad()
    loss = x**2 * t.sin(x)
    loss.backward()
    optimierer.step()

    print(f"Epoch {i+1}: x = {x.item():.4f}, Loss = {loss.item():.4f}")
    x_werte.append(x.item())
    loss_werte.append(loss.item())

loss_werte.append(x_werte[-1]**2 * np.sin(x_werte[-1]))

plt.plot(x_plot,y_plot,color = 'black')
plt.scatter(x_werte,loss_werte, color = 'royalblue',linewidths=1.0)
plt.scatter(x_werte[-1],loss_werte[-1],color = 'red',linewidths=2.5)
plt.title("SGD mit Momentum")
plt.savefig("SGD_Momemtum.png")
plt.show()