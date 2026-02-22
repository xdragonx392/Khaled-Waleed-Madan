import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
x = np.arange(0, 10)
y = x ** 2

plt.plot(x, y)
plt.title("Line Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

plt.plot(x, x, label="Linear")
plt.plot(x, x**2, label="Quadratic")
plt.legend()
plt.title("Multiple Lines")
plt.show()

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.show()

categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

plt.bar(categories, values)
plt.title("Bar Chart")
plt.show()

data = np.random.randn(1000)

plt.hist(data, bins=30)
plt.title("Histogram")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10,4))

axes[0].plot(x, x)
axes[0].set_title("Linear")

axes[1].plot(x, x**2)
axes[1].set_title("Quadratic")

plt.show()
