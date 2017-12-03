import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

gh = 28 + 4 * np.random.randn(greyhounds)
lb = 24 + 4 * np.random.randn(labs)

plt.hist([gh, lb], stacked=True, color=['r', 'b'])

plt.show()
