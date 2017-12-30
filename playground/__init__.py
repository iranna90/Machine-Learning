import numpy as np
from matplotlib import pyplot as plt,cm

ar = np.random.random([2,3])

print ar

plt.imshow(ar, cmap=cm.gray)
print "showing image"
print ar.shape
plt.show()