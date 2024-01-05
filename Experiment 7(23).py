import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
X = np.arange(xmin, xmax, 0.1)
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="r")
ax.scatter(1, 1, color="g")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
m = -1
plt.plot()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
X = np.arange(xmin, xmax, 0.1)
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
m = -1
for m in np.arange(0, 6, 0.1):
 ax.plot(X, m * X )
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="r")
ax.scatter(1, 1, color="g")
plt.plot()
plt.show()


import numpy as np
from collections import Counter

class Perceptron:

 def __init__(self, weights,bias=1,learning_rate=0.3):

     self.weights = np.array(weights)
     self.bias = bias
     self.learning_rate = learning_rate


 def unit_step_function(x):
  if x <= 0:
   return 0
   else:
    return 1

 def __call__(self, in_data):
  in_data = np.concatenate( (in_data, [self.bias]) )
  result = self.weights @ in_data
   return Perceptron.unit_step_function(result)

 def adjust(self, target_result,in_data):
    if type(in_data) != np.ndarray:
  in_data = np.array(in_data) # 
  calculated_result = self(in_data)
    error = target_result - calculated_result
 if error != 0:
 in_data = np.concatenate( (in_data, [self.bias]) )
 correction = error * in_data * self.learning_rate
 self.weights += correction
 
 def evaluate(self, data, labels):
 evaluation = Counter()
 for sample, label in zip(data, labels):
 result = self(sample) # predict
 if result == label:
evaluation["correct"] += 1
 else:
 evaluation["wrong"] += 1
 return evaluation



import numpy as np
from perceptron import *

def labelled_samples(n):
 for _ in range(n):
 s = np.random.randint(0, 2, (2,))
 yield (s, 1) if s[0] == 1 and s[1] == 1 else (s, 0)

p = Perceptron(weights=[0.3, 0.3, 0.3],
  learning_rate=0.2)

for in_data, label in labelled_samples(30):
 p.adjust(label, in_data)

test_data, test_labels = list(zip(*labelled_samples(30)))

evaluation = p.evaluate(test_data, test_labels)
print(evaluation)
mport matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
X = np.arange(xmin, xmax, 0.1)
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="r")
ax.scatter(1, 1, color="g")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
m = -p.weights[0] / p.weights[1]
c = -p.weights[2] / p.weights[1]
print(m, c)
ax.plot(X, m * X + c )
plt.plot()
plt.show()
