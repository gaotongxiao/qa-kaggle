from collections import Counter
import pickle
import numpy as np
import pdb

c = pickle.load(open("data/tf.md", 'rb'))
a = np.zeros([400003, 1], dtype=int)
for k, v in c.items():
    a[k] = v
pickle.dump(a, open("data/tf_np.md", "wb"), True)
