import pickle_utils as pu
import numpy as np
import sys

X, z = pu.load(sys.argv[1])

max_t = max(t for _, t in X)+1
N_cats = max(map(lambda x: x[0], X))+1
n_samples = np.zeros([N_cats, max_t], dtype=np.int)
proba = np.zeros([N_cats, max_t, N_cats], dtype=np.float)

for (cat, t), c in zip(X, z):
    n_samples[cat,t] += 1
    proba[cat,t,c] += 1

import code; code.interact(local=locals())

t_counts = np.sum(n_samples, axis=0)
t_counts /= np.sum(t_counts)


n_samples = np.tile(np.expand_dims(n_samples, 2), [1,1,N_cats])
mask = n_samples != 0
proba[mask] /= n_samples[mask]
proba[~mask] = 0.0




pu.dump((n_samples, proba), sys.argv[2])
sys.exit(0)




# This is all only for time instant t=1

cats = list([] for _ in range(N_cats))
for (cat, t), c in zip(X, z):
    if t <= 0:
        continue
    cats[cat].append((t, c))

mean = np.zeros([N_cats, N_cats], dtype=np.float)
num = np.zeros_like(mean, dtype=np.int)

for cat in range(N_cats):
    for t, c in cats[cat]:
        mean[cat,c] += 1/t
        num[cat,c] += 1
mean /= num

#for cat in range(N_cats):
#    for t, c in cats[cat]:


#var = np.zeros((lambda n: n*(n+1)//2)(len(mean)), dtype=np.float)
#b_c = np.zeros([N, mean.shape[0]])
#for j, (i, (cat, t)) in enumerate(filter(
#        lambda a: a[1][1] > 0,
#        enumerate(X))):
#    b_c[j,:] = mean
#    b_c[j,z[i]] -= t
