import numpy as np
import numpy.random as rand
import time as tm
import random
import collections

prob_edge = 0.1
tirage = 100000000
tStart = tm.time()

#-------------------------------------------------------------------------------
comp=0
for i in range(tirage):
    if random.random() <= prob_edge:
        var = True
        comp += 1
    else:
        var = False
print('Time: %.1f s' %(tm.time() - tStart))
print('taux= %.5f' %(float(comp)/float(tirage)))


#-------------------------------------------------------------------------------
comp=0
tStart = tm.time()
for i in range(tirage):
    if rand.rand() <= prob_edge:
        var = True
        comp += 1
    else:
        var = False
print('Time: %.1f s' %(tm.time() - tStart))
print('taux= %.5f' %(float(comp)/float(tirage)))

#-------------------------------------------------------------------------------

tStart = tm.time()


var = np.random.choice([True, False], size=tirage, p=[prob_edge, 1-prob_edge])

print('Time: %.1f s' %(tm.time() - tStart))
print('taux= %.5f' %(float(collections.Counter(var)[True]) / float(tirage)))
