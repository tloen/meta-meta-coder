#python that generates random permutations
#takes in p, some dimension of input
#generates random permutation p*p matrix
#takes in p,m generates two matrices (one input, one output
import random
import numpy as np

def genPM(d):
	pm = np.zeros((d,d))
	colsleft = np.arange(d)
	rangeBound = d-1;
	for x in range(0,d):
		col = random.randint(0, rangeBound)
		pm[x,colsleft[col]] = 1
		colsleft = np.delete(colsleft,col)
		rangeBound -= 1
	return pm

def genIO(pm,ninputs):
	veclen = pm.shape[0]
	inp = np.zeros((veclen, ninputs))
	for x in range(0,veclen):
		for y in range(0,ninputs):
		 	inp[x,y] = random.randint(0,9)
	return inp, np.matmul(pm,inp)