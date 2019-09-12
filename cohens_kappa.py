import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


fp = open('C:\\Users\\user\\Desktop\\raptor_fb_done.txt', 'r')
lines = fp.readlines()
y_nik = []
y_mits = []
for line in lines:
	line = line.strip('\n')
	l = line.split('\t')
	y_nik.append(l[0])
	y_mits.append(l[1])
	
 


print(cohen_kappa_score(y_nik, y_mits))

#run models with dimitris labels