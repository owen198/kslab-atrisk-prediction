# coding: utf-8

import matplotlib.pyplot as plt

def generate_boxplot(data, title, datasets_small_name):
	fig = plt.figure(figsize=(8,6))
 
	bplot = plt.boxplot(data,
	            notch=False, # box instead of notch shape
	            #sym='rs',    # red squares for outliers
	            vert=True)   # vertical box aligmnent
	 
	plt.xticks([y+1 for y in range(len(data))], datasets_small_name)
	plt.xlabel('Dataset')
	 
	for components in bplot.keys():
	    for line in bplot[components]:
	        line.set_color('black')     # black lines
	 
	plt.title(title)
	plt.savefig('result/' + title + '.png')


