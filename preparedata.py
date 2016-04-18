import os
import sys

class PrepareData(object):
	"""docstring for PrepareData"""
	def __init__(self, filename):
		super(PrepareData, self).__init__()
		self.fileName = filename
		self.data = []
		self.label = {"BRICKFACE": [1, 0, 0, 0, 0, 0, 0], "SKY": [0, 1, 0, 0, 0, 0, 0], "FOLIAGE": [0, 0, 1, 0, 0, 0, 0], "CEMENT": [0, 0, 0, 1, 0, 0, 0], "WINDOW":[0, 0, 0, 0, 0, 1, 0, 0], "PATH": [0, 0, 0, 0, 0, 1, 0], "GRASS": [0, 0, 0, 0, 0, 0, 0, 1]}

	"""Reading file and getting data method"""
	def getData(self):
		#read data
		file = open(os.path.join(sys.path[0], self.fileName), "r")
		while True:
			line = file.readline()
			if line == '':
				break
			line = line.replace('\n', '')
			subData = line.split(',')
			self.data.append(subData)
		file.close()

		#convert data to number
		for item in self.data:
			item[0] = self.label[item[0]]
			for i in range(1, len(item)):
				item[i] = float(item[i])
		#print self.data
		return self.data

