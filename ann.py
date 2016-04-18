import numpy as np
import random as rand
import activation_functions as activefunc
import preparedata

#Creating a neural network with below properties:
#	Layers: 3
#	The number of training set's attributes: 19
#	The neurons in input layer: 19 + 1 (value 1)
#	The neurons in hidden layer: 39 + 1 (value 1)
#	The neurons in output layer: 7
neuronsInputLayer = 20
neuronsPerHiddenLayer = 40
neuronsOutputLayer = 7
neuronsInLayers = [neuronsInputLayer, neuronsPerHiddenLayer, neuronsOutputLayer]
nLayers = 3
nWeightMat = nLayers - 1
minWeight = -0.5
maxWeight = 0.5
maxLoop = 100
balance = 0.2
maxLearingRatio = 0.96
limit = 0.0001

class NeuralNetwork(object):
	"""docstring for NeuronNetwork"""
	def __init__(self, trainDataSet):
		super(NeuralNetwork, self).__init__()

		#set train dataset
		self.trainSet = []
		self.trainSet = trainDataSet

		#initialize input layer
		self.inputLayer = []

		#initialize hidden layer
		self.hiddenLayer = []

		#initialize output layer
		self.outputLayer = []

		#initialize weight matrix
		self.weightMatrices = []

	"""learning function for ann"""
	def train(self):
		#generate random weights
		self.generateWeights(self.weightMatrices, neuronsPerHiddenLayer - 1, neuronsInputLayer)
		self.generateWeights(self.weightMatrices, neuronsOutputLayer, neuronsPerHiddenLayer)
		#print self.weightMatrices[0]
		
		iterator = 0
		oldCost = 0.0
		cost = 0.0
		learningRatio = maxLearingRatio
		while iterator < maxLoop:
			#loop for each vector
			for vector in self.trainSet:
				#forward propagation
				self.inputLayer.append(vector[1:])
				self.inputLayer[0].insert(0, 1.0)
				#print self.inputLayer

				#multiple inputlayer and weights
				z1 = np.matrix(self.inputLayer) * np.matrix(self.weightMatrices[0])
				z1 = z1.tolist()
				#print z1
				
				#get hidden layer
				t = []
				for i in z1[0]:
					t.append(activefunc.sigmoid(i))
				self.hiddenLayer.append(t)
				#print self.hiddenLayer
				self.hiddenLayer[0].insert(0, 1.0)
				#print self.hiddenLayer
				#print len(self.hiddenLayer)
				#print len(self.hiddenLayer[0])

				#get output layer
				z2 = np.matrix(self.hiddenLayer) * np.matrix(self.weightMatrices[1])
				z2 = z2.tolist()
				#print z2

				#get outputlayer
				t = []
				for i in z2[0]:
					t.append(activefunc.sigmoid(i))
				self.outputLayer.append(t)
				#print self.outputLayer
				
				#get cost function
				cost = self.costFunction(self.outputLayer)
				#print cost
				if np.abs(cost - oldCost) <= limit:
					break
				#print np.abs(cost - oldCost)
				#print cost

				#backward propagation
				delta = []
				t1 = np.zeros((neuronsInLayers[0], neuronsInLayers[1] - 1)).tolist()
				t2 = np.zeros((neuronsInLayers[1], neuronsInLayers[2])).tolist()
				delta.append(t1)
				delta.append(t2)
				#print delta

				#get error value for output layer
				sigma3 = []
				t = []
				for i in range(neuronsInLayers[2]):
					t.append(self.outputLayer[0][i] - vector[0][i])
				sigma3.append(t)
				#print sigma3
				
				#get error value for hidden layer
				sigma2 = []
				sigma2 = np.matrix(sigma3) * np.matrix(np.transpose(self.weightMatrices[1]))
				sigma2 = sigma2.tolist()
				#print sigma2
				derivative = []
				t = []
				for i in self.hiddenLayer[0]:
					t.append(activefunc.derivativeSigmoid(i))
				derivative.append(t)
				#print derivative
				t = []
				for a,b in zip(sigma2[0], derivative[0]):
					t.append(a * b)
				sigma2 = []
				sigma2.append(t[1:])
				#print sigma2

				#calculate delta[0]
				for i in range(len(delta[0])):
					for j in range(len(delta[0][i])):
						delta[0][i][j] += self.inputLayer[0][i] * sigma2[0][j]
				#print delta[0]

				#calculate delta[1]
				for i in range(len(delta[1])):
					for j in range(len(delta[1][i])):
						delta[1][i][j] += self.hiddenLayer[0][i] * sigma3[0][j]
				#print delta[1]

				#derivative for cost function
				costDerivative = []
				costDerivative.append(self.derivativeCostFunction(delta, 0))
				costDerivative.append(self.derivativeCostFunction(delta, 1))
				#print costDerivative[0]

				#update weights
				for k in range(len(self.weightMatrices)):
					for i in range(len(self.weightMatrices[k])):
						for j in range(len(self.weightMatrices[k][i])):
							self.weightMatrices[k][i][j] = self.weightMatrices[k][i][j] - learningRatio * costDerivative[k][i][j]
				#print len(self.weightMatrices[0])

				self.inputLayer = []
				self.hiddenLayer = []
				self.outputLayer = []
				oldCost = cost
				learningRatio = learningRatio / (1.0 + (iterator / maxLoop))

			if np.abs(cost - oldCost) <= limit:
					break
			iterator += 1
		print "Error: %f" %cost
		#print self.outputlayer
		

	"""generate weights"""
	def generateWeights(self, weights, width, height):
		#set the random seed
		rand.seed(0)

		#generate random weight matrix
		tempList = []
		for i in range(0, height):
			arr = np.random.uniform(minWeight, maxWeight, width)
			#print arr.shape
			arr = arr.tolist()
			#print arr
			tempList.append(arr)
		weights.append(tempList)


	"""cost function"""
	def costFunction(self, outputLayer):
		m = len(self.trainSet) * 1.0
		cost = 0.0
		firstHalf = 0.0
		secondHalf = 0.0

		#first half
		for vector in self.trainSet:
			sumTmp = 0.0
			for i in range(7):
				sumTmp += float(vector[0][i]) * np.log(outputLayer[0][i]) + (1.0 - vector[0][i]) * np.log(1.0 - outputLayer[0][i])
			firstHalf += sumTmp
		firstHalf = (-1.0) * firstHalf / m 
		#print firstHalf

		#second half
		for i in range(nWeightMat):
			for j in range(neuronsInLayers[i]):
				for k in range(neuronsInLayers[i + 1] - 1):
					secondHalf += self.weightMatrices[i][j][k] * self.weightMatrices[i][j][k]
		secondHalf = (balance / (2.0 * m)) * secondHalf
		#print secondHalf
		
		#cost
		cost = firstHalf + secondHalf
		return cost		

	"""derivative cost function"""
	def derivativeCostFunction(self, delta, layerIdx):
		if (layerIdx >= len(delta)):
			return 0
		dt = np.zeros((len(delta[layerIdx]), len(delta[layerIdx][0])))
		dt = dt.tolist()
		#print dt
		m = len(self.trainSet) * 1.0
		for i in range(len(delta[layerIdx])):
			for j in range(len(delta[layerIdx][i])):
				if j == 0:
					dt[i][j] = delta[layerIdx][i][j] / m
				else:
					dt[i][j] = delta[layerIdx][i][j] / m + balance * self.weightMatrices[layerIdx][i][j]
		return dt

	"""test function"""
	def test(self, testSet):
		precision = []
		for i in range(7):
			TP = 0.0
			FN = 0.0
			for vector in testSet:
				classGotten = self.classify(vector[1:])
				if vector[0][i] == 1:
					if classGotten[0][i] == 1:
						TP += 1.0
					else:
						FN += 1.0
			if TP == 0.0 and FN == 0.0:
				precision.append(0.0)
			else:
				precision.append(TP / (TP + FN))
		
		#print the precision each class
		print "Precision(brickface) = %f" % precision[0]
		print "Precision(sky) = %f" % precision[1]
		print "Precision(foliage) = %f" % precision[2]
		print "Precision(cement) = %f" % precision[3]
		print "Precision(window) = %f" % precision[4]
		print "Precision(path) = %f" % precision[5]
		print "Precision(grass) = %f" % precision[6]


	"""clasification function"""
	def classify(self, dataVector):
		self.inputLayer = []
		self.hiddenLayer = []
		self.outputLayer = []
		self.inputLayer.append(dataVector)
		self.inputLayer[0].insert(0, 1.0)

		#multiple inputlayer and weights
		z1 = np.matrix(self.inputLayer) * np.matrix(self.weightMatrices[0])
		z1 = z1.tolist()
		#print z1
		
		#get hidden layer
		t = []
		for i in z1[0]:
			t.append(activefunc.sigmoid(i))
		self.hiddenLayer.append(t)
		#print self.hiddenLayer
		self.hiddenLayer[0].insert(0, 1.0)
		#print self.hiddenLayer
		#print len(self.hiddenLayer)
		#print len(self.hiddenLayer[0])

		#get output layer
		z2 = np.matrix(self.hiddenLayer) * np.matrix(self.weightMatrices[1])
		z2 = z2.tolist()
		#print z2
		#get outputlayer
		t = []
		for i in z2[0]:
			t.append(activefunc.sigmoid(i))
		self.outputLayer.append(t)
		for i in range(len(self.outputLayer[0])):
			if self.outputLayer[0][i] == max(self.outputLayer[0]):
				self.outputLayer[0][i] = 1
			else:
				self.outputLayer[0][i] = 0
		return self.outputLayer
