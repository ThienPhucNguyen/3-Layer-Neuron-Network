import ann 
import preparedata

if __name__ == '__main__':
	prepare = preparedata.PrepareData("dataset")
	data = prepare.getData()
	neuralNet = ann.NeuralNetwork(data)
	neuralNet.train()	
	
	prepare = preparedata.PrepareData("test")
	dataTest = prepare.getData()
	neuralNet.test(dataTest)		