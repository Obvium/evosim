import numpy
a = 1 #tuneable param for elu
basiclayout = [3, 3, 2]
numpy.random.seed(2)

class Creature:
    def __init__(self, fat, muscles, energy, NN):
        self.fat = fat
        self.muscles = muscles
        self.emergy = energy
        self.mass = muscles + fat
        self.NN = NN

def ReLUaprox(x):
    return numpy.log(1+numpy.exp(x))


class NeuronLayer():
    def __init__(self, neurons, prevneurons):
        self.weights = 2* numpy.random.random((prevneurons, neurons)) - 1 #weights and biases
        self.biases = 2*numpy.random.random((1,neurons)) - 1

class NN:
    def __init__(self, list):
        self.list = list


    def siggi(self, x):
        return (1/(1+numpy.exp(-x)))



#normalize function
    def ELU(self, x):
        if x.any() >= 0:
            return x
        return a * (numpy.exp(x) - 1)

# could add outputs for diff layers
    def do(self, input):
        output = input
        count = len(self.list)
        for x in range(count):
            print(x)
            # divided by the neurons in the previous layer
            output = self.ELU(numpy.add((numpy.dot(output, self.list[x].weights)),self.list[x].biases)) #/ len(self.list[x].weights))
            print(output)
            #print(self.list[x].biases)
        return output

def generateNetwork(network_map):
    layerlist = []
    for x in range(len(network_map) - 1):
        layerlist.append(NeuronLayer(network_map[x+1],network_map[x]))
        #print(NeuronLayer(network_map[x + 1], network_map[x]).weights)
    return NN(layerlist)




input = [0,0,0,0,0]


#generate tests
#input layer, hidden layers, output layer
map = [5,4,600,600,600,5]
Network = generateNetwork(map)
print(Network.do(input))
#needs a array withlayers and neurons
