import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import glob
import ntpath
import pprint


#To_do
'''
1. Train this NN using a ML library that already has data. Try the spiral dataset!
2. Update the backpropogation function within the NeuralNetwork class, as currently it looks messy and does not make use of good coding practice.
3. The neural network is currently not very effective! Improve back propogation function!
4. Read up more on child and parent classes.
5. Improve accuracy metric.
6. Clean up code!!!!!!!!!!!!!!!!! Remove all unneccesary comments and code.
7. Neaten different functions and classes, as currently its messy and can result in ineffeciencies.
8. Improve get_statistics function!!!!!!!!!!!!!!!!!!!!!!!!!!!!
9. Implement F1 score as alternative statistic.
10. Add check in NN class, that ensures at least 2 hidden layers are added and that last layer has SoftMax activation.
'''

class HiddenLayer(object):
    '''Hidden layer object used by NeuralNetwork class to store the weights and biases of an initiated layer.'''
    def __init__(self,weights,biases,activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation()
      
    @staticmethod
    def initialize_weights(input_size, neuron_count):
        return np.random.uniform(-0.5,0.5,(input_size,neuron_count))*0.05
    
    @staticmethod
    def initialize_biases(neuron_count):
        return np.zeros((1,neuron_count))       
    
    def f_propogate(self,x):
        #Forward propogate one layer.
        self.input = x
        self.z = np.dot(self.input,self.weights) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a
 
class ReLu(object):       
    '''Rectified linear activation function'''
    def forward(self,s):
        self.a = np.maximum(0,s)
        return self.a
        
    def backward(self,s):
        self.d_a = np.where(s > 0,1,0)
        return self.d_a

class LeakyReLu(object):
    '''Leaky rectified linear activation function'''        
    def __init__(self):
        self.slope = 0.01
            
    def forward(self,s):
        self.a = np.where(s>0,s,s*self.slope)
        return self.a
        
    def backward(self,s):
        self.d_a = np.where(s > 0,1,self.slope)
        return self.d_a
        
class Sigmoid(object):
    '''Sigmoid activation function'''
    def forward(self,s):
        self.a = 1/(1+np.exp(-s))
        return self.a

    def backward(self,s):
        self.d_a = s*(1-s)
        return self.d_a

class SoftMax(object):
    ''''Output layer activation function'''
    def forward(self,s):
        self.a = np.exp(s - np.max(s,axis=1,keepdims=True))
        self.a = self.a / np.sum(self.a,axis=1,keepdims=True)
        return self.a
'''    
    #def backward(self,s): #How to get the derivative of the SoftMax function?
        #self.d_a = self.forward(s) * np.identity(self.forward(s).size) - self.forward(s).T @ self.forward(s)
        
        #self.d_a = np.eye(s.shape[0])
        #self.d_a = self.forward(s) * (self.d_a - self.forward(s))
        #return self.d_a
'''

class Accuracy(object):
    '''Metric used to evaluate the neural network'''
    def __init__(self):
        self.correct = 0
        self.count = 0
    
    def increase_correct(self):
        self.correct += 1
    
    def increase_count(self):
        self.count += 1
        
    def evaluate(self,y,pred):
    #y : Actual output.
    #pred : Predicted output.
        self.count += len(y)
        for k1, k2 in zip(y,pred):
            if k1 == k2:
                self.increase_correct()
    
    def set_zero(self):
        self.correct = 0
        self.count = 0
        
    def get(self):
        return self.correct / self.count

class CategoricalCrossEntropy(object):
    '''Calculate the cost of the neural network using categorical cross entropy'''
    #def __init__()

    def get(self,y,pred): #CURRENTLY ONLY WORKS IF OUTPUT IS ONE HOT ENCODING!
        self.cost = -np.log(np.max(np.multiply(y,pred)))
        return self.cost ###################DERIVATIVE OF THE CCE function is simply 1!!!!!. Therefore, the loss will simply be multiplied by 1
    
class NeuralNetwork(object):
    '''Neural network object'''
    '''Object can be initiated using the following approaches:
        1. Load an appropriate pickle file using the NeuralNetwork.load_model() function.
        2. Initialize an instance using the following steps:
            2.1. Call the __init__function by assiging this object to a variable;
            2.2. Call the add_dense function to add a hidden layer to the object;
                (Note, the minimum required layer is 2 i.e. 1 hidden layer and 1 hidden layer with SoftMax activation function)'''
    '''If NeuralNetwork parameters have not yet been optimized, call the optimize_model function'''
    
    def __init__(self, lr=0.01, metric=Accuracy, loss_function=CategoricalCrossEntropy):
        self.metric = metric()
        self.loss_function = loss_function()
        self.hidden_layers = []
        self.learning_rate = lr
    
    def add_dense(self,weights,biases,activation):
        #Adds a hidden layer to the neural network.
        self.hidden_layers.append(HiddenLayer(weights,biases,activation))
                    
    def evaluate(self, input, output):
        self.set_data(input,output)
        self.metric.set_zero()
        print(f'Actual output | {self.get_prediction(output)}')
        print(f'Predicted output | {self.get_prediction(self.forward_propogate())}')
        print(f'{self.get_statistics()}')
        self.metric.set_zero()
    
    def set_data(self,input,output):
        self.input = input
        self.actual_output = output 
       
    def forward_propogate(self):
        #Forward propogates through the entire network.
        self.predicted_output = self.input
        for z in self.hidden_layers:
            self.predicted_output = z.f_propogate(self.predicted_output)
        return self.predicted_output
     
    def backward_propogate(self):
        #Backward propogates through the network while adjusting the weights and biases.        
        self.hidden_layers.reverse()
        self.d_weights = []
        self.d_biases = []
        self.dz = self.predicted_output - self.actual_output
        #self.dz = p - y
        
        for i in range(len(self.hidden_layers)): #UPDATE THIS FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AS IT IS A BIT SLOW AND INEFFECIENT CURRENTLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            n = self.hidden_layers[i]
            if i != 0:
                self.w = self.hidden_layers[i-1].weights
                self.dz = np.dot(self.dz,self.w.T)*n.activation.backward(n.a)
            self.dw = np.dot(self.dz.T,n.input)
            self.db = np.sum(self.dz,axis=0,keepdims=True) #########################RESEARCH HOW TO UPDATE BIASES, AS IF AXIS = 1, THE BIASES SHAPE IS CHANGED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.d_weights.append(self.dw)
        self.d_biases.append(self.db)
        
        self.hidden_layers.reverse()
        self.d_weights.reverse()
        self.d_biases.reverse()
        
        for w,b,n in zip(self.d_weights,self.d_biases,self.hidden_layers):
            n.weights = n.weights - self.learning_rate*w.T
            n.biases = n.biases - self.learning_rate*b
        

       
    def fit_model(self):
        self.forward_propogate()
        self.backward_propogate()
        
    def optimize_model(self, input, output, iterations, update_interval, cycles,batch=True):
        self.metric.set_zero()
        if batch:
            for c in range(cycles):
                for batch, (i, o) in enumerate(zip(input,output)):
                    self.set_data(i, o)
                    for epoch in range(1,iterations+1):
                        network.fit_model()
                        if max(epoch,1) % (iterations/update_interval) == 0:
                            print(f'Cycle: {c+1}/{cycles}\t | Batch: {batch+1}/{len(input)}\t | Epoch: {epoch}\t | {self.get_statistics()}')
        else:
            for c in range(cycles):
                self.set_data(input,output)
                for epoch in range(1, iterations+1):
                    network.fit_model()
                    if max(epoch,1) % (iterations/update_interval) == 0:
                        print(f'Cycle: {c+1}/{cycles}\t | Epoch: {epoch}\t | {self.get_statistics()}')
        self.metric.set_zero() ################ MAKE SURE THIS DOESNT IMPACT THE METRIC OUTPUT!
  
    def get_prediction(self,x):
        if len(x.shape) != 1:
            return np.argmax(x,axis=1)
        else:
            return np.array(np.argmax(x)).reshape(1,1)

    def get_confidence(self):
        #Returns the confidence of the predicted output.
        return np.max(np.multiply(self.actual_output,self.predicted_output),axis=1)
    
    def get_statistics(self):
        #Returns the current network loss, prediction confidence, and specified metric.
        #Requires at least one iteration of the forward_propogate function.
        self.loss_function.get(self.actual_output,self.predicted_output)
        self.metric.evaluate(self.get_prediction(self.actual_output), self.get_prediction(self.predicted_output)) 
        return (f'Loss: {np.round(self.loss_function.cost,6)}\t | Confidence: {np.round(np.mean(self.get_confidence()),6)}\t | {type(self.metric).__name__}: {np.round(self.metric.get(),6)}')
    
    def get_network_information(self):
        #Provides a summary of the network parameters. 
        for i, k in enumerate(self.hidden_layers):
            print(f'Type: {type(k).__name__} {i+1} |Shape: {k.weights.shape}\t | Parameters: {k.weights.shape[0] * k.weights.shape[0] + k.biases.shape[0] * k.biases.shape[1]}\t | Activation: {type(k.activation).__name__}')
        print(f'Loss function: {type(self.loss_function).__name__}')
        print(f'Metric: {type(self.metric).__name__}')
                 
    def save_model(self,file):
        with open(file,'wb') as file:
            pickle.dump(self,file)
            
    def load_model(file):
        with open(file,'rb') as file:
            return  pickle.load(file)
    
class PreprocessData(object):
    '''Preprocess the training and testing data'''
    def __init__(self, train_input, train_output, test_input, test_output, output_count, sample_size, batch_size, dimensions = 2):
        self.output_count = output_count
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.dimensions = dimensions
        
        '''Formats the populations'''
        self.X_pop, self.Y_pop = self.add_population(train_input, train_output)
        self.x_pop, self.y_pop = self.add_population(test_input, test_output)
        
        '''Creates a sample from the population'''
        self.X_sample, self.Y_sample = self.create_sample(self.X_pop,self.Y_pop,self.sample_size)
        self.x_sample, self.x_sample = self.create_sample(self.x_pop,self.y_pop,self.sample_size)
        
        '''Converts sample into batches'''
        self.X_batch, self.Y_batch = self.create_batch(self.X_sample,self.Y_sample, self.batch_size)
        
        '''Sets attributes used by Neural Network'''
        self.input_size = len(self.X_pop[0])
         
    def add_population(self, x_data, y_data):
        return (self.Input(x_data, self.dimensions).data, self.Output(y_data,self.output_count).data)
           
    def create_sample(self, x_d, y_d, sample_size):
        if len(x_d) != len(y_d):
            return 'Matching input and output data have not been provided.'
        del_range = range(np.max(len(x_d)-sample_size,0))
        return np.delete(x_d, del_range,axis=0), np.delete(y_d, del_range,axis=0)  

    def create_batch(self, x_d, y_d, batch_size):
        uneven = len(x_d) % batch_size
        x = np.split(x_d[uneven:], len(x_d[uneven:])/batch_size, axis=0)
        y = np.split(y_d[uneven:], len(y_d[uneven:])/batch_size, axis=0)
        x.append(x_d[:uneven])
        y.append(y_d[:uneven])
        return x, y 
    
    
    class Input(object):
        '''Input data subclass'''
        def __init__(self, data, dim):
            self.data = self.fit_network_parameters(dim, data).astype('float32')
        
        def fit_network_parameters(self,dim, d):
            if len(d.shape) > dim:
                size = 1
                for k in range(1,len(d.shape)):
                    size *= d.shape[k]
                return d.reshape(d.shape[0],size)
            return d
                         
    class Output(object):
        '''Output data subclass'''
        def __init__(self, data, output):
            self.data = self.hot_encode(self.create_array(data),output).astype('float32')
                        
        def create_array(self,d):
            if len(d.shape) == 1:
                return d.reshape(len(d),1)
            else:
                return d
        
        def hot_encode(self, d, output):
            if len(d.shape) > 1:
                if d.shape[1] != output:
                    temp = np.zeros((len(d),output)) 
                    for i,k in enumerate(d):
                        temp[i,k] = 1
            return temp
    
    
def create_standardized_model(input_size,hidden_layer_count,neurons,output_size,learning_rate,metric,loss_function):
    '''Initializes a standardized NeuralNetwork object instance'''
    '''The following specifications are standardized:
        1. Each HiddenLayer has the same number of neurons (excluding the output layer);
        2. Each HiddenLayer has the same activation function.'''
        
    temp = NeuralNetwork(learning_rate)
    size = input_size
    for i in range(hidden_layer_count):
        temp.add_dense(HiddenLayer.initialize_weights(size,neurons),HiddenLayer.initialize_biases(neurons),LeakyReLu)
        size = neurons
    temp.add_dense(HiddenLayer.initialize_weights(size,output_size),HiddenLayer.initialize_biases(output_size),SoftMax)
    return temp

    
'''SETTINGS'''
fit_model = False

'''DIRECTORY'''
file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__))) 


'''REQUIRED INPUTS'''
'''Training setting'''
sample_size = 60000 #Number of items used to train neural network.
batch_size = 32 #Number of items contained within a single batch of training data.
training_iterations = 1000 #Number of iterations per batch used for purpose of training the neural network.
update_frequency = 1 #Frequency of training process progress being printed.
training_cycles = 2

'''Hyper parameters'''
hidden_layer_count = 2 #Number of hidden layers within neural network.
neurons = 100 #Number of neurons per hidden layer.
output_size = 10 #Number of neural network output classes.
learning_rate = 0.01 #Rate at which stochastic gradient descent is applied for purpose of backpropogation.


'''KERAS DATASETS'''
from keras.datasets import mnist as dataset
#from keras.datasets import cifar10 as dataset
#from keras.datasets import fashion_mnist as dataset

'''DATA IMPORT AND FORMATTING'''
(train_x,train_y), (test_x, test_y) = dataset.load_data()
data = PreprocessData(train_x, train_y, test_x, test_y, output_size, sample_size, batch_size)

'''NEURAL NETWORK INITIALIZATION.'''
if input('Would you like to load a pre-existing model? (Y/N)' + '\n').lower() == 'y': 
    try: #UPDATE THIS TRY EXCEPT STATEMENT!!!!!!!!!!!!!!!!!!!!! AS CURRENTLY THE EXCEPT IS TO GENERIC!!!!!!!!!!!!!!!!!!!!!!
        files = glob.glob(os.path.join(file_directory,'*.pickle'))
        print('The following files are available for import:')
        for i,file in enumerate(files):
            print(f'Name: {ntpath.basename(file)} \t |Reference: {i}')
        
        index = ''
        while index == '':
            index = input("\n" + "Please specify the pickle file you'd like to import?" + "\n")
        network = NeuralNetwork.load_model(os.path.join(file_directory,ntpath.basename(files[int(index)])))
        print('Model successfully loaded.' + '\n')
    except:
        print('Failed to load model.')
else:
    network = create_standardized_model(data.input_size,hidden_layer_count,neurons,output_size,0.01,Accuracy,CategoricalCrossEntropy)
network.get_network_information() #Returns network specifications.
print('\n')


'''TRAIN NEURAL NETWORK'''
if fit_model == True:
    print('Training neural network...')
    network.optimize_model(data.X_batch, data.Y_batch, training_iterations, update_frequency, training_cycles,batch=True)
    print('\n')


'''EVALUATE MODEL'''
'''extrapolates the trained neural network over the training data population'''
print('Training dataset results...')
network.evaluate(data.X_pop,data.Y_pop)
print('\n')

'''extrapolates the trained neural network over the testing data population'''
print('Testing datasets results...')
network.evaluate(data.x_pop,data.y_pop)
print('\n')

'''SAVE NEURAL NETWORK PARAMETERS'''
if input('Would you like to save this model ? (Y/N)' + '\n').lower() == 'y':
    try:
        name = ''
        while name == '':
            name = input('File name ?' + '\n')        
        network.save_model(os.path.join(file_directory,f'{name}.pickle'))
        print('Model successfully exported.')
    except:
        print('Failed to export model.')
print('Script terminated.')


random_selection = random.randrange(0,test_x.shape[0])
network.set_data(data.x_pop[random_selection],data.y_pop[random_selection])
print(network.get_prediction(network.forward_propogate()))
plt.imshow(test_x[random_selection])
plt.show()






        
    