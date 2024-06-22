import numpy as np

class NeuralNetwork:
    def __init__(self, *, input_size):
        self.input_size = input_size
        self.layers = [input_size]
        self.weights = []
        self.biases = []
        self.activations = []
        
    def add_layer(self, layer_size, activation='relu'):
        self.layers.append(layer_size)
        self.activations.append(activation)
        
    def initialize_weights(self):
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layers)):
            in_dim = self.layers[i-1]
            out_dim = self.layers[i]
            stddev = np.sqrt(2 / (in_dim + out_dim))
            
            weight_matrix = np.random.normal(loc=0.0, scale=stddev, size=(in_dim, out_dim))
            bias_vector = np.random.normal(loc=0.0, scale=stddev, size=(1, out_dim))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def activate(self, Z, activation):
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        elif activation == 'linear':
            return Z
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    
    def feed_forward(self, X):
        A = X
        activations = [A]
        for weights, bias, activation in zip(self.weights, self.biases, self.activations):
            Z = np.dot(A, weights) + bias
            A = self.activate(Z, activation)
            activations.append(A)
        return activations


nn = NeuralNetwork(input_size=2)
nn.add_layer(4, activation='relu')
nn.add_layer(3, activation='tanh')
nn.add_layer(2, activation='softmax')

nn.initialize_weights()

X = np.array([[0.1, 0.2], [-0.3, 0.4]])

activations = nn.feed_forward(X)

for i, activation in enumerate(activations):
    print(f"Layer {i}:")
    print(activation)
    print()
