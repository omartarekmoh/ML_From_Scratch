import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

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
            
            weight_matrix = np.random.normal(loc=0.0, scale=stddev, size=(out_dim, in_dim))
            bias_vector = np.random.normal(loc=0.0, scale=stddev, size=(out_dim, 1))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def activate(self, Z, activation):
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        elif activation == 'linear':
            return Z
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'binary':
            return (Z > 0.5).astype(int)  # Binary activation for output layer
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    
    def activation_derivative(self, A, activation):
        if activation == 'relu':
            return (A > 0).astype(float)
        elif activation == 'tanh':
            return 1 - np.power(A, 2)
        elif activation == 'sigmoid':
            return A * (1 - A)
        elif activation == 'linear':
            return np.ones_like(A)
        elif activation == 'softmax':
            return A * (1 - A)
        elif activation == 'binary':
            return 1
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        
    def feed_forward(self, X):
        A = X
        activations = [A]
        for weights, bias, activation in zip(self.weights, self.biases, self.activations):
            Z = np.dot(weights, A) + bias
            A = self.activate(Z, activation)
            activations.append(A)
        return activations
    
    
    def backward_propagation(self, X, y, activations):
        dz = []
        m = X.shape[1]
        dW = []
        dB = []
        for i in reversed(range(1, len(self.layers))):
            if i == len(self.layers)-1:
                dz = activations[i] - y
            else:
                dz = np.dot(self.weights[i].T, dz) * self.activation_derivative(activations[i], self.activations[i])
            
            dw = np.dot(dz, activations[i-1].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            
            dW.append(dw)
            dB.append(db)
        
        return dW[::-1], dB[::-1]

    def update_parameters(self, dW, dB, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * dB[i]


    def train(self, X, y, learning_rate=0.01, epochs=1000):
        m = X.shape[1]
        for epoch in range(epochs):
            total_loss = 0
            for i in range(m):
                x_sample = X[:, i:i+1]
                y_sample = y[:, i:i+1]
                
                activations = self.feed_forward(x_sample)
                dW, dB = self.backward_propagation(x_sample, y_sample, activations)
                self.update_parameters(dW, dB, learning_rate)
                
                loss = self.compute_loss(activations[-1], y_sample)
                total_loss += loss
            
            avg_loss = total_loss / m
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Average Loss: {avg_loss}')
    
    def compute_loss(self, A, y):
        m = y.shape[1]
        loss = -np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) / m
        return loss



X = np.random.rand(100, 4)
y = np.random.randint(0, 2, size=(100, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

nn = NeuralNetwork(input_size=X_train_normalized.shape[1])

nn.add_layer(8, activation='relu')
nn.add_layer(4, activation='tanh')
nn.add_layer(1, activation='sigmoid')

nn.initialize_weights()

nn.train(X_train_normalized.T, y_train.T, learning_rate=0.01, epochs=1000)

activations_test = nn.feed_forward(X_test_normalized.T)
predictions_test = (activations_test[-1] > 0.5).astype(int)  

test_loss = nn.compute_loss(activations_test[-1], y_test.T)

print("Test Loss: ", test_loss)
print("True Labels on Test Set:")
print(y_test.T)
print("Predictions on Test Set:")
print(predictions_test)
