
'''
Fruit classifier neural network, with a single 2-feature input layer (weight: 0-green, 1-yellow) and a single binary output layer (0-lime, 1-lemon)
'''

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):    # Initialize reandom weights
        random.seed(1)
        self.synaptic_weights = 2 * random.random((2, 1)) - 1
      
    def __sigmoid(self, x):    # Define sigmoid function for forward propogation
        return 1 / (1 + exp(-x))
      
    def __sigmoid_derivative(self, x):    # Define sigmoid derivative for backwards propogation
        return x * (1 - x)
      
    def train(self, inputs, outputs, iterations):    # Training loop
        for i in range(iterations):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

            
    def think(self, inputs):    # Evalutaion function
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    nn = NeuralNetwork()
    inputs = array([[5, 0], [4, 0], [11, 1], [9, 1]])    # Training data inputs
    outputs = array([[0, 0, 1, 1]]).T                    # Training data outputs
    nn.train(inputs, outputs, 10000)                     # Train for 10,000 iterations

print("Enter weight of fruit: ")
x = int(input(">> "))
print("Enter color of fruit: ")
y = int(input(">> "))
final_out = nn.think(array([x, y]))
print(round(final_out[0]))
