from network import NeuralNetwork
from layer import Layer
from activations import ActivationFunctions


# TODO: Add bigger networks in layers and neurons.
class Topologies:
    @staticmethod
    def wide():
        network = NeuralNetwork()

        network.add(
            # layer(Neuronas de entrada, neuronas de salida, funcion de entrada, funcion de salida)
            Layer(3, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(32, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(32, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(
                32, 1, ActivationFunctions.linear, ActivationFunctions.linear_derivative
            )
        )
        return network

    @staticmethod
    def medium():
        network = NeuralNetwork()

        network.add(
            Layer(3, 16, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(16, 8, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(
                8, 1, ActivationFunctions.linear, ActivationFunctions.linear_derivative
            )
        )
        return network

    @staticmethod
    def small():
        network = NeuralNetwork()

        network.add(
            Layer(3, 8, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(
                8, 1, ActivationFunctions.linear, ActivationFunctions.linear_derivative
            )
        )
        return network

    # FIXME bottle neck not working, don't know why, but probably it's the topology structure itself.
    @staticmethod
    def bottle_neck():
        network = NeuralNetwork()

        network.add(
            Layer(3, 32, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(32, 16, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(16, 8, ActivationFunctions.relu, ActivationFunctions.relu_derivative)
        )

        network.add(
            Layer(
                8, 1, ActivationFunctions.linear, ActivationFunctions.linear_derivative
            )
        )
