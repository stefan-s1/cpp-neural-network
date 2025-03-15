#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <functional>
#include <cassert>
#include <iostream>
#include "matrix.h"


// TODO:
// allow users to configure different activation functions for final layer and hidden layers

// TODO:
// expose different initialization options:
    // Xavier initialization
    // He initialization
    // zero initialization
    // Lecun initialization
    // Orthogonal initialization:

// Expose a method for Batch Normalization

// NeuralNet: A configurable feedforward neural network (MLP).
// The user can specify the layer dimensions (including hidden layers),
// the activation function (and its derivative), and the cost function (and its derivative).
template<typename T>
class NeuralNet {
public:
    // Type aliases for function objects.
    // ActivationFunction: applied element-wise (e.g., ReLU, sigmoid, etc.).
    // ActivationFunctionDerivative: computes dZ = dA ⊙ g'(Z) given the upstream gradient dA and the pre-activation matrix Z.
    using ActivationFunction = std::function<T(T)>;
    using ActivationFunctionDerivative = std::function<QSMatrix<T>(const QSMatrix<T>&, const QSMatrix<T>&)>;
    
    // CostFunction: computes the cost (e.g., mean squared error) given the network output and targets.
    // CostFunctionDerivative: computes the derivative of the cost function with respect to the network output.
    using CostFunction = std::function<T(const QSMatrix<T>&, const QSMatrix<T>&)>;
    using CostFunctionDerivative = std::function<QSMatrix<T>(const QSMatrix<T>&, const QSMatrix<T>&)>;

    // Constructor:
    // layer_dims: a vector specifying the number of neurons per layer (including input and output).
    // activation: default is ReLU.
    // activation_deriv: default is the ReLU activation derivative.
    // cost_func: default is mean squared error.
    // cost_deriv: default is the derivative of mean squared error.
    NeuralNet(const std::vector<int>& layer_dims);

    NeuralNet(const std::vector<int>& layer_dims,
              ActivationFunction activation,
              ActivationFunctionDerivative activation_deriv,
              CostFunction cost_func,
              CostFunctionDerivative cost_deriv);

    // Train the network on input X with targets Y for a given number of epochs and learning rate.
    void train(const QSMatrix<T>& X, const QSMatrix<T>& Y, int epochs, T learning_rate);
    
    // Predict outputs for a given input X.
    QSMatrix<T> predict(const QSMatrix<T>& X);
    
    // Return the cost history collected during training.
    const std::vector<T>& getCostHistory() const;

private:
    // Structure for storing parameters (weights and biases) for each layer.
    struct Parameters {
        QSMatrix<T> W; // Weight matrix.
        QSMatrix<T> b; // Bias matrix (stored as 1 x n, to be broadcast).
        Parameters() : W(0, 0, T()), b(0, 0, T()) {}

    };

    // The layer dimensions (including input and output layers).
    std::vector<int> layer_dims;
    // Parameters for each layer (for layer l, stored in params[l-1]).
    std::vector<Parameters> params;
    // Cost history for every training epoch.
    std::vector<T> cost_history;
    
    // Activation and cost functions.
    ActivationFunction activation;
    ActivationFunctionDerivative activation_deriv;
    CostFunction cost_func;
    CostFunctionDerivative cost_deriv;

    // Initialize parameters with small random weights and zero biases.
    void initializeParameters(T maxWeight = T(0.01));

    // Cache structure for forward propagation.
    struct Cache {
        // Z[l]: pre-activation matrix at layer l (computed as A[l-1]*W + b).
        std::vector<QSMatrix<T>> Z;
        // A[l]: activation output at layer l, with A[0] = input X.
        std::vector<QSMatrix<T>> A;
    };

    // Perform forward propagation from input X.
    Cache forwardPropagation(const QSMatrix<T>& X);
    
    // Perform back propagation given the cache from forward propagation and target Y.
    void backPropagation(const QSMatrix<T>& Y, const Cache& cache, T learning_rate);
};

// --- Default Activation and Cost Functions --- //

// ReLU activation function.
template<typename T>
T RelU(T x);

// ReLU derivative (element-wise).
template<typename T>
T RelU_derivative(T x);

// Activation derivative for ReLU.
// Computes dZ = dA ⊙ g'(Z) where g'(Z) is computed element-wise on the pre-activation matrix.
template<typename T>
QSMatrix<T> RelU_activation_derivative(const QSMatrix<T>& dA, const QSMatrix<T>& preActivation);

// Mean Squared Error (MSE) cost function.
template<typename T>
T meanSquaredError(const QSMatrix<T>& output, const QSMatrix<T>& target);

// Derivative of MSE with respect to the network output.
template<typename T>
QSMatrix<T> MSE_derivative(const QSMatrix<T>& finalOutput, const QSMatrix<T>& trueLabels);




template<typename T>
T sigmoid(T x);

template<typename T>
T sigmoid_derivative(T x);

// Activation derivative for ReLU.
// Computes dZ = dA ⊙ g'(Z) where g'(Z) is computed element-wise on the pre-activation matrix.
template<typename T>
QSMatrix<T> sigmoid_activation_derivative(const QSMatrix<T>& dA, const QSMatrix<T>& preActivation);


template<typename T>
T binaryCrossEntropy(const QSMatrix<T>& output, const QSMatrix<T>& target);

template<typename T>
QSMatrix<T> binaryCrossEntropyDerivative(const QSMatrix<T>& finalOutput, const QSMatrix<T>& trueLabels);

#include "neural_network.cpp"

#endif // NEURALNET_H
