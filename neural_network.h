# pragma once

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
    // Orthogonal initialization
    // Custom (allow users to enter their own starting weights)

// Expose a method for Batch Normalization

// Incorporate momentum based learning
// Incorporate stochastic gradient descent

// For classification models, allow users to pass in an enum and expose a "predict_classify" which will return their enum for the class
//


// NeuralNet: A configurable feedforward neural network (MLP).
// The user can specify the layer dimensions (including hidden layers),
// the activation function (and its derivative), and the cost function (and its derivative).
template<typename T>
class NeuralNet {
public:

    // Structure for storing parameters (weights and biases) for each layer.
    struct Parameters {
        Matrix<T> W; // Weight matrix.
        Matrix<T> b; // Bias matrix (stored as 1 x n, to be broadcast).
        Parameters() : W(0, 0, T()), b(0, 0, T()) {}
    };

    // Type aliases for function objects.
    // ActivationFunction: applied element-wise (e.g., ReLU, sigmoid, etc.).
    // ActivationFunctionDerivative: computes dZ = dA ⊙ g'(Z) given the upstream gradient dA and the pre-activation matrix Z.
    using ActivationFunction = std::function<T(T)>;
    using ActivationFunctionDerivative = std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)>;
    
    // CostFunction: computes the cost (e.g., mean squared error) given the network output and targets.
    // CostFunctionDerivative: computes the derivative of the cost function with respect to the network output.
    using CostFunction = std::function<T(const Matrix<T>&, const Matrix<T>&)>;
    using CostFunctionDerivative = std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)>;

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
              CostFunctionDerivative cost_deriv,
              bool standarize_data);

    // Train the network on input X with targets Y for a given number of epochs and learning rate.
    void train(const Matrix<T>& X, const Matrix<T>& Y, int epochs, T learning_rate);
    void train_mini_batch(const Matrix<T>& X, const Matrix<T>& Y, int epochs, T learning_rate, size_t batch_size);
    void train_stochastic(const Matrix<T>& X_original, const Matrix<T>& Y, int epochs, T learning_rate);

    // Predict outputs for a given input X.
    Matrix<T> predict(const Matrix<T>& X);
    
    std::vector<Parameters> getParameters();
    void setParameters(std::vector<Parameters> _params);


    // Getters
    const std::vector<int>& getLayerDims() const { return layer_dims; }
    const std::vector<Parameters>& getParams() const { return params; }
    bool getStandardizeData() const { return standardize_data; }
    ActivationFunction getActivationFunction() const { return activation; }
    ActivationFunctionDerivative getActivationDerivative() const { return activation_deriv; }
    CostFunction getCostFunction() const { return cost_func; }
    CostFunctionDerivative getCostFunctionDerivative() const { return cost_deriv; }
    const std::vector<T>& getCostHistory() const;


    // Setters
    void setLayerDims(const std::vector<int>& dims) { layer_dims = dims; }
    void setParams(const std::vector<Parameters>& _params) { params = _params; }
    void setStandardizeData(bool value) { standardize_data = value; }
    void setActivationFunction(ActivationFunction func) { activation = func; }
    void setActivationDerivative(ActivationFunctionDerivative deriv) { activation_deriv = deriv; }
    void setCostFunction(CostFunction func) { cost_func = func; }
    void setCostFunctionDerivative(CostFunctionDerivative deriv) { cost_deriv = deriv; }

private:

    // The layer dimensions (including input and output layers).
    std::vector<int> layer_dims;
    // Parameters for each layer (for layer l, stored in params[l-1]).
    std::vector<Parameters> params;
    // Cost history for every training epoch.
    std::vector<T> cost_history;
    bool standardize_data;
    
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
        std::vector<Matrix<T>> Z;
        // A[l]: activation output at layer l, with A[0] = input X.
        std::vector<Matrix<T>> A;
    };

    // Perform forward propagation from input X.
    Cache forwardPropagation(const Matrix<T>& X);
    
    // Perform back propagation given the cache from forward propagation and target Y.
    void backPropagation(const Matrix<T>& Y, const Cache& cache, T learning_rate);
};




// --- Pre-made Activation and Cost Functions --- //

namespace CostFunctions {

    template<typename T>
    T meanSquaredError(const Matrix<T>& output, const Matrix<T>& target);

    template<typename T>
    Matrix<T> MSE_derivative(const Matrix<T>& finalOutput, const Matrix<T>& trueLabels);

    template<typename T>
    T weightedMSE(const Matrix<T>& output, const Matrix<T>& target);

    template<typename T>
    Matrix<T> weightedMSE_derivative(const Matrix<T>& output, const Matrix<T>& target);

    template<typename T>
    T binaryCrossEntropy(const Matrix<T>& output, const Matrix<T>& target);

    template<typename T>
    Matrix<T> binaryCrossEntropyDerivative(const Matrix<T>& finalOutput, const Matrix<T>& trueLabels);
}

namespace ActivationFunctions {

    template<typename T>
    T ReLU(T x);

    template<typename T>
    T ReLU_derivative(T x);

    template<typename T>
    Matrix<T> ReLU_activation_derivative(const Matrix<T>& dA, const Matrix<T>& preActivation);

    template<typename T>
    T sigmoid(T x);

    template<typename T>
    T sigmoid_derivative(T x);

    template<typename T>
    Matrix<T> sigmoid_activation_derivative(const Matrix<T>& dA, const Matrix<T>& preActivation);

}

#include "neural_network.cpp"