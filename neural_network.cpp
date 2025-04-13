#pragma once

#include <vector>
#include <functional>
#include <cassert>
#include <iostream>
#include "matrix.h"
#include "neural_network.h"

#include <random>    // for std::mt19937 and shuffling of batches in train_mini_batch

// --- Default Activation and Cost Functions Implementation --- //

template<typename T>
T RelU(T x) {
    return std::max(T(0), x);
}

template<typename T>
T RelU_derivative(T x) {
    return (x > T(0)) ? 1 : 0;
}

template<typename T>
Matrix<T> RelU_activation_derivative(const Matrix<T>& dA, const Matrix<T>& preActivation) {
    //component wise multiplication of dA (derivative of cost function applied to A and target Matrix) and the activated matrix
    return dA.hadamardMultiplication(preActivation.component_wise_transformation(&RelU_derivative<T>));
}

template<typename T>
T meanSquaredError(const Matrix<T>& output, const Matrix<T>& target) {
    assert(output.get_rows() == target.get_rows() && output.get_cols() == target.get_cols());
    T error = T();
    size_t total_elements = output.get_rows() * output.get_cols();
    for (size_t row = 0; row < output.get_rows(); ++row) {
        for (size_t col = 0; col < output.get_cols(); ++col) {
            T diff = output(row, col) - target(row, col);
            error += (diff * diff);
        }
    }
    return error / total_elements;
}

template<typename T>
Matrix<T> MSE_derivative(const Matrix<T>& finalOutput, const Matrix<T>& trueLabels) {
    size_t total_elements = finalOutput.get_rows() * finalOutput.get_cols();
    return ((finalOutput - trueLabels) * (2.0 / total_elements));
}


// Default functions for binary classification:
template<typename T>
T sigmoid(T x) {
    return 1 / (1 + std::exp(-x));
}

template<typename T>
T sigmoid_derivative(T x) {
    T s = sigmoid(x);
    return s * (1 - s);
}

template<typename T>
Matrix<T> sigmoid_activation_derivative(const Matrix<T>& dA, const Matrix<T>& preActivation) {
    //component wise multiplication of dA (derivative of cost function applied to A and target Matrix) and the activated matrix
    return dA.hadamardMultiplication(preActivation.component_wise_transformation(&sigmoid_derivative<T>));
}

template<typename T>
T binaryCrossEntropy(const Matrix<T>& output, const Matrix<T>& target) {
    // Compute average binary cross entropy cost.
    T cost = T();
    size_t m = output.get_rows() * output.get_cols();
    for (size_t i = 0; i < output.get_rows(); ++i) {
        for (size_t j = 0; j < output.get_cols(); ++j) {
            T y = target(i, j);
            T o = output(i, j);
            // Avoid log(0) issues.
            T epsilon = 1e-7;
            cost += -y * std::log(o + epsilon) - (1 - y) * std::log(1 - o + epsilon);
        }
    }
    return cost / m;
}

template<typename T>
Matrix<T> binaryCrossEntropyDerivative(const Matrix<T>& finalOutput, const Matrix<T>& trueLabels) {
    // Derivative of BCE with respect to the output of the sigmoid layer.
    // (finalOutput - trueLabels) / (finalOutput * (1 - finalOutput))
    // In practice, combine with the derivative of the sigmoid to avoid numerical issues.
    // This is left as an exercise to refine.
    return (finalOutput - trueLabels); // Simplified; in real use, combine with sigmoid derivative.
}


// --- NeuralNet Class Member Functions --- //
template<typename T>
NeuralNet<T>::NeuralNet(const std::vector<int>& layer_dims,
                        ActivationFunction activation,
                        ActivationFunctionDerivative activation_deriv,
                        CostFunction cost_func,
                        CostFunctionDerivative cost_deriv,
                        bool standardize_data)
    : layer_dims(layer_dims), activation(activation), activation_deriv(activation_deriv),
      cost_func(cost_func), cost_deriv(cost_deriv), standardize_data(true)
{
    initializeParameters();
}

template <typename T>
NeuralNet<T>::NeuralNet(const std::vector<int>& layer_dims)
: layer_dims(layer_dims)
{
activation = &RelU<T>;
activation_deriv = &RelU_activation_derivative<T>;
cost_func = &meanSquaredError<T>;
cost_deriv = &MSE_derivative<T>;
initializeParameters();
}



template<typename T>
void NeuralNet<T>::initializeParameters(T maxWeight) {
    int L = layer_dims.size();
    params.resize(L - 1);
    for (int l = 1; l < L; ++l) {
        Parameters p;
        // Weight matrix dimensions: (layer_dims[l-1] x layer_dims[l])
        p.W = Matrix<T>::initRandomMatrix(layer_dims[l - 1], layer_dims[l], maxWeight);
        // Bias: 1 x layer_dims[l] (to be broadcast during addition).
        p.b = Matrix<T>(1, layer_dims[l], T(0));
        params[l - 1] = p;
    }
}

template<typename T>
std::vector<typename NeuralNet<T>::Parameters> NeuralNet<T>::getParameters(){
    return this->params;
}

template<typename T>
void NeuralNet<T>::setParameters(std::vector<typename NeuralNet<T>::Parameters> _params){
    return this->params = _params;
}

template<typename T>
typename NeuralNet<T>::Cache NeuralNet<T>::forwardPropagation(const Matrix<T>& X) {
    Cache cache;
    cache.A.push_back(X); // A[0] is the input.
    int L = params.size();
    Matrix<T> A = X;
    for (int l = 0; l < L; ++l) {
        // Compute Z = A * W + b.
        Matrix<T> Z = (A * params[l].W) + params[l].b;
        cache.Z.push_back(Z);
        // Apply the activation function element-wise.
        A = Z.component_wise_transformation(activation);
        cache.A.push_back(A);
    }
    return cache;
}


// Y is true labels, cache is obtained from forward propagation and essentially holds the effects of each layer on the subsequent ones 
template<typename T>
void NeuralNet<T>::backPropagation(const Matrix<T>& Y, const Cache& cache, T learning_rate) {
    int L = params.size();
    // Compute initial gradient from the cost derivative. dA is inital gradient
    Matrix<T> dA = cost_deriv(cache.A.back(), Y);

    // Iterate backward over layers.
    for (int current_layer = L - 1; current_layer >= 0; --current_layer) {
        
        // dZ = dA ⊙ g'(Z)
        Matrix<T> dZ = activation_deriv(dA, cache.Z[current_layer]);
        int m = cache.A[current_layer].get_rows();

        // dW = (A_prev^T * dZ) / m.
        Matrix<T> dW = (cache.A[current_layer].transpose() * dZ) * (1.0 / m);

        // Compute db by summing dZ along rows (resulting in a 1 x n matrix) and dividing by m.
        Matrix<T> db(params[current_layer].b.get_rows(), params[current_layer].b.get_cols(), T(0));
        for (size_t i = 0; i < dZ.get_rows(); ++i) {
            for (size_t j = 0; j < dZ.get_cols(); ++j) {
                db(0, j) = db(0, j) + dZ(i, j);
            }
        }
        
        db = db * (1.0 / m);
        
        // dA_prev = dZ * (W^T)
        Matrix<T> dA_prev = dZ * params[current_layer].W.transpose();
        
        // Update parameters.
        dW *= learning_rate;
        params[current_layer].W -= dW;

        db *= learning_rate;
        params[current_layer].b -= db;
        dA = dA_prev;
    }
}

template<typename T>
Matrix<T> NeuralNet<T>::predict(const Matrix<T>& X) {
    Cache cache = forwardPropagation(X);
    return cache.A.back();
}

// X is input matrix, Y is true labels
template<typename T>
void NeuralNet<T>::train(const Matrix<T>& X_original, const Matrix<T>& Y, int epochs, T learning_rate) {
    const Matrix<T>* X_ptr = &X_original;
    if (standardize_data) {
        Matrix<T> X_standardized = X_original.standardize_columns();
        X_ptr = &X_standardized;
    }

    const Matrix<T>& X = *X_ptr;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        Cache cache = forwardPropagation(X);
        T cost = cost_func(cache.A.back(), Y);
        cost_history.push_back(cost);
        backPropagation(Y, cache, learning_rate);
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " cost: " << cost << std::endl;
            // Maybe also periodically output the params (in case training gets interrupted we don't start from zero)
        }
    }
}

template <typename T>
std::vector<std::pair<Matrix<T>, Matrix<T>>> create_random_batches(const Matrix<T>& X, const Matrix<T>& Y, size_t num_batches) {
    size_t num_samples = X.get_rows();
    size_t batch_size = (num_samples + num_batches - 1) / num_batches;  // ceil division

    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0..n-1

    static std::mt19937 rng(42); // You can change this seed for non-deterministic training
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<std::pair<Matrix<T>, Matrix<T>>> batches;
    batches.reserve(num_batches);

    for (size_t i = 0; i < num_batches; ++i) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, num_samples);
        size_t actual_batch_size = end - start;

        std::vector<std::vector<T>> batchX;
        std::vector<std::vector<T>> batchY;
        batchX.reserve(actual_batch_size);
        batchY.reserve(actual_batch_size);

        for (size_t j = start; j < end; ++j) {
            batchX.push_back(X.get_row_copy(indices[j]));
            batchY.push_back(Y.get_row_copy(indices[j]));
        }

        batches.emplace_back(Matrix<T>(batchX), Matrix<T>(batchY));
    }

    return batches;
}

template<typename T>
void NeuralNet<T>::train_mini_batch(const Matrix<T>& X_original, const Matrix<T>& Y, int epochs, T learning_rate, size_t num_batches) {
    
    const Matrix<T>* X_ptr = &X_original;
    if (standardize_data) {
        Matrix<T> X_standardized = X_original.standardize_columns();
        X_ptr = &X_standardized;
    }

    const Matrix<T>& X = *X_ptr;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto batches = create_random_batches(X, Y, num_batches);

        T epoch_total_cost = 0;

        for (size_t i = 0; i < num_batches; ++i) {
            Cache cache = forwardPropagation(batches[i].first);
            T cost = cost_func(cache.A.back(), batches[i].second);
            epoch_total_cost += cost;

            backPropagation(batches[i].second, cache, learning_rate);
        }

        T avg_cost = epoch_total_cost / num_batches;
        cost_history.push_back(avg_cost);

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " average cost: " << avg_cost << std::endl;
        }

    }
}

template<typename T>
void NeuralNet<T>::train_stochastic(const Matrix<T>& X_original, const Matrix<T>& Y, int epochs, T learning_rate) {

    const Matrix<T>* X_ptr = &X_original;
    if (standardize_data) {
        Matrix<T> X_standardized = X_original.standardize_columns();
        X_ptr = &X_standardized;
    }

    const Matrix<T>& X = *X_ptr;


    std::vector<size_t> indices(X.get_rows());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0..n-1
    static std::mt19937 rng(42); // You can change this seed for non-deterministic training

    for (int epoch = 0; epoch < epochs; ++epoch) {

        T epoch_total_cost = 0;

        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < X.get_rows(); ++i) {
            Matrix<T> x_i(X.get_row_copy(indices[i]), 1, X.get_cols());
            Matrix<T> y_i(Y.get_row_copy(indices[i]), 1, Y.get_cols());

            Cache cache = forwardPropagation(x_i);
            T cost = cost_func(cache.A.back(), y_i);
            epoch_total_cost += cost;

            backPropagation(y_i, cache, learning_rate);
        }

        T avg_cost = epoch_total_cost / X.get_rows();
        cost_history.push_back(avg_cost);

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " average cost: " << avg_cost << std::endl;
        }

    }
}

template<typename T>
const std::vector<T>& NeuralNet<T>::getCostHistory() const {
    return cost_history;
}