# NeuralNet-Cpp

A lightweight, header-only neural network and matrix math library built from scratch.  
Supports dense feedforward neural networks with customizable architecture, activation functions, and cost functions.

Requires `C++17` or above, if compiled with `C++20` will enable `std::execution::unseq_par` optimizations

Matrix multiplication has `/openmp` support if your compiler supports that. Informal benchmarking
on my local machine shows about 2-3x faster training. 

For those interested, profiling shows about 40% of CPU time is spent
in operator* for Matrix\<T>.

---

## 📜 License
   - MIT

---
## 🚀 Motivation
I built this project to deepen my understanding of neural networks under the hood and to practice modern C++.  
Over time, it’s evolved into a general-purpose C++ playground where I experiment with new features and optimization techniques as I learn them.

You may notice the absence of automated testing — that’s intentional. I use this repo as a break from more formal projects, and I wanted to keep it **dependency-free**, including test frameworks. If the project grows in scope or sees broader use, I’ll consider adding tests using something like `gtest`.

---
## ✨ Features

- Header-only, dependency-free C++17+
- Highly customizable:
  - Works with any data type that supports `*`, `+`, and `-` (including `double`, `std::complex`, intervals, polynomials!)
  - Plug in custom cost functions and activation functions (and their derivatives)
- Supports both vanilla, mini-batch and, stochastic gradient descent
- Can standardize input data before training
- Lightweight matrix class with row-major layout and broadcasting support
- Parallelism via OpenMP and C++20 parallel algorithms (where supported)
- Great for learning, research, or small CPU-only models

---

## 📌 Roadmap

- Add Adam optimizer  
- Support custom input/preprocessing layers and custom output layers  
- Add softmax + categorical cross-entropy  
- Implement L2 regularization  
- Add dropout  
- Save/load model parameters  
- Feature normalization support  
- Support for CNNs, RNNs, and LSTMs  
- Support more initalization options: Xavier, He, Zero, Lecun, Orthogonal, and custom starting weights

---

## 🧠 Example Usage

```cpp
auto trainData = csvFileToData<double>("MyData.csv");
Matrix<double> X_train = trainData.first;
Matrix<double> Y_train(trainData.second, trainData.second.size(), 1);

std::vector<int> layer_dims = {X_train.get_cols(), 16, 16, 1};
NeuralNet<double> model(layer_dims);
int epochs = 5000;
double learning_rate = 0.2;
int num_batches = 10;

model.train_mini_batch(X_train, Y_train, epochs, learning_rate, num_batches);

Matrix<double> predictions = model.predict(X_test);
```
---