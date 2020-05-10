[0]: http://eigen.tuxfamily.org/index.php?title=Main_Page
[1]: https://cs231n.github.io/neural-networks-case-study/

# Net

Neural network classifier implemented in C++ and [Eigen numeric library][0].
The code is based on the [Python example][1] from Stanford CS231.

The classifer is using softmax and L2 regularization to compute the loss.
Training is done with vanilla gradient descent. The network architecture
consists of one hidden layer with RELU activations.

## Usage

The `Net` constructor requires number of features and classes in
the classification problem and size of the network hidden layer.

```cpp
Net(int n_features, int n_classes, int hidden_size);
```

Training a model requires a `MatrixXd` feature matrix, `VectorXi`
target labels and number of epochs.

```cpp
void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y, int epochs);
```

Classifying new samples requires a `MatrixXd` feature matrix.
The output is a matrix with class probabilites for each sample.

```cpp
Eigen::MatrixXd predict(const Eigen::MatrixXd &X);
```

### Example

```cpp
int main() {

    Eigen::MatrixXd X_train = load_X_train();
    Eigen::VectorXi y_train = load_y_train();
    Eigen::MatrixXd X = load_X();
    int epochs = 100;

    Net nn(2, 2, 64);
    nn.fit(X_train, y_train, epochs);

    Eigen::VectorXi y_pred = nn.predict(X);
}
```

A working example can be found in `example/main.cpp`. On OS X, it can be built by
running `make` (assuming `Eigen` is in `/usr/local/include`).
