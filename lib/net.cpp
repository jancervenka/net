/*
 * 2020, Jan Cervenka
 * jan.cervenka@yahoo.com
 */
#include "net.h"
#define W_SCALE 0.01
#define L2_LAMBDA 1e-3
#define LEARNING_RATE 1e-0

// TODO: pass as X, y as references?
// TODO: predict_class method

/*
 * Class constructor.
 *
 * n_features: number of features
 * n_classes: number of classes
 * hidden_size: number of units in the hidden layer
 */
Net::Net(int n_features, int n_classes, int hidden_size) {

    // random is uniform in [-1, 1], change to normal?
    W0 = Eigen::MatrixXd::Random(n_features, hidden_size) * W_SCALE;
    W1 = Eigen::MatrixXd::Random(hidden_size, n_classes) * W_SCALE;
    b0 = Eigen::RowVectorXd::Zero(hidden_size);
    b1 = Eigen::RowVectorXd::Zero(n_classes);

}

/*
 * Class destructor.
 */
Net::~Net() {
    std::cout << "Done!" << std::endl;
}

/*
 * Applies relu (rectified linear unit) to a matrix.
 * relu(x) = max(x, 0)
 *
 * layer_product: X * W + b
 * returns: matrix after relu activation
 */
Eigen::MatrixXd Net::relu(const Eigen::MatrixXd &layer_product) {

    return (layer_product.array() < 0).select(0, layer_product);
}

/*
 * Computes the raw score values
 * (without softmax normalization) from the input.
 *
 * X: input matrix
 * returns: score matrix
 */
Eigen::MatrixXd Net::forward_pass(const Eigen::MatrixXd &X) {

    Eigen::MatrixXd hidden_activations = relu((X * W0).rowwise() + b0);
    return (hidden_activations * W1).rowwise() + b1;
}

/*
 * Computes the fowrard pass and applies softmax.
 *
 * X: input matrix
 * returns: probability matrix
 */
Eigen::MatrixXd Net::predict(const Eigen::MatrixXd &X) {
    // apply exp elementwise on X
    Eigen::MatrixXd exp_scores = forward_pass(X).array().exp();
    // sum rows of exp_scores to get a vector of n elements,
    // n is the number of rows in X
    // use .col(0) to get a column vector
    Eigen::VectorXd row_sums = exp_scores.rowwise().sum().col(0);

    // divide element-wise every column vector of exp_scores by
    // the column of row sums to get the softmax rows (rows sums to 1)
    // we need to use array() because division (as opposed to +) is not
    // elementwise for Matrix/Vector types
    return exp_scores.array().colwise() / row_sums.array();
}

/*
 * Computes L2 loss value from the current weights.
 *
 * returns: L2 loss
 */
double Net::get_l2_loss() {
    // L2 only applied to W0 (hidden layer)
    return 0.5 * L2_LAMBDA * W0.array().pow(2).sum();
}

/*
 * Computes softmax loss given the current predictions and the
 * true labels.
 *
 * X: input matrix
 * y: true labels
 * returns: softmax loss
 */
double Net::get_softmax_loss(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {

    int correct_class;
    double neg_log_proba_sum = 0;
    Eigen::MatrixXd probas = predict(X);

    for (int i = 0; i < y.size(); i++) {
        // correct class for sample i, (class labels must start from 0)
        correct_class = y(i);
        neg_log_proba_sum += -std::log(probas(i, correct_class));
    }

    return neg_log_proba_sum / (double) y.size();
}

/*
 * Computes total loss as a sum of softmax loss and L2 loss.
 *
 * X: input matrix
 * y: true labels
 * returns: overall loss
 */
double Net::get_loss(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
    return get_softmax_loss(X, y) + get_l2_loss();
}

/*
 * Backpropagates one parameter update.
 *
 * X: input matrix
 * y: true labels
 */
void Net::backpropagate(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {

    int correct_class;
    Eigen::MatrixXd dscores = predict(X);

    for (int i = 0; i < y.size(); i++) {
        correct_class = y(i);
        dscores(i, correct_class) -= 1;
    }

    dscores /= (double) y.size();

    Eigen::MatrixXd hidden_activations = relu((X * W0).rowwise() + b0);
    Eigen::MatrixXd dW1 = hidden_activations.transpose() * dscores;
    // b1 is k biases where k is the number of classes, thats why the
    // the colwise sum of dscores (dscores has k columns, one for each class)

    Eigen::RowVectorXd db1 = dscores.colwise().sum();

    Eigen::MatrixXd dhidden = dscores * W1.transpose();
    // this is equivalent to numpy dhidden[hidden_activations <= 0] = 0
    // where dhidden and hidden_activations are same sized matrices
    dhidden = (hidden_activations.array() <= 0).select(0, dhidden);

    Eigen::MatrixXd dW0 = X.transpose() * dhidden;
    Eigen::RowVectorXd db0 = dhidden.colwise().sum();

    dW1 += L2_LAMBDA * W1;
    dW0 += L2_LAMBDA * W0;

    W0 += -LEARNING_RATE * dW0;
    W1 += -LEARNING_RATE * dW1;
    b0 += -LEARNING_RATE * db0;
    b1 += -LEARNING_RATE * db1;
}

/*
 * Trains the network.
 *
 * X: input matrix
 * y: true labels
 * epochs: number of training epochs
 */
void Net::fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y, int epochs) {

    for (int i = 0; i < epochs; i++) {
        backpropagate(X, y);
        std::cout << "Epoch " << i << " loss: " << get_loss(X, y) << std::endl;
    }
}
