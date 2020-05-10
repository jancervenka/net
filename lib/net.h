/*
 * 2020, Jan Cervenka
 * jan.cervenka@yahoo.com
 */
#include <iostream>
#include <Eigen/Dense>

/*
 * Net
 * --------------------
 * Class implementing a two layer neural net
 * classifier with softmax loss and relu activations.
 */
class Net {

    public:
        Net(int n_features, int n_classes, int hidden_size);
        ~Net();
        void fit(const Eigen::MatrixXd &X, Eigen::VectorXi const &y, int epochs);
        Eigen::MatrixXd predict(const Eigen::MatrixXd &X);

    private:
        Eigen::MatrixXd W0, W1;
        Eigen::RowVectorXd b0, b1;
        Eigen::MatrixXd relu(const Eigen::MatrixXd &layer_product);
        Eigen::MatrixXd forward_pass(const Eigen::MatrixXd &X);
        double get_l2_loss();
        double get_softmax_loss(const Eigen::MatrixXd &X, const Eigen::VectorXi &y);
        double get_loss(const Eigen::MatrixXd &X, const Eigen::VectorXi &y);
        void backpropagate(const Eigen::MatrixXd &X, const Eigen::VectorXi &y);

};
