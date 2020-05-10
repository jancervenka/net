/*
 * 2020, Jan Cervenka
 * jan.cervenka@yahoo.com
 */
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "net.h"
#define N_FEATURES 2

/*
 * Container for feature matrix and target vector.
 *
 */
struct XyTuple {
    Eigen::MatrixXd X;
    Eigen::VectorXi y;
};

/*
 * Generates a two cluster dataset.
 *
 * returm: XyTuple struct with an X matrix and y vector
 */
XyTuple generate_binary_classifaction_problem() {

    Eigen::MatrixXd X(40, N_FEATURES);
    Eigen::MatrixXi y(40, 1);
    int i, j, cluster_offset = 0;

    for (i = 0; i < 40; i++) {

        cluster_offset = i > 19 ? 50 : 0;

        for (j = 0; j < N_FEATURES; j++)
            X(i, j) = std::rand() % 5 + cluster_offset;

        y(i, 0) = cluster_offset == 0 ? 0 : 1;
    }

    X = X.array() / X.mean();
    return (struct XyTuple) {.X = X, .y = y};
}

int main() {

    XyTuple problem = generate_binary_classifaction_problem();

    Net nn(N_FEATURES, 2, 64);
    nn.fit(problem.X, problem.y, 100);

    Eigen::MatrixXd y_pred = nn.predict(problem.X);
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    for (int i = 0; i < problem.y.size(); i++) {
        std::cout << "True label: " << problem.y(i) << ", ";
        std::cout << "probabilities: " << y_pred.row(i) << std::endl;
    }
}
