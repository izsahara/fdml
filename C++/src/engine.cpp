#include <fdml/utilities.h>
#include <fdml/kernels.h>
#include <fdml/optimizers.h>
#include <fdml/base_models.h>
#include <fdml/deep_models.h>
#include <iostream>
#include <fstream>
#include <rapidcsv.h>

using namespace fdml::kernels;
using namespace fdml::optimizers;
using namespace fdml::base_models::gaussian_process;
using namespace fdml::deep_models::gaussian_process;
using fdml::utilities::metrics::rmse;
using fdml::utilities::operations::write_data;

using std::cout;
using std::endl;

TMatrix read_data(std::string filename) {

    rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams('\t'));
    int nrows = doc.GetRowCount();
    int ncols = doc.GetColumnCount();

    TMatrix data(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i) {
        std::vector<double> row = doc.GetRow<double>(i);
        for (std::vector<double>::size_type j = 0; j != row.size(); j++) {
            data(i, j) = row[j];
        }
    }
    return data;
}

void engine() {    
    TMatrix X_train = read_data("../datasets/engine/X_train.txt");
    TMatrix Y_train = read_data("../datasets/engine/Y_train.txt");
    TMatrix X_test = read_data("../datasets/engine/X_test.txt");
    TMatrix Y_test = read_data("../datasets/engine/Y_test.txt");

    // TMatrix X_train = read_data("E:/23620029-Faiz/GIT/datasets/engine/X_train.txt");
    // TMatrix Y_train = read_data("E:/23620029-Faiz/GIT/datasets/engine/Y_train.txt");
    // TMatrix X_test = read_data("E:/23620029-Faiz/GIT/datasets/engine/X_test.txt");
    // TMatrix Y_test = read_data("E:/23620029-Faiz/GIT/datasets/engine/Y_test.txt");    

    // ======================= LAYER 1  ======================= //

    shared_ptr<Kernel> kernel11 = make_shared<Matern52>(0.0, 1.0);
    shared_ptr<Kernel> kernel12 = make_shared<Matern52>(0.0, 1.0);
    shared_ptr<Kernel> kernel13 = make_shared<Matern52>(0.0, 1.0);

    Node node11(kernel11);
    Node node12(kernel12);
    Node node13(kernel13);

    node11.likelihood_variance.fix();
    node12.likelihood_variance.fix();
    node13.likelihood_variance.fix();

    std::vector<Node> nodes1{ node11, node12, node13 };
    Layer layer1(nodes1);
    layer1.set_inputs(X_train);

    // ======================= LAYER 2  ======================= //
    shared_ptr<Kernel> kernel21 = make_shared<Matern52>(0.0, 1.0);
    shared_ptr<Kernel> kernel22 = make_shared<Matern52>(0.0, 1.0);
    shared_ptr<Kernel> kernel23 = make_shared<Matern52>(0.0, 1.0);
    Node node21(kernel21);
    Node node22(kernel22);
    Node node23(kernel23);

    node21.likelihood_variance.fix();
    node22.likelihood_variance.fix();
    node23.likelihood_variance.fix();

    std::vector<Node> nodes2{ node21, node22, node23 };
    Layer layer2(nodes2);

    // ======================= LAYER 3  ======================= //
    shared_ptr<Kernel> kernel31 = make_shared<Matern52>(0.0, 1.0);
    Node node31(kernel31);
    node31.likelihood_variance.fix();

    std::vector<Node> nodes3{ node31 };
    Layer layer3(nodes3);
    layer3.set_outputs(Y_train);

    // ======================================================= //
    std::vector<Layer> layers{ layer1, layer2, layer3 };
    SIDGP model(layers);
    model.train(500);
    model.estimate();

    MatrixPair Z = model.predict(X_test, Y_test, 100, 5);
    TMatrix mean = Z.first;
    TMatrix var = Z.second;
    // write_data("datasets/engine/mean.txt", mean);
    // write_data("datasets/engine/var.txt", var);

    double rmse_ = rmse(Y_test, mean);
    std::cout << "RMSE = " << rmse_ << std::endl;
    double nrmse = rmse(Y_test, mean) / (Y_test.maxCoeff() - Y_test.minCoeff());
    std::cout << "NRMSE = " << nrmse << std::endl;

}

int main() {
    engine();
    return 0;
}