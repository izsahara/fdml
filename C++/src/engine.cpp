#include <fdml/utilities.h>
#include <fdml/kernels.h>
#include <fdml/base_models.h>
#include <fdml/deep_models.h>
#include <iostream>
#include <fstream>
#include <rapidcsv.h>

using namespace fdml::kernels;
using namespace fdml::base_models::gaussian_process;
using namespace fdml::deep_models::gaussian_process;
using fdml::utilities::metrics::rmse;
using std::cout;
using std::endl;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, "\t", "\n");
template <typename Derived>
void write_data(std::string name, const Eigen::MatrixBase<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

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
    
    // TMatrix X_train = read_data("/home/flowdiagnostics/GIT/fdml/datasets/engine/X_train.txt");
    // TMatrix Y_train = read_data("/home/flowdiagnostics/GIT/fdml/datasets/engine/Y_train.txt");
    // TMatrix X_test = read_data("/home/flowdiagnostics/GIT/fdml/datasets/engine/X_test.txt");
    // TMatrix Y_test = read_data("/home/flowdiagnostics/GIT/fdml/datasets/engine/Y_test.txt");    

    // Layer 1
    shared_ptr<Kernel> kernel11 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel12 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel13 = make_shared<Matern52>(1.0, 1.0);

    Node2 node11(kernel11);
    Node2 node12(kernel12);
    Node2 node13(kernel13);

    node11.likelihood_variance.fix();
    node12.likelihood_variance.fix();
    node13.likelihood_variance.fix();

    // node11.scale.fix();
    // node12.scale.fix();
    // node13.scale.fix();

    std::vector<Node2> nodes1{ node11, node12, node13 };
    Layer2 layer1(nodes1);
    layer1.set_inputs(X_train);

    // Layer 2
    shared_ptr<Kernel> kernel21 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel22 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel23 = make_shared<Matern52>(1.0, 1.0);
    Node2 node21(kernel21);
    Node2 node22(kernel22);
    Node2 node23(kernel23);

    node21.likelihood_variance.fix();
    node22.likelihood_variance.fix();
    node23.likelihood_variance.fix();

    // node21.scale.fix();
    // node22.scale.fix();
    // node23.scale.fix();

    std::vector<Node2> nodes2{ node21, node22, node23 };
    Layer2 layer2(nodes2);

    // Layer 3
    shared_ptr<Kernel> kernel31 = make_shared<Matern52>(1.0, 1.0);
    //kernel31->length_scale.set_bounds(VectorPair(TVector::Constant(1, 1.0), TVector::Constant(1, 13.0)));
    Node2 node31(kernel31);
    node31.likelihood_variance.fix();
    // node31.scale.fix();

    std::vector<Node2> nodes3{ node31 };
    Layer2 layer3(nodes3);
    layer3.set_outputs(Y_train);

    std::vector<Layer2> layers{ layer1, layer2, layer3 };
    SIDGP2 model(layers);
    model.train(500);
    model.estimate();

    MatrixPair Z = model.predict(X_test, Y_test, 100, 300);
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