#include <fdml/utilities.h>
#include <fdml/kernels.h>
#include <fdml/base_models.h>
#include <fdml/deep_models.h>
#include <iostream>
#include <fstream>
#include <rapidcsv.h>
#include <filesystem>

using namespace fdml::kernels;
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

SIDGP config1(const TMatrix& X_train, const TMatrix& Y_train) {

    // ======================= Layer 1  ======================= //
    shared_ptr<Kernel> kernel11 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel12 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel13 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel14 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel15 = make_shared<Matern52>(1.0, 1.0);

    Node2 node11(kernel11);
    Node2 node12(kernel12);
    Node2 node13(kernel13);
    Node2 node14(kernel14);
    Node2 node15(kernel15);

    node11.likelihood_variance.fix();
    node12.likelihood_variance.fix();
    node13.likelihood_variance.fix();
    node14.likelihood_variance.fix();
    node15.likelihood_variance.fix();

    std::vector<Node2> nodes1{ node11, node12, node13, node14, node15 };
    Layer2 layer1(nodes1);
    layer1.set_inputs(X_train);

    // ======================= Layer 2  ======================= //
    shared_ptr<Kernel> kernel21 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel22 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel23 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel24 = make_shared<Matern52>(1.0, 1.0);
    shared_ptr<Kernel> kernel25 = make_shared<Matern52>(1.0, 1.0);

    Node2 node21(kernel21);
    Node2 node22(kernel22);
    Node2 node23(kernel23);
    Node2 node24(kernel24);
    Node2 node25(kernel25);

    node21.likelihood_variance.fix();
    node22.likelihood_variance.fix();
    node23.likelihood_variance.fix();
    node24.likelihood_variance.fix();
    node25.likelihood_variance.fix();

    std::vector<Node2> nodes2{ node21, node22, node23, node24, node25 };
    Layer2 layer2(nodes2);

    // ======================= Layer 3  ======================= //
    shared_ptr<Kernel> kernel31 = make_shared<Matern52>(1.0, 1.0);
    Node2 node31(kernel31);
    node31.likelihood_variance.fix();
    std::vector<Node2> nodes3{ node31 };
    Layer2 layer3(nodes3);
    layer3.set_outputs(Y_train);

    std::vector<Layer2> layers{ layer1, layer2, layer3 };
    SIDGP2 model(layers);
    return model;
}

void nrel(const TMatrix& X_train, const TMatrix& X_test, const std::string& config, 
          const TMatrix& Y_train, TMatrix& Y_test, const std::string& experiment) {

    std::string results_folder = "../results/nrel";
    if (!std::filesystem::exists(results_folder)) {
        std::filesystem::create_directory(results_folder);
    }
    std::string config_folder = results_folder + "/" + config;
    if (!std::filesystem::exists(config_folder)) {
        std::filesystem::create_directory(config_folder);
    }
    if (config == "1") {
        // SIDGP2 model = config1(X_train, Y_train);
        // model.train(500);
        // model.estimate();
        // MatrixPair Z = model.predict(X_test, Y_test, 100, 5);
        // TMatrix mu = Z.first;
        // TMatrix var = Z.second;

        // std::string mu_path = config_folder + "/" + experiment + "-M.dat";
        // std::string var_path = config_folder + "/" + experiment + "-V.dat";

        // write_data(mu_path, mu);
        // write_data(var_path, var);
    }

}

void RootMxc1(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-RootMxc1.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-RootMxc1.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "RootMxc1");

}

void RootMyc1(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-RootMyc1.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-RootMyc1.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "RootMyc1");
}

void TwrBsMxt(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-TwrBsMxt.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-TwrBsMxt.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "TwrBsMxt");

}

void TwrBsMyt(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-TwrBsMyt.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-TwrBsMyt.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "TwrBsMyt");

}

void Anch1Ten(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-Anch1Ten.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-Anch1Ten.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "Anch1Ten");

}

void Anch2Ten(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-Anch2Ten.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-Anch2Ten.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "Anch2Ten");

}

void Anch3Ten(const TMatrix& X_train, const TMatrix& X_test, const std::string& config){
    TMatrix Y_train = read_data("../datasets/nrel/250/TR-Anch3Ten.dat");
    TMatrix Y_test = read_data("../datasets/nrel/250/TS-Anch3Ten.dat");
    nrel(X_train, X_test, config, Y_train, Y_test, "Anch3Ten");

}


int main() {
    TMatrix X_train = read_data("../datasets/nrel/250/X_train.dat");
    TMatrix X_test = read_data("../datasets/nrel/250/X_test.dat");
    Anch1Ten(X_train, X_test, "1");
    return 0;
}