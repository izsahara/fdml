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

SIDGP config1(const TMatrix& X_train, const TMatrix& Y_train){
    Eigen::Index nftr = X_train.cols();
    // ======================= Layer 1  ======================= //
    shared_ptr<Kernel> kernel11 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);
    shared_ptr<Kernel> kernel12 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);

    Node node11(kernel11);
    Node node12(kernel12);

    node11.likelihood_variance.fix();
    node12.likelihood_variance.fix();

    std::vector<Node> nodes1{ node11, node12 };
    Layer layer1(nodes1);
    layer1.set_inputs(X_train);

    // ======================= Layer 2  ======================= //
    shared_ptr<Kernel> kernel21 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);
    shared_ptr<Kernel> kernel22 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);

    Node node21(kernel21);
    Node node22(kernel22);

    node21.likelihood_variance.fix();
    node22.likelihood_variance.fix();

    std::vector<Node> nodes2{ node21, node22 };
    Layer layer2(nodes2);

    // ======================= Layer 3  ======================= //
    shared_ptr<Kernel> kernel31 = make_shared<Matern52>(1.0, 1.0);
    Node node31(kernel31);
    node31.likelihood_variance.fix();

    std::vector<Node> nodes3{ node31 };
    Layer layer3(nodes3);
    layer3.set_outputs(Y_train);

    std::vector<Layer> layers{ layer1, layer2, layer3 };
    SIDGP model(layers);
    return model;
}

void plot(const TMatrix& X_plot, std::string& exp, SIDGP& model) {
    std::cout << "================ PLOT ================" << std::endl;
    MatrixPair Zplot = model.predict(X_plot, 100, 300);
    TMatrix Zpm = Zplot.first;
    TMatrix Zpv = Zplot.second;
    std::string Zpm_path = "../results/analytic2/" + exp + "PM.dat";
    std::string Zpv_path = "../results/analytic2/" + exp + "PV.dat";
    write_data(Zpm_path, Zpm);
    write_data(Zpv_path, Zpv);
}

void analytic2(const std::string& exp){
    TMatrix X_train = read_data("../datasets/analytic2/X_train.dat");
    TMatrix Y_train = read_data("../datasets/analytic2/Y_train.dat");
    TMatrix X_test = read_data("../datasets/analytic2/X_test.dat");
    TMatrix Y_test = read_data("../datasets/analytic2/Y_test.dat");    
    TMatrix X_plot = read_data("../datasets/analytic2/X_plot.dat");

    TMatrix Zmcs, Zvcs;
    while (true){
        SIDGP model = config1(X_train, Y_train);
        model.train(500, 100);        
        model.estimate();
        plot(X_plot, exp, model);
        std::cout << "================= MCS ================" << std::endl;
        MatrixPair Z = model.predict(X_test, Y_test, 75, 300);
        Zmcs = Z.first;
        Zvcs = Z.second;
        if (!(Zmcs.array().isNaN()).any()) {
            break;
        }
    }

    std::string Zmcs_path = "../results/analytic2/" + exp + "MCSM.dat";
    std::string Zvcs_path = "../results/analytic2/" + exp + "MCSV.dat";
    write_data(Zmcs_path, Zmcs);
    write_data(Zvcs_path, Zvcs);

    double nrmse = rmse(Y_test, Zmcs) / (Y_test.maxCoeff() - Y_test.minCoeff());
    std::cout << "NRMSE = " << nrmse << std::endl;    
}


int main(){
    for (unsigned int i = 14; i < 21; ++i){
        std::cout << "================= EXP " << i << " " << "================" << std::endl;
        analytic2(std::to_string(i));
    }
    return 0;
}