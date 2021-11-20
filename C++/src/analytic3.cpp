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

void analytic3(std::string exp){
    TMatrix X_train = read_data("../datasets/analytic3/X_train.dat");
    TMatrix Y_train = read_data("../datasets/analytic3/Y_train.dat");
    TMatrix X_test = read_data("../datasets/analytic3/X_test.dat");
    TMatrix Y_test = read_data("../datasets/analytic3/Y_test.dat");    

    Eigen::Index nftr = X_train.cols();
    // ======================= Layer 1  ======================= //
    shared_ptr<Kernel> kernel11 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);
    shared_ptr<Kernel> kernel12 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);
    shared_ptr<Kernel> kernel13 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);

    Node2 node11(kernel11);
    Node2 node12(kernel12);
    Node2 node13(kernel13);

    node11.likelihood_variance.fix();
    node12.likelihood_variance.fix();
    node13.likelihood_variance.fix();

    std::vector<Node2> nodes1{ node11, node12, node13 };
    Layer2 layer1(nodes1);
    layer1.set_inputs(X_train);

    // ======================= Layer 2  ======================= //
    shared_ptr<Kernel> kernel21 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);
    shared_ptr<Kernel> kernel22 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);
    shared_ptr<Kernel> kernel23 = make_shared<Matern52>(TVector::Ones(nftr, 1.0), 1.0);

    Node2 node21(kernel21);
    Node2 node22(kernel22);
    Node2 node23(kernel23);

    node21.likelihood_variance.fix();
    node22.likelihood_variance.fix();
    node23.likelihood_variance.fix();

    std::vector<Node2> nodes2{ node21, node2, node23 };
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
    model.train(500, 100);
    model.estimate();

    // ======================= PREDICT  ======================= //
    
    std::cout << "================= MCS ================" << std::endl;
    MatrixPair Z = model.predict(X_test, Y_test, 100, 300);
    TMatrix Zmcs = Z.first;
    TMatrix Zvcs = Z.second;
    std::string Zmcs_path = "../results/analytic3/" + exp + "MCSM.dat";
    std::string Zvcs_path = "../results/analytic3/" + exp + "MCSV.dat";
    write_data(Zmcs_path, Zmcs);
    write_data(Zvcs_path, Zvcs);    
}


int main(){
    analytic3("1");
    return 0;
}