// E30
/* E30 = 1 + ( [25 * x1**2 * x2**2] - [3 * sqrt(5)) * (3 * length**3 + 5 * length * x1 * x2) * (x1 + x2)] + [15 * length**2 * (x1**2 + x2**2 + 3 * x1 * x2)] ) / (9 * length**4) 
// E301 = 25 * x1 **2 * x2 **2 
TMatrix E301 = 25.0 * op_sq_prod.array();
// E3021 =  (3 * length**3 + 5 * length * x1 * x2) = (5 * length * x1 * x2) + (3 * length**3)
TMatrix E3021 = (op_prod.array().rowwise() * (5.0 * tmp_ls.col(c).array()).matrix().array()).array().rowwise() + (3.0 * pow(tmp_ls.col(c).array(), 3)).matrix().array();				
// E3022 = 3 * sqrt(5) * (x1 + x2)
TMatrix E3022 = (3 * sqrt(5)) * op_sum.array();
// E302 = E3021 * E3022
TMatrix E302 = E3021.array() * E3022.array();
// E3031 = 15 * length**2
TRVector E3031 = 15.0 * square(tmp_ls.col(c).array());
// E3032 = (x1**2 + x2**2 + 3 * x1 * x2)
TMatrix E3032 = op_sq_sum.array() + (3.0 * op_prod.array());
// E303 = E3031 * E3032
TMatrix E303 = E3032.array().rowwise() * E3031.array();			
// E304 = (9 * length**4)
TRVector E304 = 9.0 * pow(tmp_ls.col(c).array(), 4);
// E305 = ([E301] - [E302] + [E303]) / E304
TMatrix E305 = (E301.array() - E302.array() + E303.array()).rowwise() / E304.array();
//TMatrix E30 = 1.0 + E305.array();
*/

// E31
/* E31= ( [18*sqrt(5)*length**3]  +  [15*sqrt(5)*length*(x1**2+x2**2)]  -  [(75*length**2+50*x1*x2)*(x1+x2)]  +  [60*sqrt(5)*length*x1*x2)]  /  (9*length**4)																
// E31_1 = 18*sqrt(5)*length**3
//TRVector E31_1 = 18.0 * sqrt(5.0) * pow(tmp_ls.col(c).array(), 3);
// E31_2 = 15 * sqrt(5) * length * (x1**2+x2**2)
//TMatrix E31_2 = op_sq_sum.array().rowwise() * (15.0 * sqrt(5.0) * tmp_ls.col(c).array()).matrix().array();
// E31_3 = (75 * length**2 + 50*x1*x2) * (x1+x2)
//TMatrix E31_3 = ((50.0 * op_prod.array()).rowwise() + (75.0 * square(tmp_ls.col(c).array())).matrix().array()).array() * op_sum.array();
// E31_4 = 60 * sqrt(5) * length * x1*x2
//TMatrix E31_4 = op_prod.array().rowwise() * (60.0 * sqrt(5.0) * tmp_ls.col(c).array()).matrix().array();
//TMatrix E3_3 = E3_31.array() * op_sum.array();
TMatrix E31 = ( (E31_2.array() - E31_3.array() + E31_4.array()).rowwise() + E31_1.array()).array().rowwise() / (9.0 * pow(tmp_ls.col(c).array(), 4)).matrix().array();
*/

// E32
/* E32 = 5 * ( [5*x1**2 + 5*x2**2 + 15*length**2] - [9*sqrt(5)*length*(x1+x2)]  +  [20*x1*x2]) / (9*length**4)				
// E32_1 = [5*x1**2 + 5*x2**2 + 15*length**2]
TMatrix E32_1 = (5.0 * op_sq_sum.array()).rowwise() + (15.0 * square(tmp_ls.col(c).array())).matrix().array();
// E32_2 = [9*sqrt(5)*length*(x1+x2)]
TMatrix E32_2 = op_sum.array().rowwise() * (9.0 * sqrt(5.0) * tmp_ls.col(c).array()).matrix().array();
// E32_3 = [20*x1*x2]
TMatrix E32_3 = 20.0 * op_prod.array();
*/

// E33 = 10 * (3 * sqrt(5) * length - 5 * x1 - 5 * x2) / (9 * length * *4)
// E34 = 25 / (9 * length**4)
// E3A31 = E30 + [muC * E31] + [(muC**2 + z_v) * E32] + [(muC**3 + 3 * z_v * muC) * E33] + (muC**4 + 6 * z_v * muC**2 + 3 * z_v**2) * E34
// E3A32 = E31 + [(muC + x2) * E32] + [(muC**2 + 2 * z_v + x2**2 + muC * x2) * E33] + [(muC**3 + x2**3 + x2 * muC**2 + muC * x2**2 + 3 * z_v * x2 + 5 * z_v * muC)] * E34


// P1
/*
// exp((10 * z_v + sqrt(5) * length * (x1 + x2 - 2 * z_m)) / length**2) * (0.5 * E3A31 * (1 + erf((muC - x2) / sqrt(2 * z_v))) + E3A32 * sqrt(0.5 * z_v / pi) * exp(-0.5 * (x2 - muC) * *2 / z_v))

// exp( (10 * z_v + sqrt(5) * length * (x1 + x2 - 2 * z_m)) / length * *2)
//TMatrix check1 = exp((10.0 * tmp_var(c) + (sqrt(5.0) * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / pow(tmp_ls(c), 2));
//exp((10.0 * tmp_var(c) + (sqrt(5.0) * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_var(c))))) / pow(tmp_ls(c), 2));
//TMatrix check2 = (0.5 * E3A31.array() * (1.0 + erf((tmp_muC(c) - XX.col(1).array()) / sqrt(2.0 * tmp_var(c)))));
//(0.5 * E3A31.array() * (1.0 + erf((tmp_muC(c) - XX.col(1).array()) / sqrt(2.0 * tmp_var(c)))));
//(E3A32.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_muC(c)) / tmp_var(c)));
//TMatrix check3 = (E3A32.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_muC(c)) / tmp_var(c)));

*/

// E40
/* 
* E40 = 1 + ( 25*x1**2*x2**2 + 3*sqrt(5)*(3*length**3-5*length*x1*x2)*(x2-x1) + 15*length**2*(x1**2+x2**2-3*x1*x2) ) / (9*length**4)
* [25*x1**2*x2**2] +
* [3*sqrt(5)*(3*length**3-5*length*x1*x2)*(x2-x1)] +
* [15*length**2*(x1**2+x2**2 - 3*x1*x2)]
*/
// E41
/*
* E41 = 5 * (3*sqrt(5)*length*(x2**2-x1**2) + 3*length**2*(x1+x2) - 10*x1*x2*(x1+x2)) / (9*length**4)
* [3*sqrt(5)*length*(x2**2-x1**2)] +
* [3*length**2*(x1+x2)] - 
* [10*x1*x2*(x1+x2)]
*/
// E42
/*
* E42 = 5 * (5*x1**2 + 5*x2**2 - 3*length**2 - 3*sqrt(5)*length*(x2-x1) + 20*x1*x2) / (9*length**4)
* [5*x1**2 + 5*x2**2] -
* [3*length**2] -
* [3*sqrt(5)*length*(x2-x1)] +
* [20*x1*x2]
*/
// E43 = -50 * (X1+X2) / (9*length**4)
// E44 = 25 / (9 * length * *4)


/*
*	E40 = 1 + ( 25*x1**2*x2**2 + 3*sqrt(5)*(3*length**3-5*length*x1*x2)*(x2-x1)  +  15*length**2*(x1**2+x2**2-3*x1*x2) ) / (9*length**4)
E50 = 1 + ( 25*x1**2*x2**2 + 3*sqrt(5)*(3*length**3+5*length*x1*x2)*(x1+x2)  +  15*length**2*(x1**2+x2**2+3*x1*x2) ) / (9*length**4)

E31 = (18*sqrt(5)*length**3 + 15*sqrt(5)*length*(x1**2 + x2**2) - (75*length**2 + 50*x1*x2)*(x1 + x2) + 60*sqrt(5)*length*x1*x2) / (9*length**4)
E51 = (18*sqrt(5)*length**3 + 15*sqrt(5)*length*(x1**2 + x2**2) + (75*length**2 + 50*x1*x2)*(x1 + x2) + 60*sqrt(5)*length*x1*x2) / (9*length**4)

E32 = 5*(5*x1**2 + 5*x2**2 + 15*length**2 - 9*sqrt(5)*length*(x1 + x2) + 20*x1*x2) / (9*length**4)
E52 = 5*(5*x1**2 + 5*x2**2 + 15*length**2 + 9*sqrt(5)*length*(x1 + x2) + 20*x1*x2) / (9*length**4)

E33 = 10*(3*sqrt(5)*length - 5*x1 - 5*x2) / (9*length**4)
E53 = 10*(3*sqrt(5)*length + 5*x1 + 5*x2) / (9*length**4)

E54=25/(9*length**4)						
*/

/*
E3A31 = E30 + muC*E31 + (muC**2 + z_v)*E32 + (muC**3 + 3*z_v*muC)*E33 + (muC**4 + 6*z_v*muC**2 + 3*z_v**2)*E34
E5A51 = E50 - muD*E51 + (muD**2 + z_v)*E52 - (muD**3 + 3*z_v*muD)*E53 + (muD**4 + 6*z_v*muD**2 + 3*z_v**2)*E54

E5A52 = E51 - (muD + x1)*E52 + (muD**2 + 2*z_v + x1**2 + muD*x1)*E53 - (muD**3 + x1**3 + x1*muD**2 + muD*x1**2 + 3*z_v*x1 + 5*z_v*muD)*E54
E3A32 = E31 + (muC + x2)*E32 + (muC**2 + 2*z_v + x2**2 + muC*x2)*E33 + (muC**3 + x2**3 + x2*muC**2 + muC*x2**2 + 3*z_v*x2 + 5*z_v*muC)*E34

P1 = exp((10*z_v + sqrt(5)*length*(x1 + x2 - 2*z_m))/length**2) * (0.5*E3A31 * (1 + erf( (muC- x2) / sqrt(2*z_v))) + E3A32*sqrt(0.5*z_v/pi)*exp(-0.5*(x2-muC)**2/z_v))
P3 = exp((10*z_v - sqrt(5)*length*(x1 + x2 - 2*z_m))/length**2) * (0.5*E5A51 * (1 + erf( (x1 - muD)/ sqrt(2*z_v))) + E5A52*sqrt(0.5*z_v/pi)*exp(-0.5*(x1-muD)**2/z_v))				
*/


#ifdef OLD_CODE
if (!(variance_.array() == 0.0).any()) {
	TMatrix mt1 = ((-(Xz.array().rowwise() * (2.0 * sqrt(5.0) * ls.array()).matrix().array()).array()).array().rowwise() +
		(5.0 * var.array()).matrix().array());
	TMatrix pt1 = (((Xz.array().rowwise() * (2.0 * sqrt(5.0) * ls.array()).matrix().array()).array()).array().rowwise() +
		(5.0 * var.array()).matrix().array());

	// t1
	/* 	
	TMatrix pmuA = pnorm((muA.array().rowwise() / sqrt(var.array())).matrix());
	term10 = np.exp( (5 * z_v[i] - 2 * sqrt(5) * length[i] * zX[i]) / (2 * length[i] ** 2))
	term11 = (1 + sqrt(5) * muA[i] / length[i] + 5 * (muA[i] **2 + z_v[i]) / (3 * length[i] * *2))
	term12 = pnorm(muA[i] / sqrt(z_v[i]))
	term13 = sqrt(5) + (5 * muA[i]) / (3 * length[i])
	term14 = sqrt(0.5 * z_v[i] / pi) / length[i] * np.exp(-0.5 * muA[i] * *2 / z_v[i])

	TMatrix term10 = exp(mt1.array().rowwise() / (2 * square(ls.array())).matrix().array());
	TMatrix term11 = (1.0 + ((sqrt(5.0) * muA.array()).rowwise() / ls.array()).array()) + (5.0 * ((square(muA.array()).rowwise() + var.array()).rowwise() / (3 * square(ls.array()))).array());
	TMatrix term12 = pnorm((muA.array().rowwise() / sqrt(var.array())).matrix());
	TMatrix term13 = sqrt(5.0) + ((5.0 * muA.array()).rowwise() / (3.0 * ls.array())).array();
	TMatrix term14 = (exp((-0.5 * square(muA.array())).rowwise() / var.array())).array().rowwise() * static_cast<TRVector>(sqrt(0.5 * (var.array() / PI)) / ls.array()).array();
	*/

	TMatrix t1 = exp(mt1.array().rowwise() / (2 * square(ls.array())).matrix().array()).array() * 
		((((1.0 + ((sqrt(5.0) * muA.array()).rowwise() / ls.array()).array()) +
		(5.0 * ((square(muA.array()).rowwise() + var.array()).rowwise() / (3 * square(ls.array()))).array())).array() *
		(pnorm((muA.array().rowwise() / sqrt(var.array())).matrix())).array()) +
		((sqrt(5.0) + ((5.0 * muA.array()).rowwise() / (3.0 * ls.array())).array()) *
		((exp((-0.5 * square(muA.array())).rowwise() / var.array())).array().rowwise() * 
		(sqrt(0.5 * (var.array() / PI)) / ls.array()).matrix().array())));

	// t2 
	/*
	TMatrix pmuB = pnorm(((-muB.array()).rowwise() / sqrt(var.array())).matrix());
	term21 = np.exp((5 * z_v[i] + 2 * sqrt(5) * length[i] * zX[i]) / (2 * length[i] ** 2))
	term221 = (1 - sqrt(5) * muB[i] / length[i] + 5 * (muB[i] ** 2 + z_v[i]) / (3 * length[i] ** 2))
	term222 = pnorm(-muB[i] / sqrt(z_v[i]))
	term223 = (sqrt(5) - (5 * muB[i]) / (3 * length[i]))
	term224 = sqrt(0.5 * z_v[i] / pi) / length[i] * np.exp(-0.5 * muB[i] ** 2 / z_v[i])		

	TMatrix term21 = exp(pt1.array().rowwise() / (2 * square(ls.array())).matrix().array());
	TMatrix term221 = (1.0 - ((sqrt(5.0) * muB.array()).rowwise() / ls.array()).array()) + (5.0 * ((square(muB.array()).rowwise() + var.array()).rowwise() / (3 * square(ls.array()))).array());
	TMatrix term222 = pnorm(((-muB.array()).rowwise() / sqrt(var.array())).matrix());
	TMatrix term223 = sqrt(5.0) - ((5.0 * muB.array()).rowwise() / (3.0 * ls.array())).array();
	TMatrix term224 = (exp((-0.5 * square(muB.array())).rowwise() / var.array())).array().rowwise() * static_cast<TRVector>(sqrt(0.5 * (var.array() / PI)) / ls.array()).array();
	TMatrix term2 = term31.array() *  ((term321.array() * term322.array()) + (term323.array() * term324.array()));
	*/

	TMatrix t2 = exp(pt1.array().rowwise() / (2 * square(ls.array())).matrix().array()) * (
		(((1.0 - ((sqrt(5.0) * muB.array()).rowwise() / ls.array()).array()) + 
		(5.0 * ((square(muB.array()).rowwise() + var.array()).rowwise() / (3 * square(ls.array()))).array())).array() *
		pnorm(((-muB.array()).rowwise() / sqrt(var.array())).matrix()).array()) +
		((sqrt(5.0) - ((5.0 * muB.array()).rowwise() / (3.0 * ls.array())).array()).array() *
		((exp((-0.5 * square(muB.array())).rowwise() / var.array())).array().rowwise() * 
		(sqrt(0.5 * (var.array() / PI)) / ls.array()).matrix().array()).array()));

	I = (t1 + t2).rowwise().prod();			
}
#endif

#ifdef DEBUG_MATERN52_J

std::tuple<TMatrix, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix>
	debug_J(const TVector& mean, const TVector& variance_, const TMatrix& X) 
{
	std::tuple<TMatrix, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix, TMatrix> Xout;
	//std::tuple<TMatrix, TMatrix, TMatrix> Xout;
	TMatrix J = TMatrix::Ones(X.rows(), X.rows());
	// To avoid repetitive transpose, create tmp (RowMajor) length_scale and variance_
	TRVector ls = static_cast<TRVector>(length_scale.value());
	TRVector mu = static_cast<TRVector>(mean);
	TRVector var = static_cast<TRVector>(variance_);

	// Find all variances that are zero
	std::vector<Eigen::Index> indices(variance_.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::vector<Eigen::Index> zero_indices;
	std::vector<Eigen::Index> non_zero_indices;
	for (Eigen::Index i = 0; i < variance_.size(); ++i)
	{
		if (variance_[i] == 0.0) { zero_indices.push_back(i); }
	}
	std::set_difference(indices.begin(), indices.end(), zero_indices.begin(), zero_indices.end(),
		std::inserter(non_zero_indices, non_zero_indices.begin()));

	TRVector muC = mu.array() - ( (2.0 * sqrt(5) * var.array()) / ls.array());
	TRVector muD = mu.array() + ( (2.0 * sqrt(5) * var.array()) / ls.array());

	auto non_zero_variance = [&J, &X, &muC, &muD, &ls, &mu, &var, &non_zero_indices, &Xout]()
	{

		// Mask with non zero variance indicies
		TMatrix  tmp_X   = X(Eigen::all, non_zero_indices);
		TRVector tmp_ls  = ls(non_zero_indices);
		TRVector tmp_mu  = mu(non_zero_indices);
		TRVector tmp_var = var(non_zero_indices);
		TRVector tmp_muC = muC(non_zero_indices);				
		TRVector tmp_muD = muD(non_zero_indices);				
		// E3A31 Coefficients
		// [(muC**2 + z_v) * E32] 
		TRVector CE32 = (square(tmp_muC.array()) + tmp_var.array());
		// (muC**3 + 3*z_v*muC)
		TRVector CE33 = pow(tmp_muC.array(), 3) + (3.0 * tmp_var.array() * tmp_muC.array());
		// (muC**4 + 6*z_v*muC**2 + 3*z_v**2)
		TRVector CE34 = pow(tmp_muC.array(), 4) + (6.0 * tmp_var.array() * square(tmp_muC.array())) + (3.0 * square(tmp_var.array()));
		// E5A51 Coefficients
		// [(muD**2 + z_v) * E52]
		TRVector DE52 = (square(tmp_muD.array()) + tmp_var.array());
		// [(muD**3 + 3*z_v*muD) * E53]
		TRVector DE53 = pow(tmp_muD.array(), 3) + (3.0 * tmp_var.array() * tmp_muD.array());
		// [(muD**4 + 6*z_v*muD**2 + 3*z_v**2) * E54]
		TRVector DE54 = pow(tmp_muD.array(), 4) + (6.0 * tmp_var.array() * square(tmp_muD.array())) + (3.0 * square(tmp_var.array()));

		// ========================== 1 loop ========================== //
		Eigen::Index c = 1;
		TMatrix XX(int(pow(tmp_X.rows(), 2)), 2);
		TMatrix X1 = tmp_X(Eigen::all, c).transpose().replicate(tmp_X.rows(), 1);
		TMatrix X2 = tmp_X(Eigen::all, c).replicate(1, tmp_X.rows());
		TVector V1 = Eigen::Map<TVector>(X1.data(), X1.size());
		TVector V2 = Eigen::Map<TVector>(X2.data(), X2.size());
		XX << (V1.array() > V2.array()).select(V2, V1), (V1.array() > V2.array()).select(V1, V2);

		// Define Repetitive operations
		// (x1 * x2)
		TMatrix op_prod = XX.rowwise().prod();
		// (x1 + x2)
		TMatrix op_sum = XX.rowwise().sum();
		// (x1**2 + x2**2)
		TMatrix op_sq_sum = square(XX.array()).rowwise().sum();
		// (x1**2 * x2**2)
		TMatrix op_sq_prod = square(XX.array()).rowwise().prod();
		// Define repetitive terms
		double ls_sqrd = pow(tmp_ls(c), 2);
		double denominator = (9.0 * pow(tmp_ls(c), 4));
		double sqrtf = sqrt(5.0);
		double EX4 = 25.0 / (denominator);

		TMatrix E30   = 1.0 + (((25.0 * op_sq_prod.array()) -								
						(((op_prod.array() * (5.0 * tmp_ls(c))) + (3.0 * pow(tmp_ls(c), 3))) * ((3.0 * sqrtf) * op_sum.array())).array() +
						((op_sq_sum.array() + (3.0 * op_prod.array())) * (15.0 * ls_sqrd))) / (denominator));
		TMatrix E31   = (((op_sq_sum.array() * (15.0 * sqrtf * tmp_ls(c))) -								
						(((50.0 * op_prod.array()) + (75.0 * ls_sqrd)) * op_sum.array()) +
						(op_prod.array() * (60.0 * sqrtf * tmp_ls(c)))) +
						(18.0 * sqrtf * pow(tmp_ls(c), 3))) / (denominator);
		TMatrix E32   = (5.0 * (((5.0 * op_sq_sum.array()) + (15.0 * ls_sqrd)) - (op_sum.array() * (9.0 * sqrtf * tmp_ls(c))) +								
						(20.0 * op_prod.array()))) / (denominator);			
		TMatrix E33   =	(10.0 *  ((- 5.0 * op_sum.array()) + (3.0 * sqrtf * tmp_ls(c)))) / (denominator);								
		TMatrix E3A31 = E30.array() + (tmp_muC(c) * E31.array()) + (CE32(c) * E32.array()) + (CE33(c) * E33.array()) + (CE34(c) * EX4);				
		TMatrix E3A32 = E31.array() + (tmp_muC(c) + XX.col(1).array()) * E32.array() +								
						(pow(tmp_muC(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(1).array()) + (tmp_muC(c) * XX.col(1).array())) * E33.array() +
						(pow(tmp_muC(c), 3) + pow(XX.col(1).array(), 3) + (pow(tmp_muC(c), 2) * XX.col(1).array()) + (tmp_muC(c) * square(XX.col(1).array())) +
						(3.0 * tmp_var(c) * XX.col(1).array()) + (5.0 * tmp_var(c) * tmp_muC(c))) * EX4;

		TMatrix P1	  = (exp((10.0 * tmp_var(c) + (sqrtf * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / ls_sqrd)) *								
						((0.5 * E3A31.array() * (1.0 + erf((tmp_muC(c) - XX.col(1).array()) / sqrt(2.0 * tmp_var(c))))) +
						(E3A32.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_muC(c)) / tmp_var(c))));
		
		TMatrix E40	  = 1.0 + (((25.0 * op_sq_prod.array()) + 
						(3.0 * sqrtf * ((3.0 * pow(tmp_ls(c), 3)) - (5.0 * tmp_ls(c) * op_prod.array())) * (XX.col(1).array() - XX.col(0).array())) +
						(15.0 * ls_sqrd * (op_sq_sum.array() - (3.0 * op_prod.array())))) / (denominator));
		TMatrix E41   = 5.0 * ((3.0 * sqrtf * tmp_ls(c) * (square(XX.col(1).array()) - square(XX.col(0).array()))) +
						(3.0 * ls_sqrd * op_sum.array()) - (10.0 * op_prod.array() * op_sum.array())) / (denominator);
		TMatrix E42   = 5.0 * ((5.0 * op_sq_sum.array()) - (3.0 * ls_sqrd) -  (3.0 * sqrtf * tmp_ls(c) * (XX.col(1).array() - XX.col(0).array())) +
						(20.0 * op_prod.array())) / (denominator);		
		TMatrix E43	  = -50.0 * (V1.array() + V2.array()) / (denominator);
		TMatrix E4A41 = E40.array() +
						(tmp_mu(c) * E41.array()) + ((pow(tmp_mu(c), 2) + tmp_var(c)) * E42.array()) +
						((pow(tmp_mu(c), 3) + 3.0 * tmp_var(c) * tmp_mu(c)) * E43.array()) +
						((pow(tmp_mu(c), 4) + 6.0 * tmp_var(c) * pow(tmp_mu(c), 2) + 3.0 * pow(tmp_var(c), 2)) * EX4);				
		TMatrix E4A42 = E41.array() +
						((tmp_mu(c) + XX.col(0).array()) * E42.array()) +
						((pow(tmp_mu(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(0).array()) + (tmp_mu(c) * XX.col(0).array())) * E43.array()) +
						((pow(tmp_mu(c), 3) + pow(XX.col(0).array(), 3) + (pow(tmp_mu(c), 2) * XX.col(0).array()) + (tmp_mu(c) * square(XX.col(0).array())) +
						(3.0 * tmp_var(c) * XX.col(0).array()) + (5.0 * tmp_mu(c) * tmp_var(c))) * EX4);
		TMatrix E4A43 = E41.array() +
						((tmp_mu(c) + XX.col(1).array()) * E42.array()) +
						((pow(tmp_mu(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(1).array()) + (tmp_mu(c) * XX.col(1).array())) * E43.array()) +
						((pow(tmp_mu(c), 3) + pow(XX.col(1).array(), 3) + (pow(tmp_mu(c), 2) * XX.col(1).array()) + (tmp_mu(c) * square(XX.col(1).array())) +
						(3.0 * tmp_var(c) * XX.col(1).array()) + (5.0 * tmp_mu(c) * tmp_var(c))) * EX4);

		TMatrix P2    = exp(-sqrtf * (XX.col(1).array() - XX.col(0).array()) / tmp_ls(c)) * 					
						((0.5 * E4A41.array() * (erf((XX.col(1).array() - tmp_mu(c)) / (sqrt(2.0 * tmp_var(c)))) -
						erf((XX.col(0).array() - tmp_mu(c)) / (sqrt(2.0 * tmp_var(c)))))) +
						(E4A42.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(0).array() - tmp_mu(c)) / tmp_var(c))) -
						(E4A43.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_mu(c)) / tmp_var(c))));

		TMatrix E50	  = 1.0 + (((25.0 * op_sq_prod.array()) +
						(3.0 * sqrtf * ((3.0 * pow(tmp_ls(c), 3)) + (5.0 * tmp_ls(c) * op_prod.array())) * op_sum.array()) +
						(15.0 * ls_sqrd * (op_sq_sum.array() + (3.0 * op_prod.array())))) / (denominator));
		TMatrix E51   = (((op_sq_sum.array() * (15.0 * sqrtf * tmp_ls(c))) +
						(((50.0 * op_prod.array()) + (75.0 * ls_sqrd)) * op_sum.array()) +
						(op_prod.array() * (60.0 * sqrtf * tmp_ls(c)))) +
						(18.0 * sqrtf * pow(tmp_ls(c), 3))) / (denominator);
		TMatrix E52   = (5.0 * (((5.0 * op_sq_sum.array()) + (15.0 * ls_sqrd)) + (op_sum.array() * (9.0 * sqrtf * tmp_ls(c))) + (20.0 * op_prod.array()))) / (denominator);
		TMatrix E53   = (10.0 * ((5.0 * op_sum.array()) + (3.0 * sqrtf * tmp_ls(c)))) / (denominator);
		TMatrix E5A51 = E50.array() - (tmp_muD(c) * E51.array()) + (DE52(c) * E52.array()) - (DE53(c) * E53.array()) + (DE54(c) * EX4);
		TMatrix E5A52 = E51.array() - (tmp_muD(c) + XX.col(0).array()) * E52.array() +
						(pow(tmp_muD(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(0).array()) + (tmp_muD(c) * XX.col(0).array())) * E53.array() -
						(pow(tmp_muD(c), 3) + pow(XX.col(0).array(), 3) + (pow(tmp_muD(c), 2) * XX.col(0).array()) + (tmp_muD(c) * square(XX.col(0).array())) +
						(3.0 * tmp_var(c) * XX.col(0).array()) + (5.0 * tmp_var(c) * tmp_muD(c))) * EX4;
		TMatrix P3	  = (exp((10.0 * tmp_var(c) - (sqrtf * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / ls_sqrd)) *
						((0.5 * E5A51.array() * (1.0 + erf((XX.col(0).array() - tmp_muD(c)) / sqrt(2.0 * tmp_var(c))))) +
						(E5A52.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(0).array() - tmp_muD(c)) / tmp_var(c))));
		TMatrix JD = P1.array() + P2.array() + P3.array();
		// ========================== 1 loop ========================== //

		// Final
		/*			
			// Swap X1, X2; if X1<X2: {x1=X1; x2=X2} else: {x1=X2; x2=X1}
			//TMatrix XX(int(pow(tmp_X.rows(), 2)), 2);
			//for (Eigen::Index c = 0; c < tmp_X.cols(); ++c) {
			//	TMatrix X1 = tmp_X(Eigen::all, c).transpose().replicate(tmp_X.rows(), 1);
			//	TMatrix X2 = tmp_X(Eigen::all, c).replicate(1, tmp_X.rows());
			//	TVector V1 = Eigen::Map<TVector>(X1.data(), X1.size());
			//	TVector V2 = Eigen::Map<TVector>(X2.data(), X2.size());
			//	XX << (V1.array() > V2.array()).select(V2, V1), (V1.array() > V2.array()).select(V1, V2);
			//	
			//	// Define Repetitive operations
			//	// (x1 * x2)
			//	TMatrix op_prod = XX.rowwise().prod();
			//	// (x1**2 * x2**2)
			//	TMatrix op_sq_prod = square(XX.array()).rowwise().prod();
			//	// (x1 + x2)
			//	TMatrix op_sum = XX.rowwise().sum();
			//	// E30 = 1 + (25 * x1**2 * x2**2 - 3 * sqrt(5) * (3 * length**3 + 5 * length * x1 * x2) * (x1 + x2) + 15 * length**2 * (x1**2 + x2**2 + 3 * x1 * x2)) / (9 * length**4)
			//	// E301 = 25 * x1 **2 * x2 **2 - 3 * sqrt(5)
			//	// E302 = (3 * length**3 + 5 * length * x1 * x2) = (5 * length * x1 * x2) + (3 * length**3)
			//	// E303 = 
			//	TMatrix E301 = (25.0 * op_sq_prod.array()) - 3.0 * sqrt(5.0);
			//	TMatrix E302 = (op_prod.array().rowwise() * (5.0 * tmp_ls.array()).matrix().array()).array().rowwise() + (3.0 * pow(ls.array(), 3)).matrix().array();
			//}
		*/
				
		//Xout = std::make_tuple(E30, E31, E32, E33, EX4, E3A31, E3A32, P1);
		//Xout = std::make_tuple(E40, E41, E42, E43, EX4, E4A41, E4A42, E4A43, P2);
		Xout = std::make_tuple(E50, E51, E52, E53, E5A51, E5A52, P3, JD);
		//Xout = std::make_tuple(check1, check2 , check3);
	};

	//if (non_zero_indices.size() > 0) { TMatrix XX = non_zero_variance(); }
	non_zero_variance();
	return Xout;

}
#endif


		
MatrixPair Matern52_debug_IJ(const TVector& mean, const TVector& variance_, const TMatrix& X, const Eigen::Index& idx) {
	TMatrix I  = TMatrix::Ones(X.rows(), 1);
	TMatrix J  = TMatrix::Ones(X.rows(), X.rows());
	

	// To avoid repetitive transpose, create tmp (RowMajor) length_scale and variance_
	TRVector ls  = static_cast<TRVector>(length_scale.value());
	TRVector mu  = static_cast<TRVector>(mean);
	TRVector var = static_cast<TRVector>(variance_);

	// Find all variances that are zero
	std::vector<Eigen::Index> indices(variance_.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::vector<Eigen::Index> zero_indices;
	std::vector<Eigen::Index> non_zero_indices;
	for (Eigen::Index i = 0; i < variance_.size(); ++i) 
	{if (variance_[i] == 0.0) { zero_indices.push_back(i); }}
	std::set_difference(indices.begin(), indices.end(), zero_indices.begin(), zero_indices.end(),
		std::inserter(non_zero_indices, non_zero_indices.begin()));

	TMatrix  Xz  = (-(X.array().rowwise() - mean.transpose().array()));
	TMatrix  muA = Xz.array().rowwise() - (sqrt(5) * var.array()) / ls.array() ;
	TMatrix  muB = (Xz.array().rowwise() + (sqrt(5) * var.array()) / ls.array());
	TRVector muC = mu.array() - ((2.0 * sqrt(5) * var.array()) / ls.array());
	TRVector muD = mu.array() + ((2.0 * sqrt(5) * var.array()) / ls.array());

	auto zero_variance = [&I, &J, &Xz, &ls, &zero_indices]()
	{
		TMatrix  tmp_Xz = Xz(Eigen::all, zero_indices);
		TRVector tmp_ls = ls(zero_indices);
		// Id = (1 + sqrt(5)*np.abs(zX[i])/length[i] + 5*zX[i]**2/(3*length[i]**2)) * np.exp(-sqrt(5)*np.abs(zX[i])/length[i])
		TMatrix Id = (1.0 + ((sqrt(5.0) * abs(tmp_Xz.array())).array().rowwise() / tmp_ls.array()) +
					 (5.0 * square(tmp_Xz.array())).rowwise() / (3.0 * square(tmp_ls.array()))) *
					 (exp((-sqrt(5.0) * abs(tmp_Xz.array()).array()).rowwise() / tmp_ls.array()));
		I.array() *= (Id.rowwise().prod()).array();
		J.array() *= (I * I.transpose()).array();
	};

	auto non_zero_variance = [&I, &J, &X, &mean, &ls, &var, &muA, &muB, &muC, &muD, &non_zero_indices]()
	{
		TVector J0 = TVector::Ones(X.rows() * X.rows());
		// Mask with non zero variance indicies
		TMatrix  tmp_X   = X(Eigen::all, non_zero_indices);
		TRVector tmp_ls  = ls(non_zero_indices);
		TVector  tmp_mu  = mean(non_zero_indices);
		TRVector tmp_var = var(non_zero_indices);
		TMatrix  tmp_muA = muA(Eigen::all, non_zero_indices);
		TMatrix  tmp_muB = muB(Eigen::all, non_zero_indices);
		TRVector tmp_muC = muC(non_zero_indices);
		TRVector tmp_muD = muD(non_zero_indices);

		/* ================================ COMPUTE I  ================================ */
		TMatrix Xz  = (-(tmp_X.array().rowwise() - tmp_mu.transpose().array()));
		TMatrix mt1 = ((-(Xz.array().rowwise() * (2.0 * sqrt(5.0) * tmp_ls.array()).matrix().array()).array()).array().rowwise() +
					  (5.0 * tmp_var.array()).matrix().array());
		TMatrix pt1 = (((Xz.array().rowwise() * (2.0 * sqrt(5.0) * tmp_ls.array()).matrix().array()).array()).array().rowwise() +
					  (5.0 * tmp_var.array()).matrix().array());

		TMatrix t1  = exp(mt1.array().rowwise() / (2 * square(tmp_ls.array())).matrix().array()).array() *
					  ((((1.0 + ((sqrt(5.0) * tmp_muA.array()).rowwise() / tmp_ls.array()).array()) +
					  (5.0 * ((square(tmp_muA.array()).rowwise() + tmp_var.array()).rowwise() / (3 * square(tmp_ls.array()))).array())).array() *
					  (pnorm((tmp_muA.array().rowwise() / sqrt(tmp_var.array())).matrix())).array()) +
					  ((sqrt(5.0) + ((5.0 * tmp_muA.array()).rowwise() / (3.0 * tmp_ls.array())).array()) *
					  ((exp((-0.5 * square(tmp_muA.array())).rowwise() / tmp_var.array())).array().rowwise() *
					  (sqrt(0.5 * (tmp_var.array() / PI)) / tmp_ls.array()).matrix().array())));

		TMatrix t2  = exp(pt1.array().rowwise() / (2 * square(tmp_ls.array())).matrix().array()) * (
					  (((1.0 - ((sqrt(5.0) * tmp_muB.array()).rowwise() / tmp_ls.array()).array()) +
					  (5.0 * ((square(tmp_muB.array()).rowwise() + tmp_var.array()).rowwise() / (3 * square(tmp_ls.array()))).array())).array() *
					  pnorm(((-tmp_muB.array()).rowwise() / sqrt(tmp_var.array())).matrix()).array()) +
					  ((sqrt(5.0) - ((5.0 * tmp_muB.array()).rowwise() / (3.0 * tmp_ls.array())).array()).array() *
					  ((exp((-0.5 * square(tmp_muB.array())).rowwise() / tmp_var.array())).array().rowwise() *
					  (sqrt(0.5 * (tmp_var.array() / PI)) / tmp_ls.array()).matrix().array()).array()));
		
		I.array() *= ((t1 + t2).rowwise().prod()).array();

		/* ================================ COMPUTE J  ================================ */
		// E3A31 Coefficients
		// [(muC**2 + z_v) * E32] 
		TRVector CE32 = (square(tmp_muC.array()) + tmp_var.array());
		// (muC**3 + 3*z_v*muC)
		TRVector CE33 = pow(tmp_muC.array(), 3) + (3.0 * tmp_var.array() * tmp_muC.array());
		// (muC**4 + 6*z_v*muC**2 + 3*z_v**2)
		TRVector CE34 = pow(tmp_muC.array(), 4) + (6.0 * tmp_var.array() * square(tmp_muC.array())) + (3.0 * square(tmp_var.array()));
		// E5A51 Coefficients
		// [(muD**2 + z_v) * E52]
		TRVector DE52 = (square(tmp_muD.array()) + tmp_var.array());
		// [(muD**3 + 3*z_v*muD) * E53]
		TRVector DE53 = pow(tmp_muD.array(), 3) + (3.0 * tmp_var.array() * tmp_muD.array());
		// [(muD**4 + 6*z_v*muD**2 + 3*z_v**2) * E54]
		TRVector DE54 = pow(tmp_muD.array(), 4) + (6.0 * tmp_var.array() * square(tmp_muD.array())) + (3.0 * square(tmp_var.array()));
		// Loop over each column
		for (Eigen::Index c = 0; c < tmp_X.cols(); ++c) {
			TMatrix XX(int(pow(tmp_X.rows(), 2)), 2);
			TMatrix X1 = tmp_X(Eigen::all, c).transpose().replicate(tmp_X.rows(), 1);
			TMatrix X2 = tmp_X(Eigen::all, c).replicate(1, tmp_X.rows());
			TVector V1 = Eigen::Map<TVector>(X1.data(), X1.size());
			TVector V2 = Eigen::Map<TVector>(X2.data(), X2.size());
			XX << (V1.array() > V2.array()).select(V2, V1), (V1.array() > V2.array()).select(V1, V2);
			// Define Repetitive operations
			// (x1 * x2)
			TMatrix op_prod = XX.rowwise().prod();
			// (x1 + x2)
			TMatrix op_sum = XX.rowwise().sum();
			// (x1**2 + x2**2)
			TMatrix op_sq_sum = square(XX.array()).rowwise().sum();
			// (x1**2 * x2**2)
			TMatrix op_sq_prod = square(XX.array()).rowwise().prod();
			// Define repetitive terms
			double ls_sqrd = pow(tmp_ls(c), 2);
			double denominator = (9.0 * pow(tmp_ls(c), 4));
			double sqrtf = sqrt(5.0);
			double EX4 = 25.0 / (denominator);
			/* ================================ COMPUTE E3  ================================ */
			TMatrix E30   = 1.0 + (((25.0 * op_sq_prod.array()) -								
							(((op_prod.array() * (5.0 * tmp_ls(c))) + (3.0 * pow(tmp_ls(c), 3))) * ((3.0 * sqrtf) * op_sum.array())).array() +
							((op_sq_sum.array() + (3.0 * op_prod.array())) * (15.0 * ls_sqrd))) / (denominator));
			TMatrix E31   = (((op_sq_sum.array() * (15.0 * sqrtf * tmp_ls(c))) -								
							(((50.0 * op_prod.array()) + (75.0 * ls_sqrd)) * op_sum.array()) +
							(op_prod.array() * (60.0 * sqrtf * tmp_ls(c)))) +
							(18.0 * sqrtf * pow(tmp_ls(c), 3))) / (denominator);
			TMatrix E32   = (5.0 * (((5.0 * op_sq_sum.array()) + (15.0 * ls_sqrd)) - (op_sum.array() * (9.0 * sqrtf * tmp_ls(c))) +								
							(20.0 * op_prod.array()))) / (denominator);			
			TMatrix E33   =	(10.0 *  ((- 5.0 * op_sum.array()) + (3.0 * sqrtf * tmp_ls(c)))) / (denominator);								
			TMatrix E3A31 = E30.array() + (tmp_muC(c) * E31.array()) + (CE32(c) * E32.array()) + (CE33(c) * E33.array()) + (CE34(c) * EX4);				
			TMatrix E3A32 = E31.array() + (tmp_muC(c) + XX.col(1).array()) * E32.array() +								
							(pow(tmp_muC(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(1).array()) + (tmp_muC(c) * XX.col(1).array())) * E33.array() +
							(pow(tmp_muC(c), 3) + pow(XX.col(1).array(), 3) + (pow(tmp_muC(c), 2) * XX.col(1).array()) + (tmp_muC(c) * square(XX.col(1).array())) +
							(3.0 * tmp_var(c) * XX.col(1).array()) + (5.0 * tmp_var(c) * tmp_muC(c))) * EX4;

			TMatrix P1	  = (exp((10.0 * tmp_var(c) + (sqrtf * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / ls_sqrd)) *								
							((0.5 * E3A31.array() * (1.0 + erf((tmp_muC(c) - XX.col(1).array()) / sqrt(2.0 * tmp_var(c))))) +
							(E3A32.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_muC(c)) / tmp_var(c))));
			/* ================================ COMPUTE E4  ================================ */
			TMatrix E40	  = 1.0 + (((25.0 * op_sq_prod.array()) + 
							(3.0 * sqrtf * ((3.0 * pow(tmp_ls(c), 3)) - (5.0 * tmp_ls(c) * op_prod.array())) * (XX.col(1).array() - XX.col(0).array())) +
							(15.0 * ls_sqrd * (op_sq_sum.array() - (3.0 * op_prod.array())))) / (denominator));
			TMatrix E41   = 5.0 * ((3.0 * sqrtf * tmp_ls(c) * (square(XX.col(1).array()) - square(XX.col(0).array()))) +
							(3.0 * ls_sqrd * op_sum.array()) - (10.0 * op_prod.array() * op_sum.array())) / (denominator);
			TMatrix E42   = 5.0 * ((5.0 * op_sq_sum.array()) - (3.0 * ls_sqrd) -  (3.0 * sqrtf * tmp_ls(c) * (XX.col(1).array() - XX.col(0).array())) +
							(20.0 * op_prod.array())) / (denominator);		
			TMatrix E43	  = -50.0 * (V1.array() + V2.array()) / (denominator);
			TMatrix E4A41 = E40.array() +
							(tmp_mu(c) * E41.array()) + ((pow(tmp_mu(c), 2) + tmp_var(c)) * E42.array()) +
							((pow(tmp_mu(c), 3) + 3.0 * tmp_var(c) * tmp_mu(c)) * E43.array()) +
							((pow(tmp_mu(c), 4) + 6.0 * tmp_var(c) * pow(tmp_mu(c), 2) + 3.0 * pow(tmp_var(c), 2)) * EX4);				
			TMatrix E4A42 = E41.array() +
							((tmp_mu(c) + XX.col(0).array()) * E42.array()) +
							((pow(tmp_mu(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(0).array()) + (tmp_mu(c) * XX.col(0).array())) * E43.array()) +
							((pow(tmp_mu(c), 3) + pow(XX.col(0).array(), 3) + (pow(tmp_mu(c), 2) * XX.col(0).array()) + (tmp_mu(c) * square(XX.col(0).array())) +
							(3.0 * tmp_var(c) * XX.col(0).array()) + (5.0 * tmp_mu(c) * tmp_var(c))) * EX4);
			TMatrix E4A43 = E41.array() +
							((tmp_mu(c) + XX.col(1).array()) * E42.array()) +
							((pow(tmp_mu(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(1).array()) + (tmp_mu(c) * XX.col(1).array())) * E43.array()) +
							((pow(tmp_mu(c), 3) + pow(XX.col(1).array(), 3) + (pow(tmp_mu(c), 2) * XX.col(1).array()) + (tmp_mu(c) * square(XX.col(1).array())) +
							(3.0 * tmp_var(c) * XX.col(1).array()) + (5.0 * tmp_mu(c) * tmp_var(c))) * EX4);

			TMatrix P2    = exp(-sqrtf * (XX.col(1).array() - XX.col(0).array()) / tmp_ls(c)) * 					
							((0.5 * E4A41.array() * (erf((XX.col(1).array() - tmp_mu(c)) / (sqrt(2.0 * tmp_var(c)))) -
							erf((XX.col(0).array() - tmp_mu(c)) / (sqrt(2.0 * tmp_var(c)))))) +
							(E4A42.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(0).array() - tmp_mu(c)) / tmp_var(c))) -
							(E4A43.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(1).array() - tmp_mu(c)) / tmp_var(c))));
			/* ================================ COMPUTE E5  ================================ */
			TMatrix E50	  = 1.0 + (((25.0 * op_sq_prod.array()) +
							(3.0 * sqrtf * ((3.0 * pow(tmp_ls(c), 3)) + (5.0 * tmp_ls(c) * op_prod.array())) * op_sum.array()) +
							(15.0 * ls_sqrd * (op_sq_sum.array() + (3.0 * op_prod.array())))) / (denominator));
			TMatrix E51   = (((op_sq_sum.array() * (15.0 * sqrtf * tmp_ls(c))) +
							(((50.0 * op_prod.array()) + (75.0 * ls_sqrd)) * op_sum.array()) +
							(op_prod.array() * (60.0 * sqrtf * tmp_ls(c)))) +
							(18.0 * sqrtf * pow(tmp_ls(c), 3))) / (denominator);
			TMatrix E52   = (5.0 * (((5.0 * op_sq_sum.array()) + (15.0 * ls_sqrd)) + (op_sum.array() * (9.0 * sqrtf * tmp_ls(c))) + (20.0 * op_prod.array()))) / (denominator);
			TMatrix E53   = (10.0 * ((5.0 * op_sum.array()) + (3.0 * sqrtf * tmp_ls(c)))) / (denominator);
			TMatrix E5A51 = E50.array() - (tmp_muD(c) * E51.array()) + (DE52(c) * E52.array()) - (DE53(c) * E53.array()) + (DE54(c) * EX4);
			TMatrix E5A52 = E51.array() - (tmp_muD(c) + XX.col(0).array()) * E52.array() +
							(pow(tmp_muD(c), 2) + (2.0 * tmp_var(c)) + square(XX.col(0).array()) + (tmp_muD(c) * XX.col(0).array())) * E53.array() -
							(pow(tmp_muD(c), 3) + pow(XX.col(0).array(), 3) + (pow(tmp_muD(c), 2) * XX.col(0).array()) + (tmp_muD(c) * square(XX.col(0).array())) +
							(3.0 * tmp_var(c) * XX.col(0).array()) + (5.0 * tmp_var(c) * tmp_muD(c))) * EX4;
			TMatrix P3	  = (exp((10.0 * tmp_var(c) - (sqrtf * tmp_ls(c) * (op_sum.array() - (2.0 * tmp_mu(c))))) / ls_sqrd)) *
							((0.5 * E5A51.array() * (1.0 + erf((XX.col(0).array() - tmp_muD(c)) / sqrt(2.0 * tmp_var(c))))) +
							(E5A52.array() * sqrt(0.5 * tmp_var(c) / PI) * exp(-0.5 * square(XX.col(0).array() - tmp_muD(c)) / tmp_var(c))));					
			J0.array()   *= (P1.array() + P2.array() + P3.array());
		
		}
		J.array() *= Eigen::Map<TMatrix>(J0.data(), X.rows(), X.rows()).array();
	
	};


	if (zero_indices.size() > 0) { zero_variance(); }
	if (non_zero_indices.size() > 0) { non_zero_variance(); }			
	return std::make_pair(I, J);
}




MatrixPair SE_debug_IJ(const TVector& mean, const TMatrix& X, const Eigen::Index& idx) {
	
	TMatrix I = TMatrix::Ones(X.rows(), 1);
	TMatrix J = TMatrix::Ones(X.rows(), X.rows());
	TMatrix Xz = ((X.transpose().array().colwise() - mean.array())).transpose();

	// Compute I
	TMatrix xi = (exp(((-1 * square(Xz.array())).array().rowwise() / static_cast<TRVector>(xi_term2.row(idx)).array())).matrix()) * (xi_term1.row(idx).asDiagonal());
	I = xi.rowwise().prod();
	// Compute J
	zeta(J, Xz, zeta0.row(idx), QD1.row(idx), QD2);
	return std::make_pair(I, J);
}
const TMatrix xi_term1_() const { return xi_term1; }
const TMatrix xi_term2_() const { return xi_term2; }
const TMatrix zeta0_() const { return zeta0; }
const TMatrix QD1_() const { return QD1; }
const TVector QD2_() const { return QD2; }

