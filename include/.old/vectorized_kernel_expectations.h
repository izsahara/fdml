	// Squared Exponential computation of I
	{
		//TXz = X[None, :, : ] - mean[:, None, : ]
		Eigen::Tensor<double, 3 /* znrows, xnrows, ncols*/> TXz =
			TX.reshape(std::array<Index, 3>{1, xnrows, ncols}).broadcast(std::array<Index, 3>{znrows, 1, 1})
			- Tmu.reshape(std::array<Index, 3>{znrows, 1, ncols}).broadcast(std::array<Index, 3>{1, xnrows, 1});

		// Define recurring reshapes and broadcast
		auto var3d = Tvar.reshape(std::array<Index, 3>{znrows, 1, ncols});
		auto ls3d = Tls.reshape(std::array<Index, 3>{1, 1, ncols}).broadcast(std::array<Index, 3>{znrows, 1, 1});

		// term1 = np.sqrt(1 + 2 * variance / length **2)
		// term2 = np.exp(-VXz.T**2 / (2*variance+length**2))
		//Eigen::Tensor<double, 3 /* (nrows, 1, ncols) */> term1 = (1.0 + ((2.0 * var3d).eval() / ls3d.square()).eval()).sqrt();
		//Eigen::Tensor<double, 3 /* (znrows, xnrows, ncols) */> term2 = ((-1.0*TXz.square()).eval() / ((2.0 * var3d).eval() + ls3d.square()).eval().broadcast(std::array<Index, 3>{1, xnrows, 1})).exp();
		//Eigen::Tensor<double, 3 /* (znrows, xnrows, ncols) */> TI = (1.0 / term1.broadcast(std::array<Index, 3>{1, xnrows, 1})).eval() * term2;


		Eigen::Tensor<double, 2 /* (znrows, xnrows) */> TI =
			((1.0 / ((1.0 + ((2.0 * var3d).eval() / ls3d.square()).eval()).sqrt()).eval().broadcast(std::array<Index, 3>{1, xnrows, 1})).eval() *
			((-1.0 * TXz.square()).eval() / ((2.0 * var3d).eval() + ls3d.square()).eval().broadcast(std::array<Index, 3>{1, xnrows, 1})).exp()).eval().prod(std::array<Index, 1>{2});
		
		//const Eigen::Tensor<double, 2>::Dimensions& d1 = TI.dimensions();
		//cout << d1[0] <<  " " <<  d1[1] << endl;
		//cout << d1[0] <<  " " <<  d1[1] << " " << " " << d1[2] << endl;
		//cout << TI.chip(53, 0) << endl;
		//cout << endl;
	}