#include <DiSolver.h>
/*
DiricheltEnery::DiricheltEnery(std::vector<Vector3> InputRawPoints,
	std::vector<Vector3> InputRawNormals,
	//std::vector<Vector3> InputPplus,
	//std::vector<Vector3> InputPminus,
	//std::vector<Vector3> InputPplusNormals,
	//std::vector<Vector3> InputPminusNormals,
	std::vector<double> InputWeightAs) {
	// 
	rawPoints = InputRawPoints;
	rawNormals = InputRawNormals;
	//Pplus = InputPplus;
	//Pminus = InputPminus;
	//PplusNormals = InputPplusNormals;
	//PminusNormals = InputPminusNormals;
	WeightAs = InputWeightAs;
    ADWeightAs.resize(WeightAs.size());
    for (int i = 0; i < WeightAs.size(); i++) {
        ADWeightAs(i) = autodiff::real(WeightAs[i]);
    }

    ADrawPoints.resize(rawPoints.size() * 3);
    ADrawNormals.resize(rawNormals.size() * 3);

    assert(rawPoints.size() == rawNormals.size());
    int rawPointsSZ = rawPoints.size();
    for (int i = 0; i < rawPointsSZ; i++) {
        ADrawPoints(i * 3 + 0) = autodiff::real(rawPoints[i].x);
        ADrawPoints(i * 3 + 1) = autodiff::real(rawPoints[i].y);
        ADrawPoints(i * 3 + 2) = autodiff::real(rawPoints[i].z);

        ADrawNormals(i * 3 + 0) = autodiff::real(rawNormals[i].x);
        ADrawNormals(i * 3 + 1) = autodiff::real(rawNormals[i].y);
        ADrawNormals(i * 3 + 2) = autodiff::real(rawNormals[i].z);
    }
    
}

autodiff::real DiricheltEnery::calc_all_w(autodiff::ArrayXreal ps, autodiff::ArrayXreal normals, autodiff::Array3real queryQ, autodiff::ArrayXreal ADWeightAs) {
    // ps : 3 * n 行, 1列
    // normals : 3 * n 行, 1 列
    // queryQ ： 3 行, 1 列

    const autodiff::real _PI = autodiff::real(std::acos(-1.0));
    const autodiff::real _eps = autodiff::real(1e-8);

    auto _Dot = [&](autodiff::Array3real V, autodiff::Array3real U) -> autodiff::real {

        return V(0) * U(0) + V(1) * U(1) + V(2) * U(2);
        };
    auto _norm2 = [&](autodiff::Array3real V) ->autodiff::real {
        return _Dot(V, V);
        };
    auto calcPoissonKernel = [&](autodiff::Array3real P, autodiff::Array3real N, autodiff::Array3real Q, int i) -> autodiff::real {
        autodiff::Array3real Vqp = autodiff::Array3real(P(0) - Q(0), P(1) - Q(1), P(2) - Q(2));
        autodiff::real up = _Dot(N, Vqp);
        autodiff::real down = _norm2(Vqp);

        //std::cout << "poissonkernel:" << "up = " << up << ", down = " << down << ", weightA = " << weightA << "\n";
        return (up * ADWeightAs(i) / (_PI * autodiff::real(2.0) * (down) + _eps));
        };
    int psSize = ps.size();
    autodiff::real val = 0;

    //#pragma omp parallel for
    for (int i = 0; i < psSize; i += 3) {
        //std::cout << "i = " << i << " \n";
        autodiff::Array3real P = autodiff::Array3real(ps(i), ps(i + 1), ps(i + 2));
        autodiff::Array3real N = autodiff::Array3real(normals(i), normals(i + 1), normals(i + 2));

        val += calcPoissonKernel(P, N, queryQ, i);
    }

    return val;
}



double DiricheltEnery::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    
    double fx = 0.0;

    autodiff::real epsD = autodiff::real(1e-8);
    int rawPointsSZ = rawPoints.size();
    std::vector<Vector3> visNormals(rawPointsSZ);
    autodiff::ArrayXreal nowNormals(rawPointsSZ * 3);
    autodiff::ArrayXreal nowPplus(rawPointsSZ * 3), nowPminus(rawPointsSZ * 3);
    assert(rawPointsSZ == Pplus.size());
    assert(rawPointsSZ == Pminus.size());
    assert(x.size() * 3 == ADrawPoints.size());

    for (int i = 0; i < x.size(); i ++) {
        nowNormals(i * 3 + 0) = autodiff::real(std::cos(x(i)));
        nowNormals(i * 3 + 1) = autodiff::real(std::sin(x(i)));
        nowNormals(i * 3 + 2) = autodiff::real(0);

        visNormals.push_back(Vector3{ std::cos(x(i)), std::sin(x(i)), 0 });

        nowPplus(i * 3 + 0) = ADrawPoints(i * 3 + 0) + epsD * nowNormals(i * 3 + 0);
        nowPplus(i * 3 + 1) = ADrawPoints(i * 3 + 1) + epsD * nowNormals(i * 3 + 1);
        nowPplus(i * 3 + 2) = ADrawPoints(i * 3 + 2) + epsD * nowNormals(i * 3 + 2);

        nowPminus(i * 3 + 0) = ADrawPoints(i * 3 + 0) - epsD * nowNormals(i * 3 + 0);
        nowPminus(i * 3 + 1) = ADrawPoints(i * 3 + 1) - epsD * nowNormals(i * 3 + 1);
        nowPminus(i * 3 + 2) = ADrawPoints(i * 3 + 2) - epsD * nowNormals(i * 3 + 2);
    }
    polyscope::getPointCloud("O1")->addVectorQuantity("iter : " + std::to_string(iter), visNormals);
    //polyscope::registerPointCloud("iter : " + std::to_string(iter), )

    for (int i = 0; i < rawPointsSZ; i++) {
        autodiff::Array3real ADqueryPlus = autodiff::Array3real(autodiff::real(nowPplus(i * 3 + 0)), autodiff::real(nowPplus(i * 3 + 1)), autodiff::real(nowPplus(i * 3 + 2)));
        autodiff::Array3real ADqueryPlusNormal = autodiff::Array3real(nowNormals(i * 3 + 0), nowNormals(i * 3 + 1), nowNormals(i * 3 + 2));
        ADqueryPlusNormal = autodiff::real(-1.0) * ADqueryPlusNormal;

        autodiff::Array3real ADqueryMinus = autodiff::Array3real(nowNormals(i * 3 + 0), nowNormals(i * 3 + 1), nowNormals(i * 3 + 2));

        autodiff::real valPplus, valMinus;
        Eigen::Vector3d nowPlusgrad = autodiff::gradient(&DiricheltEnery::calc_all_w, wrt(ADqueryPlus), at(ADrawPoints, nowNormals, ADqueryPlus, autodiff::real(ADWeightAs(i))), valPplus);
        Eigen::Vector3d nowMinusgrad = autodiff::gradient(&DiricheltEnery::calc_all_w, wrt(ADqueryMinus), at(ADrawPoints, nowNormals, ADqueryMinus, autodiff::real(ADWeightAs(i))), valMinus);

        Eigen::Vector3d queryPlusNormal = Eigen::Vector3d(double(ADqueryPlusNormal(0)), double(ADqueryPlusNormal(1)), double(ADqueryPlusNormal(2)));

        fx += nowPlusgrad.dot(queryPlusNormal) * double(valPplus - valMinus);
    }

    
    std::cout << "iter : " << iter << ", fx = " << fx << "\n";
    iter++;
    return fx;
}


*/