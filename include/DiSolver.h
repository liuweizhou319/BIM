#define _CRT_SECURE_NO_WARNINGS
/*
#include "utils.h"

#include <LBFGS.h>
#include <LBFGSB.h>

using namespace geometrycentral;

class DiricheltEnery {
	
public:

	std::vector<Vector3> rawPoints, rawNormals, Pplus, Pminus, PplusNormals, PminusNormals;
	std::vector<double> WeightAs;
	
	autodiff::ArrayXreal ADWeightAs;
	autodiff::ArrayXreal ADrawPoints, ADrawNormals, ADnowPplus, ADnowPminus;

	int iter = 0;

	std::vector<std::vector<Vector3> > grid2D;

	DiricheltEnery(std::vector<Vector3> InputRawPoints, 
						std::vector<Vector3> InputRawNormals,
						//std::vector<Vector3> InputPplus, 
						//std::vector<Vector3> InputPminus, 
						//std::vector<Vector3> InputPplusNormals, 
						//std::vector<Vector3> InputPminusNormals,
						std::vector<double> InputWeightAs);

	autodiff::real calc_all_w(autodiff::ArrayXreal ps, 
							autodiff::ArrayXreal normals, 
							autodiff::Array3real queryQ, 
							autodiff::ArrayXreal ADWeightAs);

	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
}; 
*/