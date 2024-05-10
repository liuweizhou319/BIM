#include "utils.h"

namespace test2D {

#include <igl/read_triangle_mesh.h>
#include "shapeGen.h"

Eigen::MatrixXd meshV, plane;
Eigen::MatrixXi meshF;

int iter = 0;
int sz;
std::vector<Vector3> rawPoints;
std::vector<double> ADWeightAs;
std::vector<int> isThinkBand;
//autodiff::ArrayXreal ADWeightAs;
autodiff::ArrayXreal ADrawPoints, ADrawNormals, ADnowPplus, ADnowPminus;

std::vector<std::vector<int> > CPs;

void DataPre(std::string filepath = "../../../../data/marble_cat_iso.obj") {
	igl::read_triangle_mesh(filepath, meshV, meshF);
	plane = meshV;
	sz = plane.rows();
}

}