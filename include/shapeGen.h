#include <cmath>
#include <algorithm>
#include <vector>
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/timing.h"

#include "geometrycentral/pointcloud/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"
#include "geometrycentral/pointcloud/point_cloud_io.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace geometrycentral::pointcloud;
typedef geometrycentral::pointcloud::Point gcPoint;

class CircleGen2D {
public:
	const double PI = std::acos(-1.0);
	Vector3 O;
	double R;
	int numPs;
	double ArcRatio; // 生成不完整的圆
	std::vector<double> weightAs;
	std::vector<double> thetas;
	std::vector<Vector3> ps, normals;
	CircleGen2D();
	CircleGen2D(Vector3 inputO, double inputR, int inputNumPs, double InputArcRatio, int isRandom);

};

class Polygon2D {
public:
	std::vector<Vector3> PolygonVs;
	std::vector<Vector3> allVs, allNormals;
	std::vector<double> weightAs;
	std::vector<double> thetas;
	int VnumOnEdge;
	Polygon2D();
	Polygon2D(std::vector<Vector3> InputPolygonVs, int InputVnumOnEdge, int isOpen, int isRandom);
};