#include <fstream>
#include <iostream>
#include <iomanip> 
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/timing.h"



#include "geometrycentral/surface/heat_method_distance.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/remeshing.h"
#include "geometrycentral/pointcloud/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"
#include "geometrycentral/pointcloud/point_cloud_io.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "geometrycentral/pointcloud/point_cloud_heat_solver.h"
#include "geometrycentral/surface/heat_method_distance.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "polyscope/volume_mesh.h"

// libigl
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/readMESH.h>

#include <igl/fast_winding_number.h>
#include <igl/read_triangle_mesh.h>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <igl/slice_mask.h>
#include <Eigen/Geometry>
#include <igl/octree.h>
#include <igl/barycenter.h>
#include <igl/knn.h>
#include <igl/random_points_on_mesh.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/per_face_normals.h>
#include <igl/copyleft/cgal/point_areas.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/get_seconds.h>
#include <igl/signed_distance.h>

#include <igl/winding_number.h>

#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

/*
autodiff::Array3real Vector3_to_ADVec3(geometrycentral::Vector3 vec) {
	autodiff::Array3real ADVec3 = autodiff::Array3real(autodiff::real(vec.x), autodiff::real(vec.y), autodiff::real(vec.z));
	return ADVec3;
}

geometrycentral::Vector3 ADVec3_to_Vector3(autodiff::Array3real ADVec) {
	geometrycentral::Vector3 Vec = geometrycentral::Vector3{ double(ADVec(0)), double(ADVec(1)), double(ADVec(2)) };
	return Vec;
}
*/
