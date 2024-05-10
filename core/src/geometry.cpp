// PLEASE READ:
//
// This file implements additional geometry routines for the VertexPositionGeometry class in Geometry Central. Because
// we are "inside" the class, we no longer have to call
//
//          geometry->inputVertexPositions[v], etc.
//
// We can just call
//
//          this->inputVertexPositions[v], etc.
//
// or simply
//
//          inputVertexPositions[v], etc.
//
// In addition, we no longer access the corresponding surface mesh via
//
//          mesh->vertices(), etc.
//
// but instead <mesh> is not a pointer anymore, so we use
//
//          mesh.vertices(), etc.
//
// Functions in this file can be called from other projects simply by using geometry->cotan(he),
// geometry->barycentricDualArea(v), etc. where "geometry" is a pointer to a VertexPositionGeometry. This avoids having
// to declare a GeometryRoutines object in every project, and also mimics the way that geometry routines are normally
// called in Geometry Central.
//
// Other notes: In this file, you can use the constant pi by using PI.

#include "geometrycentral/surface/vertex_position_geometry.h"
#include <complex>
#include <cassert>

namespace geometrycentral {
    namespace surface {

        /*
         * Compute the Euler characteristic of the mesh.
         */
        int VertexPositionGeometry::eulerCharacteristic() const {
            return (int)mesh.nVertices() - (int)mesh.nEdges() + (int)mesh.nFaces();
        }

        /*
         * Compute the mean length of all the edges in the mesh.
         *
         * Input:
         * Returns: The mean edge length.
         */
        double VertexPositionGeometry::meanEdgeLength() const {

            double total = 0.0;
            for (Edge e : mesh.edges()) {
                total += edgeLength(e);
            }
            return total / mesh.nEdges();
        }

        /*
         * Compute the total surface area of the mesh.
         *
         * Input:
         * Returns: The surface area of the mesh.
         */
        double VertexPositionGeometry::totalArea() const {

            double total = 0.0;
            for (Face f : mesh.faces()) {
                total += faceArea(f);
            }
            return total;
        }

        /*
         * Computes the cotangent of the angle opposite to a halfedge.
         *
         * Input: The halfedge whose cotan weight is to be computed.
         * Returns: The cotan of the angle opposite the given halfedge.
         */
        double VertexPositionGeometry::cotan(Halfedge he) const {
            // return edgeCotanWeight(he.edge());

            if (he.isInterior() == false) return 0.0;

            Vector3 a = inputVertexPositions[he.next().tipVertex()],
                b = inputVertexPositions[he.tailVertex()],
                c = inputVertexPositions[he.tipVertex()],
                u = b - a,
                v = c - a;
            return dot(u, v) / cross(u, v).norm();
        }

        /*
         * Computes the barycentric dual area of a vertex.
         *
         * Input: The vertex whose barycentric dual area is to be computed.
         * Returns: The barycentric dual area of the given vertex.
         */
        double VertexPositionGeometry::barycentricDualArea(Vertex v) const {
            double area = 0;
            for (Face f : v.adjacentFaces()) area += faceArea(f) / 3;
            return area;
        }

        /*
         * Computes the angle (in radians) at a given corner.
         *
         *
         * Input: The corner at which the angle needs to be computed.
         * Returns: The angle clamped between 0 and ¦Ð.
         */
        double VertexPositionGeometry::angle(Corner c) const {

            // TODO
            Halfedge he = c.halfedge();


            double calcAngle = 0.0;
            Vector3 a = inputVertexPositions[he.tailVertex()];
            Vector3 b = inputVertexPositions[he.tipVertex()];
            Vector3 cc = { 0, 0, 0 };
            for (Halfedge h : c.vertex().incomingHalfedges()) {
                if (h.next() == he) {
                    cc = inputVertexPositions[h.tailVertex()];
                }
            }
            Vector3 p = b - a, q = cc - a;
            p = p.normalize();
            q = q.normalize();
            calcAngle = acos(dot(p, q));
            return calcAngle; // placeholder
        }

        /*
         * Computes the signed angle (in radians) between two adjacent faces.
         *
         * Input: The halfedge (shared by the two adjacent faces) on which the dihedral angle is computed.
         * Returns: The dihedral angle.
         */
        double VertexPositionGeometry::dihedralAngle(Halfedge he) const {

            // TODO
            Vector3 Nijk = faceNormal(he.face());
            Vector3 Nijl = faceNormal(he.twin().face());

            Vector3 eij = inputVertexPositions[he.tipVertex()] - inputVertexPositions[he.tailVertex()];
            eij = eij.normalize();

            double calcDihedralAngle = 0;
            double y = dot(eij, cross(Nijk, Nijl));
            double x = dot(Nijk, Nijl);
            calcDihedralAngle = atan2(y, x);

            return calcDihedralAngle; // placeholder
        }

        /*
         * Computes the normal at a vertex using the "equally weighted" method.
         *
         * Input: The vertex on which the normal is to be computed.
         * Returns: The "equally weighted" normal vector.
         */
        Vector3 VertexPositionGeometry::vertexNormalEquallyWeighted(Vertex v) const {

            // TODO

            Vector3 calcNormal = { 0, 0, 0 };
            for (Face f : v.adjacentFaces()) {
                calcNormal += faceNormal(f);
            }
            calcNormal /= 1.0 * v.degree();
            return calcNormal.normalize(); // placeholder
        }

        /*
         * Computes the normal at a vertex using the "tip angle weights" method.
         *
         * Input: The vertex on which the normal is to be computed.
         * Returns: The "tip angle weights" normal vector.
         */
        Vector3 VertexPositionGeometry::vertexNormalAngleWeighted(Vertex v) const {

            // TODO
            Vector3 calcNormal = { 0, 0, 0 };
            for (Halfedge he : v.outgoingHalfedges()) {
                calcNormal += angle(he.corner()) * faceNormal(he.face());
            }

            return calcNormal.normalize(); // placeholder
        }

        /*
         * Computes the normal at a vertex using the "inscribed sphere" method.
         *
         * Input: The vertex on which the normal is to be computed.
         * Returns: The "inscribed sphere" normal vector.
         */
        Vector3 VertexPositionGeometry::vertexNormalSphereInscribed(Vertex v) const {

            // TODO
            Vector3 vNsi = { 0, 0, 0 };
            for (Halfedge he : v.incomingHalfedges()) {
                //std::assert(he.tailVertex() == v);

                Vector3 eij = inputVertexPositions[he.tailVertex()] - inputVertexPositions[he.tipVertex()];
                Vector3 eik = inputVertexPositions[he.next().tipVertex()] - inputVertexPositions[he.next().tailVertex()];
                vNsi += cross(eij, eik) * (1.0 / (eij.norm2() * eik.norm2()));
            }

            vNsi = -vNsi;
            return vNsi.normalize(); // placeholder
        }

        /*
         * Computes the normal at a vertex using the "face area weights" method.
         *
         * Input: The vertex on which the normal is to be computed.
         * Returns: The "face area weighted" normal vector.
         */
        Vector3 VertexPositionGeometry::vertexNormalAreaWeighted(Vertex v) const {

            // TODO
            Vector3 vNaw = { 0, 0, 0 };
            for (Face f : v.adjacentFaces()) {
                vNaw += (faceArea(f) * faceNormal(f));
            }
            return vNaw.normalize(); // placeholder
        }

        /*
         * Computes the normal at a vertex using the "Gauss curvature" method.
         *
         * Input: The vertex on which the normal is to be computed.
         * Returns: The "Gauss curvature" normal vector.
         */


        Vector3 VertexPositionGeometry::vertexNormalGaussianCurvature(Vertex v) const {
            Vector3 calcCurvature = { 0, 0, 0 };
            for (Edge e : v.adjacentEdges()) {
                Halfedge he = e.halfedge();
                Vector3 eij = inputVertexPositions[e.secondVertex()] - inputVertexPositions[e.firstVertex()];
                if (e.firstVertex() != v) eij = -eij;
                calcCurvature += dihedralAngle(he) * eij.normalize();
            }
            calcCurvature *= 0.5;
            return calcCurvature.normalize();
        }

        /*
         * Computes the normal at a vertex using the "mean curvature" method (equivalent to the "area gradient" method).
         *
         * Input: The vertex on which the normal is to be computed.
         * Returns: The "mean curvature" normal vector.
         */
        Vector3 VertexPositionGeometry::vertexNormalMeanCurvature(Vertex v) const {
            Vector3 calcCurvature = { 0, 0, 0 };
            for (Edge e : v.adjacentEdges()) {
                Halfedge he = e.halfedge();
                Vector3 eij = inputVertexPositions[e.secondVertex()] - inputVertexPositions[e.firstVertex()];
                if (e.firstVertex() != v) eij = -eij;
                calcCurvature += (cotan(he) + cotan(he.twin())) * eij;
            }
            calcCurvature *= 0.5;
            return calcCurvature.normalize();
        }

        /*
         * Computes the angle defect at a vertex.
         *
         * Input: The vertex whose angle defect is to be computed.
         * Returns: The angle defect of the given vertex.
         */
        double VertexPositionGeometry::angleDefect(Vertex v) const {

            // TODO
            double calcAngleDefect = 2.0 * PI;
            for (Halfedge he : v.outgoingHalfedges()) {
                calcAngleDefect -= angle(he.corner());
            }
            return calcAngleDefect; // placeholder
        }

        /*
         * Computes the total angle defect of the mesh.
         *
         * Input:
         * Returns: The total angle defect
         */
        double VertexPositionGeometry::totalAngleDefect() const {

            // TODO
            double totAD = 0;
            for (Vertex v : mesh.vertices()) {
                totAD += angleDefect(v);
            }
            return totAD; // placeholder
        }

        /*
         * Computes the (integrated) scalar mean curvature at a vertex.
         *
         * Input: The vertex whose mean curvature is to be computed.
         * Returns: The mean curvature at the given vertex.
         */
        double VertexPositionGeometry::scalarMeanCurvature(Vertex v) const {

            // TODO
            double smc = 0;
            for (Edge e : v.adjacentEdges()) {
                Halfedge he = e.halfedge();
                smc += dihedralAngle(he) * edgeLength(e);
            }
            smc *= 0.5;
            return smc; // placeholder
        }

        /*
         * Computes the circumcentric dual area of a vertex.
         *
         * Input: The vertex whose circumcentric dual area is to be computed.
         * Returns: The circumcentric dual area of the given vertex.
         */
        double VertexPositionGeometry::circumcentricDualArea(Vertex v) const {
            double area = 0;
            for (Edge e : v.adjacentEdges()) {
                Halfedge he = e.halfedge();
                area += pow(edgeLength(e), 2) * (cotan(he) + cotan(he.twin()));
            }
            return area / 8.0;
        }

        /*
         * Computes the (pointwise) minimum and maximum principal curvature values at a vertex.
         *
         * Input: The vertex on which the principal curvatures need to be computed.
         * Returns: A std::pair containing the minimum and maximum principal curvature values at a vertex.
         */
        std::pair<double, double> VertexPositionGeometry::principalCurvatures(Vertex v) const {

            /*
                k1 = H + sqrt(H * H - 2k)
            */
            double K = angleDefect(v) / circumcentricDualArea(v);
            double H = scalarMeanCurvature(v) / circumcentricDualArea(v);
            //std::cout << "K = " << K << ", H = " << H << "\n";
            //std::cout << "fuck = " << H * H -  K << "\n";
            double k1 = H + sqrt(H * H - K);
            double k2 = H - sqrt(H * H - K);

            return std::make_pair(k2, k1); // placeholder
        }


        /*
         * Builds the sparse POSITIVE DEFINITE Laplace matrix. Do this by building the negative semidefinite Laplace matrix,
         * multiplying by -1, and shifting the diagonal elements by a small constant (e.g. 1e-8).
         *
         * Input:
         * Returns: Sparse positive definite Laplace matrix for the mesh.
         */
        SparseMatrix<double> VertexPositionGeometry::laplaceMatrix() const {

            // TODO
            const double epsTri = 1e-8;
            // 
            std::vector<Eigen::Triplet<double> > vTri;
            for (Vertex v : mesh.vertices()) {
                double totcotan = 0;
                for (Halfedge he : v.outgoingHalfedges()) {
                    Vertex vto = he.tipVertex();
                    vTri.push_back(
                        Eigen::Triplet<double>(v.getIndex(), vto.getIndex(), -0.5 * (cotan(he) + cotan(he.twin())))
                    );
                    totcotan += (cotan(he) + cotan(he.twin()));
                }
                vTri.push_back(
                    Eigen::Triplet<double>(v.getIndex(), v.getIndex(), 0.5 * totcotan + epsTri)
                );
            }
            std::cout << "vTri. size() = " << vTri.size() << std::endl;
            Eigen::SparseMatrix<double> SM(mesh.nVertices(), mesh.nVertices());
            SM.setFromTriplets(vTri.begin(), vTri.end());

            return SM; // placeholder
        }

        /*
         * Builds the sparse diagonal mass matrix containing the barycentric dual area of each vertex.
         *
         * Input:
         * Returns: Sparse mass matrix for the mesh.
         */
        SparseMatrix<double> VertexPositionGeometry::massMatrix() const {

            // TODO

            std::vector<Eigen::Triplet<double> > vTri;
            for (Vertex v : mesh.vertices()) {
                vTri.push_back(
                    Eigen::Triplet<double>(v.getIndex(), v.getIndex(), barycentricDualArea(v))
                );
            }
            Eigen::SparseMatrix<double> SM(mesh.nVertices(), mesh.nVertices());
            SM.setFromTriplets(vTri.begin(), vTri.end());
            return SM; // placeholder
        }

        /*
         * Builds the sparse complex POSITIVE DEFINITE Laplace matrix. Do this by building the negative semidefinite Laplace
         * matrix, multiplying by -1, and shifting the diagonal elements by a small constant (e.g. 1e-8).
         *
         * Input:
         * Returns: Sparse complex positive definite Laplace matrix for the mesh.
         */
        SparseMatrix<std::complex<double>> VertexPositionGeometry::complexLaplaceMatrix() const {

            // TODO

            const double epsTri = 1e-8;

            std::vector<Eigen::Triplet<std::complex<double> > >vTri;
            for (Vertex v : mesh.vertices()) {
                double totcotan = 0;
                for (Halfedge he : v.outgoingHalfedges()) {
                    Vertex vto = he.tipVertex();
                    vTri.push_back(
                        Eigen::Triplet<std::complex<double> >(v.getIndex(), vto.getIndex(), std::complex<double>(-0.5 * (cotan(he) + cotan(he.twin())), 0.0))
                    );
                    totcotan += (cotan(he) + cotan(he.twin()));
                }
                vTri.push_back(
                    Eigen::Triplet<std::complex<double> >(v.getIndex(), v.getIndex(), std::complex<double>(0.5 * totcotan + epsTri, 0.0))
                );
            }
            Eigen::SparseMatrix<std::complex<double> >SM(mesh.nVertices(), mesh.nVertices());
            SM.setFromTriplets(vTri.begin(), vTri.end());


            return SM; // placeholder
        }

        /*
         * Compute the center of mass of a mesh.
         */
        Vector3 VertexPositionGeometry::centerOfMass() const {

            // Compute center of mass.
            Vector3 center = { 0.0, 0.0, 0.0 };
            for (Vertex v : mesh.vertices()) {
                center += inputVertexPositions[v];
            }
            center /= mesh.nVertices();

            return center;
        }

        /*
         * Centers a mesh about the origin.
         * Also rescales the mesh to unit radius if <rescale> == true.
         */
        void VertexPositionGeometry::normalize(const Vector3& origin, bool rescale) {

            // Compute center of mass.
            Vector3 center = centerOfMass();

            // Translate to origin [of original mesh].
            double radius = 0;
            for (Vertex v : mesh.vertices()) {
                inputVertexPositions[v] -= center;
                radius = std::max(radius, inputVertexPositions[v].norm());
            }

            // Rescale.
            if (rescale) {
                for (Vertex v : mesh.vertices()) {
                    inputVertexPositions[v] /= radius;
                }
            }

            // Translate to origin [of original mesh].
            for (Vertex v : mesh.vertices()) {
                inputVertexPositions[v] += origin;
            }
        }

    } // namespace surface
} // namespace geometrycentral