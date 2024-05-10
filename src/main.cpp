
#include "utils.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace geometrycentral::pointcloud;
typedef geometrycentral::pointcloud::Point gcPoint;


struct PCData {
    std::unique_ptr<PointCloud> cloud;
    std::unique_ptr<PointPositionGeometry> geom;
};


struct SurfData {
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geom;
};
PCData Points;
SurfData meshs;

void showSelected() {
    /*
    // Show selected vertices.
    std::vector<Vector3> vertPos;
    std::vector<std::array<size_t, 2>> vertInd;
    for (std::set<size_t>::iterator it = polyscope::state::subset.vertices.begin();
        it != polyscope::state::subset.vertices.end(); ++it) {
        vertPos.push_back(geometry->inputVertexPositions[*it]);
    }
    polyscope::SurfaceGraphQuantity* showVerts = psMesh->addSurfaceGraphQuantity("selected vertices", vertPos, vertInd);
    showVerts->setEnabled(true);
    showVerts->setRadius(vertexRadius);
    showVerts->setColor(ORANGE_VEC);

    // Show selected edges.
    std::vector<Vector3> edgePos;
    std::vector<std::array<size_t, 2>> edgeInd;
    for (std::set<size_t>::iterator it = polyscope::state::subset.edges.begin();
        it != polyscope::state::subset.edges.end(); ++it) {
        Edge e = mesh->edge(*it);
        edgePos.push_back(geometry->inputVertexPositions[e.firstVertex()]);
        edgePos.push_back(geometry->inputVertexPositions[e.secondVertex()]);
        size_t i = edgeInd.size();
        edgeInd.push_back({ 2 * i, 2 * i + 1 });
    }
    std::cout << "edge num = " << edgePos.size() << std::endl;
    polyscope::SurfaceGraphQuantity* showEdges = psMesh->addSurfaceGraphQuantity("selected edges", edgePos, edgeInd);
    showEdges->setEnabled(true);
    showEdges->setRadius(edgeRadius);
    showEdges->setColor(ORANGE_VEC);

    // Show selected faces.
    std::vector<std::array<double, 3>> faceColors(mesh->nFaces());
    for (size_t i = 0; i < mesh->nFaces(); i++) {
        faceColors[i] = BLUE;
    }
    for (std::set<size_t>::iterator it = polyscope::state::subset.faces.begin();
        it != polyscope::state::subset.faces.end(); ++it) {
        faceColors[*it] = ORANGE;
    }
    polyscope::SurfaceFaceColorQuantity* showFaces = psMesh->addFaceColorQuantity("selected faces", faceColors);
    showFaces->setEnabled(true);
    */
}

void redraw() {
    showSelected();
    polyscope::requestRedraw();
}




void test2D() {
    
    const double PI = std::acos(-1.0);
    const double eps = 1e-8;
    int sz = 200;
    double spacing = 2.0 / (1.0 * sz);
    std::vector<int> isThinkBand(sz * sz + 5, -1);
    Eigen::MatrixXd Gplane(sz * sz, 3);
    Eigen::VectorXd W(sz * sz);
    Eigen::MatrixXd plane(sz * sz, 3);
    {

        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                int idx = i * sz + j;
                plane(idx, 0) = -1.0 + 1.0 * i * spacing;
                plane(idx, 1) = -1.0 + 1.0 * j * spacing;
                plane(idx, 2) = 0;
            }
        }
    }

    

    int tot = 300;
    
    Eigen::MatrixXd ps(tot, 3);
    Eigen::MatrixXd normals(tot, 3);
    double weightA = (0.7 * 2.0 * PI) / (1.0 * tot);

    std::random_device rd;  // 将用于获得随机数引擎的种子
    std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0);

    for (int i = 0; i < tot; i++) {
        double theta = 2.0 * i * PI / (1.0 * tot);
        ps(i, 0) = std::sqrt(0.7) * std::cos(theta);
        ps(i, 1) = std::sqrt(0.7) * std::sin(theta);
        ps(i, 2) = 0;

        double _ = dis2(gen);
        theta += _;
        normals(i, 0) = std::cos(theta);
        normals(i, 1) = std::sin(theta);
        normals(i, 2) = 0;

        //Vector3 dirtNormal{ sinu * cosv ,sinu * sinv, cosu };
        //dirtNormals.push_back(dirtNormal);

        double u = dis(gen), v = dis(gen);
        double sinu = std::sin(u), cosu = std::cos(u), sinv = std::sin(v), cosv = std::cos(v);
        //normals(i, 0) = sinu * cosv;
        //normals(i, 1) = sinu * sinv;
        //normals(i, 2) = cosu;
        //normals(i, 0) = cosv;
        //normals(i, 1) = sinu;
        if (i < 10) {
            std::cout << "ps[" << i << "] = " << ps.row(i) << ", normal = " << normals.row(i) << "\n";
        }
    }
    auto calcPossionKernel = [=](Vector3 p, Vector3 pNormal, Vector3 q) -> double {
        double up = dot(pNormal, p - q);
        double down = (p - q).norm2();
        return  (up * weightA / (2.0 * PI * (down) + eps));
        };
    std::cout << "begin calc poisson \n";
#pragma omp parallel for
    for (int i = 0; i < sz * sz; i++) {
        double val = 0;
#pragma omp parallel for
        for (int j = 0; j < tot; j++) {
            Vector3 p{ ps(j, 0), ps(j, 1), 0 }, pN{ normals(j, 0), normals(j, 1), 0 }, q{ plane(i, 0), plane(i, 1), 0 };
            val += calcPossionKernel(p, pN, q);
        }
        W(i) = val;
    }

    // eps p
    std::vector<Vector3> epsPs;
    std::vector<Vector3> epsPsNormals;
    std::vector<double> epsPsW;
    
    for (int i = 0; i < tot; i++) {
        Vector3 nowNormal{ normals(i, 0), normals(i, 1), 0 };
        Vector3 nowP{ ps(i, 0), ps(i, 1), 0 };
        epsPs.push_back(nowP + nowNormal * 1e-8);
        epsPs.push_back(nowP - nowNormal * 1e-8);
    }
    epsPsW.resize(epsPs.size());
    int epsPsSZ = epsPs.size();
//#pragma omp parallel for
    for (int i = 0; i < epsPsSZ; i++) {
        double val = 0;

        std::cout << "i = " << i << ", query= " << epsPs[i] << "\n";

        for (int j = 0; j < tot; j++) {
            Vector3 p{ ps(j, 0), ps(j, 1), 0 }, pN{ normals(j, 0), normals(j, 1), 0 }, q = epsPs[i];

            //if (i % 2 == 0) pN = -pN;

            val += calcPossionKernel(p, pN, q);
        }
        epsPsW[i] = val;
    }

    for (int i = 0; i < epsPsSZ; i += 2) {
        printf("i = %d, w = %.6f, i = %d, w = %.6f \n ", i, epsPsW[i], i + 1, epsPsW[i + 1]);
    }


    auto getIndex = [=](int i, int j)-> int {
        return i * sz + j;
        };
    /*
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            int idx = getIndex(i, j);
            int xlidx = i - 1 >= 0 ? getIndex(i - 1, j) : idx;
            int xridx = i + 1 < sz ? getIndex(i + 1, j) : idx;
            int yuidx = j + 1 < sz ? getIndex(i, j + 1) : idx;
            int ydidx = j - 1 >= 0 ? getIndex(i, j - 1) : idx;

            Gplane(idx, 0) = W(xridx) - W(xlidx);
            Gplane(idx, 1) = W(yuidx) - W(ydidx);
            Gplane(idx, 2) = 0;
        }
    }
    */
    std::cout << "begin calc thinkBand \n";
#pragma omp parallel for
    for (int i = 0; i < sz * sz; i++) {
        if (isThinkBand[i] != -1) continue;
        for (int j = 0; j < tot; j++) {
            double dist = (ps(j, 0) - plane(i, 0)) * (ps(j, 0) - plane(i, 0)) + (ps(j, 1) - plane(i, 1)) * (ps(j, 1) - plane(i, 1));
            dist = std::sqrt(dist);
            if (std::abs(dist) < 1.1 * spacing) {
                isThinkBand[i] = j;
            }
        }
    }
    std::cout << "here \n";

    double totGrad = 0, totLB = 0, totDi = 0;

    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            //if (i == 0 || j == 0 || i == sz - 1 || j == sz - 1) {
            int idx = getIndex(i, j);

            if (isThinkBand[idx] == -1) continue;

            int xlidx = i - 1 >= 0 ? getIndex(i - 1, j) : idx;
            int xridx = i + 1 < sz ? getIndex(i + 1, j) : idx;
            int yuidx = j + 1 < sz ? getIndex(i, j + 1) : idx;
            int ydidx = j - 1 >= 0 ? getIndex(i, j - 1) : idx;

            Gplane(idx, 0) = W(xridx) - W(xlidx);
            Gplane(idx, 1) = W(yuidx) - W(ydidx);
            Gplane(idx, 2) = 0;

            Vector3 Gw{ W(xridx) - W(xlidx) , W(yuidx) - W(ydidx) , 0 };
            Vector3 normalnow{ normals(isThinkBand[idx], 0), normals(isThinkBand[idx], 1), 0 };
            normalnow = normalnow.normalize();
            totDi += dot(Gw, normalnow);

            totGrad += (W(xridx) - W(xlidx)) * (W(xridx) - W(xlidx)) + (W(yuidx) - W(ydidx)) * ((W(yuidx) - W(ydidx)));
            totLB += (4.0 * W(idx) - W(xridx) - W(xlidx) - W(yuidx) - W(ydidx));
            //}
            /*
            else {
                double Gx = 0, Gy = 0;
                Gx = 1.0 * (W(getIndex(i - 1, j - 1)) - W(getIndex(i - 1, j + 1))) + 10.0 * (W(getIndex(i, j - 1)) - W(getIndex(i, j + 1))) + 1.0 * (W(getIndex(i + 1, j - 1)) - W(getIndex(i + 1, j + 1)));
                Gy = 1.0 * (W(getIndex(i - 1, j - 1)) - W(getIndex(i + 1, j - 1))) + 10.0 * (W(getIndex(i - 1, j)) - W(getIndex(i + 1, j))) + 1.0 * (W(getIndex(i - 1, j + 1)) - W(getIndex(i + 1, j + 1)));
                Gplane(getIndex(i, j), 0) = Gx;
                Gplane(getIndex(i, j), 1) = Gy;
                Gplane(getIndex(i, j), 2) = 0;
            }
            */
        }
    }


    std::cout << "tot grad = " << totGrad << ", totLB = " << totLB << ", totDi = " << totDi << "\n";

    polyscope::registerPointCloud("test points", ps);
    polyscope::getPointCloud("test points")->setPointRadius(0.001);
    polyscope::getPointCloud("test points")->addVectorQuantity("init normals", normals);
    polyscope::registerPointCloud("plane grid points", plane);
    polyscope::getPointCloud("plane grid points")->setPointRadius(0.001);
    polyscope::getPointCloud("plane grid points")->addVectorQuantity("init gradient field", Gplane);
    polyscope::getPointCloud("plane grid points")->addScalarQuantity("W", W);
    polyscope::registerPointCloud("eps point", epsPs);
    polyscope::getPointCloud("eps point")->addScalarQuantity("w", epsPsW);
    /*
    std::vector<float> vx, vy;
    {
        for (int i = 0; i < Gplane.rows(); i++) {
            vx.push_back((float)Gplane(i, 0));
            vy.push_back((float)Gplane(i, 1));
        }
    }

    VF gradientField(sz, sz, spacing, spacing, vx, vy);
    gradientField.RunHelmholtzDecomposition();
    std::vector<Vector3> nowd, nowr, nowh, calcfinal;
    for (int i = 0; i < gradientField.d.sz; i++) {
        nowd.push_back(Vector3{ gradientField.d.u[i], gradientField.d.v[i], 0});
        nowr.push_back(Vector3{ gradientField.r.u[i], gradientField.r.v[i], 0 });
        nowh.push_back(Vector3{ gradientField.h.u[i], gradientField.h.v[i], 0 });
        calcfinal.push_back(Vector3{ gradientField.d.u[i] + gradientField.r.u[i] + gradientField.h.u[i], gradientField.d.v[i] + gradientField.r.v[i] + gradientField.h.v[i] });
    }

    polyscope::getPointCloud("plane grid points")->addVectorQuantity("d", nowd);
    polyscope::getPointCloud("plane grid points")->addVectorQuantity("r", nowr);
    polyscope::getPointCloud("plane grid points")->addVectorQuantity("h", nowh);
    polyscope::getPointCloud("plane grid points")->addVectorQuantity("d + r + h", calcfinal);
    */
}



#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"
using namespace autodiff;



autodiff::real calc_w(const autodiff::ArrayXreal& ps, const autodiff::ArrayXreal& normals,  const autodiff::Array3real& queryQ, const autodiff::real weightA) {
    // ps : 3 * n 行, 1列
    // normals : 3 * n 行, 1 列
    // queryQ ： 3 行, 1 列

    const autodiff::real _PI = autodiff::real(std::acos(-1.0));
    const autodiff::real _eps = autodiff::real(1e-8);

    auto _Dot = [&](autodiff::Array3real V, autodiff::Array3real U) -> autodiff::real {
        /*
        Vector3 gcV{ (double)V(0), (double)V(1), (double)V(2) };
        Vector3 gcU{ (double)U(0), (double)U(1), (double)U(2) };
        return autodiff::real(dot(gcV, gcU));
        */
        return V(0) * U(0) + V(1) * U(1) + V(2) * U(2);
    };
    auto _norm2 = [&](autodiff::Array3real V) ->autodiff::real {
        return _Dot(V, V) ;
    };
    auto calcPoissonKernel = [&](autodiff::Array3real P, autodiff::Array3real N, autodiff::Array3real Q) -> autodiff::real {
        autodiff::Array3real Vqp = autodiff::Array3real(P(0) - Q(0), P(1) - Q(1), P(2) - Q(2));
        autodiff::real up = _Dot(N, Vqp);
        autodiff::real down = _norm2(Vqp);
        /*
        if (down < autodiff::real(1e-6)) 
            return autodiff::real(0);
        */
        //std::cout << "poissonkernel:" << "up = " << up << ", down = " << down << ", weightA = " << weightA << "\n";
        return ( up * weightA / (_PI * autodiff::real(2.0) * (down ) + _eps) );
    };
    int psSize = ps.size();
    autodiff::real val = 0;

//#pragma omp parallel for
    for (int i = 0; i < psSize; i += 3) {
        //std::cout << "i = " << i << " \n";
        autodiff::Array3real P = autodiff::Array3real(ps(i), ps(i + 1), ps(i + 2));
        autodiff::Array3real N = autodiff::Array3real(normals(i), normals(i + 1), normals(i + 2));
        
        val += calcPoissonKernel(P, N, queryQ);
    }

    return val;
}

void go() {
    const double PI = std::acos(-1.0);
    const double eps = 1e-8;
    int sz = 200;
    double spacing = 2.0 / (1.0 * sz);
    std::vector<int> isThinkBand(sz * sz + 5, -1);
    Eigen::MatrixXd Gplane(sz * sz, 3);
    Eigen::VectorXd W(sz * sz);
    Eigen::MatrixXd plane(sz * sz, 3);
    {

        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                int idx = i * sz + j;
                plane(idx, 0) = -1.0 + 1.0 * i * spacing;
                plane(idx, 1) = -1.0 + 1.0 * j * spacing;
                plane(idx, 2) = 0;
            }
        }
    }



    int tot = 300;

    Eigen::MatrixXd ps(tot, 3);
    Eigen::MatrixXd normals(tot, 3);
    Eigen::VectorXd epsPW(tot * 2); 
    double weightA = (0.7 * 2.0 * PI) / (1.0 * tot);

    std::random_device rd;  // 将用于获得随机数引擎的种子
    std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0);

    for (int i = 0; i < tot; i++) {
        double theta = 2.0 * i * PI / (1.0 * tot);
        ps(i, 0) = std::sqrt(0.7) * std::cos(theta);
        ps(i, 1) = std::sqrt(0.7) * std::sin(theta);
        ps(i, 2) = 0;

        double _ = dis2(gen);
        theta += _;
        normals(i, 0) = std::cos(theta);
        normals(i, 1) = std::sin(theta);
        normals(i, 2) = 0;

        //Vector3 dirtNormal{ sinu * cosv ,sinu * sinv, cosu };
        //dirtNormals.push_back(dirtNormal);

        double u = dis(gen), v = dis(gen);
        double sinu = std::sin(u), cosu = std::cos(u), sinv = std::sin(v), cosv = std::cos(v);
        //normals(i, 0) = sinu * cosv;
        //normals(i, 1) = sinu * sinv;
        //normals(i, 2) = cosu;
        //normals(i, 0) = cosu;
        //normals(i, 1) = sinu;
        if (i < 10) {
            std::cout << "ps[" << i << "] = " << ps.row(i) << ", normal = " << normals.row(i) << "\n";
        }
    }
    
    // autodiff 的数据结构
    autodiff::ArrayXreal ADps(tot * 3), ADnormals(tot * 3), ADqueryQs(tot * 2 * 3);
    autodiff::ArrayXreal ADplaneps(sz * sz * 3);
    std::vector<Eigen::Vector3d> grad_epspw(tot * 2), grad_psw(sz * sz);

    for (int i = 0; i < tot; i++) {
        
        ADps(i * 3 + 0) = autodiff::real(ps(i, 0));
        ADps(i * 3 + 1) = autodiff::real(ps(i, 1));
        ADps(i * 3 + 2) = autodiff::real(ps(i, 2));
        ADnormals(i * 3 + 0) = autodiff::real(normals(i, 0));
        ADnormals(i * 3 + 1) = autodiff::real(normals(i, 1));
        ADnormals(i * 3 + 2) = autodiff::real(normals(i, 2));
        ADqueryQs(i * 6 + 0) = autodiff::real(ps(i, 0) + normals(i, 0) * eps );
        ADqueryQs(i * 6 + 1) = autodiff::real(ps(i, 1) + normals(i, 1) * eps );
        ADqueryQs(i * 6 + 2) = autodiff::real(ps(i, 2) + normals(i, 2) * eps );
        ADqueryQs(i * 6 + 3) = autodiff::real(ps(i, 0) - normals(i, 0) * eps );
        ADqueryQs(i * 6 + 4) = autodiff::real(ps(i, 1) - normals(i, 1) * eps );
        ADqueryQs(i * 6 + 5) = autodiff::real(ps(i, 2) - normals(i, 2) * eps );
        //std::cout << "i = " << i << ", ADps = (" << ADps(i * 3 + 0) << ", " << ADps(i * 3 + 1) << ", " << ADps(i * 3 + 2) << "), ps = " << ps.row(i).transpose() << "\n";
    }
    std::cout << "begin calc \n";
    autodiff::real tote = 0;
    for (int i = 0; i < tot * 2; i++) {
        autodiff::Array3real ADqueryQ = autodiff::Array3real(ADqueryQs(i * 3 + 0), ADqueryQs(i * 3 + 1), ADqueryQs(i * 3 + 2));
        std::cout << "i = " << i << ", query= " << ADqueryQ.transpose() << "\n";
        //W(i) = (double)calc_w(ADps, ADnormals, ADqueryQ, autodiff::real(weightA));
        autodiff::real nowval_w;
        Eigen::Vector3d nowgrad = autodiff::gradient(calc_w, wrt(ADqueryQ), at(ADps, ADnormals, ADqueryQ, autodiff::real(weightA)), nowval_w);
        autodiff::real _nowval_w = calc_w(ADps, ADnormals, ADqueryQ, autodiff::real(weightA));
        std::cout << "nowval_w = " << nowval_w << ", _nowcalc_w" << _nowval_w << "\n";
        epsPW(i) = (double)nowval_w;
        grad_epspw[i] = (Eigen::Vector3d(nowgrad.x(), nowgrad.y(), nowgrad.z()));
        if (i % 2 == 0) {
            tote += grad_epspw[i].dot(-normals.row(i / 2)) * epsPW(i);
        }
        else if (i % 2 == 1) {
            tote += grad_epspw[i].dot(normals.row(i / 2)) * epsPW(i);
        }
    }

    for (int i = 0; i < tot * 2; i += 2) {
        std::cout << "i = " << i << ", w = " << epsPW(i) << ", i = " << i + 1 << ", W = " << epsPW(i + 1) << ", diff w" << epsPW(i) - epsPW(i + 1) << "\n";
    }
    std::cout << "tot e  = " << tote << "\n";
    /*
#pragma omp parallel for
    for (int i = 0; i < sz * sz; i++) {
        //printf("i = %d\n", i);
        autodiff::Array3real ADqueryQ = autodiff::Array3real(plane(i, 0), plane(i, 1), plane(i, 2));
        autodiff::real nowval_w;
        Eigen::Vector3d nowgrad = autodiff::gradient(calc_w, wrt(ADqueryQ), at(ADps, ADnormals, ADqueryQ, autodiff::real(weightA)), nowval_w);
        autodiff::real _nowval_w = calc_w(ADps, ADnormals, ADqueryQ, autodiff::real(weightA));
        W(i) = (double)nowval_w;
        grad_psw[i] = (Eigen::Vector3d(nowgrad.x(), nowgrad.y(), nowgrad.z()));
    }
    */
    
    std::cout << "begin plot \n";
    polyscope::getPointCloud("eps point")->addVectorQuantity("grad_eps point", grad_epspw);
    polyscope::getPointCloud("eps point")->addScalarQuantity("autodiff eps w", epsPW);

    polyscope::getPointCloud("test points")->addVectorQuantity("go normal", normals);
    /*
    polyscope::getPointCloud("plane grid points")->addScalarQuantity("autodiff W", W);
    polyscope::getPointCloud("plane grid points")->addVectorQuantity("autodiff grad W", grad_psw);
    */
    
}

void functionCallback() {
    if (ImGui::Button("test2D")) {
        test2D();
    }
    if (ImGui::Button("2D")) {
        go();
    }
}

#include <LBFGS.h>
using namespace LBFGSpp;
using Eigen::VectorXd;
int ccc = 0;
class Rosenbrock
{
private:
    int n;
    std::vector<double> _a;
public:
    Rosenbrock(int n_) : n(n_) {
        _a.resize(n, 1.0);
    }
    double operator()(const VectorXd& x, VectorXd& grad)
    {
        double fx = 0.0;
        for (int i = 0; i < n; i += 2)
        {
            double t1 = 1.0 - x[i];
            double t2 = 10 * (x[i + 1] - x[i] * x[i]);
            grad[i + 1] = 20 * t2;
            grad[i] = -2.0 * (x[i] * grad[i + 1] + t1);
            fx += t1 * t1 + t2 * t2;
            fx *= _a[i];
        }
        //std::cout << "ccc = " << ccc << "\n";
        ccc++;
        return fx;
    }
};
void testLBFGSpp() {
    const int n = 10;
    // Set up parameters
    LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 5;

    // Create solver and function object
    LBFGSSolver<double> solver(param);
    Rosenbrock fun(n);

    // Initial guess
    VectorXd x = VectorXd::Zero(n);
    // x will be overwritten to be the best point found
    double fx;
    int niter = solver.minimize(fun, x, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
}

#include "shapeGen.h"
//#include "DiSolver.h"
#include <LBFGSB.h>

int iter = 0;
const int sz = 200;
std::vector<Vector3> rawPoints;
std::vector<double> ADWeightAs;
std::vector<int> isThinkBand;
//autodiff::ArrayXreal ADWeightAs;
autodiff::ArrayXreal ADrawPoints, ADrawNormals, ADnowPplus, ADnowPminus;

std::vector<std::vector<int> > CPs;
Eigen::MatrixXd plane;





double foo(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    double fx = 0.0;

    autodiff::real epsD = autodiff::real(1e-3);
    int rawPointsSZ = rawPoints.size();

    std::vector<Vector3> visNormals(rawPointsSZ);

    autodiff::ArrayXreal nowNormals(rawPointsSZ * 3);
    autodiff::ArrayXreal nowPplus(rawPointsSZ * 3), nowPminus(rawPointsSZ * 3);
    //assert(rawPointsSZ == Pplus.size());
    //assert(rawPointsSZ == Pminus.size());
    //assert(x.size() * 3 == ADrawPoints.size());

    for (int i = 0; i < x.size(); i++) {
        nowNormals(i * 3 + 0) = autodiff::real(std::cos(x(i)));
        nowNormals(i * 3 + 1) = autodiff::real(std::sin(x(i)));
        nowNormals(i * 3 + 2) = autodiff::real(0);

        visNormals[i] = (Vector3{ std::cos(x(i)), std::sin(x(i)), 0 });

        nowPplus(i * 3 + 0) = ADrawPoints(i * 3 + 0) + epsD * nowNormals(i * 3 + 0);
        nowPplus(i * 3 + 1) = ADrawPoints(i * 3 + 1) + epsD * nowNormals(i * 3 + 1);
        nowPplus(i * 3 + 2) = ADrawPoints(i * 3 + 2) + epsD * nowNormals(i * 3 + 2);

        nowPminus(i * 3 + 0) = ADrawPoints(i * 3 + 0) - epsD * nowNormals(i * 3 + 0);
        nowPminus(i * 3 + 1) = ADrawPoints(i * 3 + 1) - epsD * nowNormals(i * 3 + 1);
        nowPminus(i * 3 + 2) = ADrawPoints(i * 3 + 2) - epsD * nowNormals(i * 3 + 2);
    }
    std::cout << "x.size = " << x.size() << ", visNormals.size = " << visNormals.size() << ", rawPointsSZ = " << rawPointsSZ << "\n";
    polyscope::getPointCloud("O1")->addVectorQuantity("iter : " + std::to_string(iter), visNormals);
    //polyscope::registerPointCloud("iter : " + std::to_string(iter), )

    for (int i = 0; i < rawPointsSZ; i++) {
        autodiff::Array3real ADqueryPlus = autodiff::Array3real(autodiff::real(nowPplus(i * 3 + 0)), autodiff::real(nowPplus(i * 3 + 1)), autodiff::real(nowPplus(i * 3 + 2)));
        autodiff::Array3real ADqueryPlusNormal = autodiff::Array3real(nowNormals(i * 3 + 0), nowNormals(i * 3 + 1), nowNormals(i * 3 + 2));
        ADqueryPlusNormal = autodiff::real(-1.0) * ADqueryPlusNormal;

        autodiff::Array3real ADqueryMinus = autodiff::Array3real(nowNormals(i * 3 + 0), nowNormals(i * 3 + 1), nowNormals(i * 3 + 2));

        autodiff::real valPplus, valMinus;

        Eigen::Vector3d nowPlusgrad = autodiff::gradient(calc_w, wrt(ADqueryPlus), at(ADrawPoints, nowNormals, ADqueryPlus, autodiff::real(ADWeightAs[i])), valPplus);
        Eigen::Vector3d nowMinusgrad = autodiff::gradient(calc_w, wrt(ADqueryMinus), at(ADrawPoints, nowNormals, ADqueryMinus, autodiff::real(ADWeightAs[i])), valMinus);

        Eigen::Vector3d queryPlusNormal = Eigen::Vector3d(double(ADqueryPlusNormal(0)), double(ADqueryPlusNormal(1)), double(ADqueryPlusNormal(2)));

        fx += nowPlusgrad.dot(queryPlusNormal) * double(valPplus - valMinus);
    }
    /*
    for (int i = 0; i < x.size(); i++) {
        Vector3 nowN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Vector3 PminusN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Vector3 PplusN = -1.0 * PminusN;
        double grad_Pplus = 0, grad_Pminus = 0;
        //p plus

        Vector3 dn_plus{ std::sin(x(i)), -std::cos(x(i)), 0 };
        grad_Pplus += calcPossionKernel(rawPoints[i], -dn_plus, Pplus[i], ADWeightAs[i]) * (dot(grad_p[i], PplusN));
        grad_Pplus += calcPossionKernel(rawPoints[i], nowN, Pplus[i], ADWeightAs[i]) * (dot(grad_p[i], dn_plus));
        //p minus

        Vector3 dn_minus{ -std::sin(x(i)), std::cos(x(i)), 0 };
        grad_Pminus += calcPossionKernel(rawPoints[i], dn_minus, Pminus[i], ADWeightAs[i]) * (dot(grad_p[i], PminusN));
        grad_Pminus += calcPossionKernel(rawPoints[i], nowN, Pminus[i], ADWeightAs[i]) * (dot(grad_p[i], dn_minus));
        grad(i) = grad_Pplus + grad_Pminus;

        //grad(i) = dot(grad_p[i] , -dn_minus);
        std::cout << "i = " << i << ", grad = " << grad(i) << "\n";

    }
    */

    //fx = -fx;

    std::cout << "iter : " << iter++ << "\n";
    std::cout << "fx = " << fx << ", grad.norm = " << grad.norm() << "\n";

    /*
    polyscope::getPointCloud("plane")->addScalarQuantity("W " + std::to_string(iter), W);
    polyscope::getPointCloud("plane")->addVectorQuantity("grad " + std::to_string(iter), Grad_plane);
    polyscope::getPointCloud("O1")->addVectorQuantity("normal " + std::to_string(iter), normals);
    */

    //fx = -fx;
    std::cout << "iter : " << iter << ", fx = " << fx << "\n";
    iter++;
    return fx;
}

/*
* Vector field decom
*
*/
#include "DealVF.h"

Eigen::MatrixXd Vd;
Eigen::MatrixXi Fd;

double foo2(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    double spacing = 2.0 / (1.0 * sz);
    double area = 1.0; // spacing * spacing
    const double epsP = 1.5e-3;

    int tot = rawPoints.size();

    Eigen::MatrixXd normals(x.size(), 3);
    for (int i = 0; i < x.size(); i++) {
        normals(i, 0) = std::cos(x(i));
        normals(i, 1) = std::sin(x(i));
        normals(i, 2) = 0;
    }
    std::cout << "spacing = " << spacing << "\n";
    const double eps = 1e-8;
    const double epsD = 0.01;
    Eigen::VectorXd W(sz * sz);
    auto calcPossionKernel = [=](Vector3 p, Vector3 pNormal, Vector3 q, double nowWeightA) -> std::pair<double, Vector3> {
        const double epsDD = 0.0001;
        Vector3 L = p - q;
        /*
        if (L.norm() < epsD) {
            L = L.normalize() * epsD;
        }
        */
        double up = dot(L, pNormal);
        //double down = std::max(L.norm2(), epsDD * epsDD);

        double down = L.norm2() + eps / (2.0 * PI);

        double val = (up * nowWeightA / (2.0 * PI * (down)));

        double x2y2 = down * down;
        double dx = 0, dy = 0;
        //if (L.norm() < epsDD) {
        if(0){
            dx = -pNormal.x;
            dy = -pNormal.y;
        }
        else {
            dx = pNormal.x * ( L.y * L.y - L.x * L.x ) - 2.0 * pNormal.y * L.x * L.y;
            dy = pNormal.y * ( L.x * L.x - L.y * L.y ) - 2.0 * pNormal.x * L.x * L.y;
        }
        dx *= nowWeightA;
        dy *= nowWeightA;
        dx /= (2.0 * PI );
        dy /= (2.0 * PI );
        Vector3 valGrad{ dx / x2y2, dy / x2y2, 0 };

        return std::make_pair(val, valGrad);        
    };
    auto getIndex = [=](int i, int j)-> int {
        return i * sz + j;
    };
    
    double DirichletEnery = 0.0, DirichletEnery_deri = 0;
    Eigen::VectorXd grad_alldomain = Eigen::VectorXd::Zero(x.size());
    std::vector<Vector3> Grad_plane(sz * sz), d_theta_grad_plane(sz * sz), Grad_plane_deri(sz * sz);
    Eigen::VectorXd FEM_d_theta(tot);

    //std::cout << "begin calc poisson \n";
    
    //std::cout << "foo2 tot = " << tot << "\n";
#pragma omp parallel for
    for (int i = 0; i < sz * sz; i++) {
        double val = 0, w_d_theta_val = 0;
        Vector3 grad_i{ 0, 0, 0 }, grad_u{0, 0, 0};
//#pragma omp parallel for
        for (int j = 0; j < tot; j++) {
            Vector3 p{ rawPoints[j].x, rawPoints[j].y, 0}, pN{normals(j, 0), normals(j, 1), 0}, q{plane(i, 0), plane(i, 1), 0};
            
            if ((p - q).norm2() < epsP) {
                continue;
            }
            
            auto now_val_and_grad = calcPossionKernel(p, pN, q, ADWeightAs[j]);
            val += now_val_and_grad.first;
            grad_i += now_val_and_grad.second;
            /*
            std::cout << "i = " << i << ", j = " << j << ", calcPossionKernel(p, Vector3{-pN.y, pN.x, 0}, q, ADWeightAs[j]).second = "
                << calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, q, ADWeightAs[j]).second <<", 2=" << now_val_and_grad.second <<  "\n";
            */
            auto nowPoissonKernel = calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, q, ADWeightAs[j]);
            grad_alldomain[j] += 2.0 * dot(nowPoissonKernel.second * spacing * spacing, now_val_and_grad.second * spacing * spacing) ;
            /*
            if (grad_alldomain[j] < -1.0 * 1e-20) {
                std::cout << "wtf j = " << j << "，i = " << i << ", grad_alldomain = " << grad_alldomain[j] << "\n";
            }
            */
            /*
            if (i == 28750) {
                std::cout << "j = " << j << ", potential = " << now_val_and_grad.first << ",grad = " << now_val_and_grad.second << ", dist = " << p - q << ", normal = " << pN << "\n";
            }
            */
        }

        W(i) = val;
        Grad_plane_deri[i] = grad_i;
        
        Vector3  q{ plane(i, 0), plane(i, 1), 0 };
        for (int j = 0; j < tot; j++) {
            Vector3 p{ rawPoints[j].x, rawPoints[j].y, 0 };
            /*
            if ((p - q).norm2() < epsP) {
                Grad_plane_deri[i] = Vector3{ 0 , 0, 0 };
            }
            */
        }
        

        DirichletEnery_deri += Grad_plane_deri[i].norm2();
    }
    /*
    for (int i = 0; i < tot; i++) {
        std::cout << "grad_domain = " << grad_alldomain[i]  << "\n";
    }
    */
    {
        
        
        Eigen::VectorXd Wd(Vd.rows());
#pragma omp parallel for
        for (int i = 0; i < Vd.rows(); i++) {
            double val = 0, w_d_theta_val = 0;
            Vector3 grad_i{ 0, 0, 0 }, grad_u{ 0, 0, 0 };
            //#pragma omp parallel for
            for (int j = 0; j < tot; j++) {
                Vector3 p{ rawPoints[j].x, rawPoints[j].y, 0 }, pN{ normals(j, 0), normals(j, 1), 0 }, q{ Vd(i, 0), Vd(i, 1), 0 };

                if ((p - q).norm2() < epsP) {
                    continue;
                }

                auto now_val_and_grad = calcPossionKernel(p, pN, q, ADWeightAs[j]);
                val += now_val_and_grad.first;
                grad_i += now_val_and_grad.second;

                auto nowPoissonKernel = calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, q, ADWeightAs[j]);
                grad_alldomain[j] += 2.0 * dot(nowPoissonKernel.second * spacing * spacing, now_val_and_grad.second * spacing * spacing);

            }

            Wd(i) = val;
        }
        polyscope::getSurfaceMesh("disk")->addVertexScalarQuantity("w" + std::to_string(iter), Wd);
    }

#pragma omp parallel for
    for (int i = 0; i < sz; i++) {
#pragma omp parallel for
        for (int j = 0; j < sz; j++) {
            //if (i == 0 || j == 0 || i == sz - 1 || j == sz - 1) {
            int idx = getIndex(i, j);

            //if (isThinkBand[idx] == -1) continue;

            int xlidx = i - 1 >= 0 ? getIndex(i - 1, j) : idx;
            int xridx = i + 1 < sz ? getIndex(i + 1, j) : idx;
            int yuidx = j + 1 < sz ? getIndex(i, j + 1) : idx;
            int ydidx = j - 1 >= 0 ? getIndex(i, j - 1) : idx;

            //std::cout << "idx = " << idx << "W  = " << W(idx) << ", W diff left = " << W(xridx) - W(xlidx) << ", W diff right = " << W(yuidx) - W(ydidx) <<   "\n";
            Vector3 Gw{ W(xridx) - W(xlidx) , W(yuidx) - W(ydidx) , 0 };
            //DirichletEnery += Gw.norm2();
            Gw = -Gw;
            Grad_plane[idx] = Gw * area;
            DirichletEnery += Grad_plane[idx].norm2();

        }
    }
    
#pragma omp parallel for
    for (int k = 0; k < tot; k++) {
        double nowval = 0;
        Vector3 p{ rawPoints[k].x, rawPoints[k].y, 0 }, pN{ normals(k, 0), normals(k, 1), 0 };
#pragma omp parallel for
        for (int i = 0; i < sz; i++) {
#pragma omp parallel for
            for (int j = 0; j < sz; j++) {
                //FEM_d_theta
                int idx = getIndex(i, j);

                //if (isThinkBand[idx] == -1) continue;

                int xlidx = i - 1 >= 0 ? getIndex(i - 1, j) : idx;
                int xridx = i + 1 < sz ? getIndex(i + 1, j) : idx;
                int yuidx = j + 1 < sz ? getIndex(i, j + 1) : idx;
                int ydidx = j - 1 >= 0 ? getIndex(i, j - 1) : idx;

                Vector3 qu{ plane(yuidx, 0), plane(yuidx, 1), 0 };
                Vector3 qd{ plane(ydidx, 0), plane(ydidx, 1), 0 };
                Vector3 ql{ plane(xlidx, 0), plane(xlidx, 1), 0 };
                Vector3 qr{ plane(xridx, 0), plane(xridx, 1), 0 };
                double d_theta_w_u = calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, qu, ADWeightAs[j]).first;
                double d_theta_w_d = calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, qd, ADWeightAs[j]).first;
                double d_theta_w_l = calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, ql, ADWeightAs[j]).first;
                double d_theta_w_r = calcPossionKernel(p, Vector3{ -pN.y, pN.x, 0 }, qr, ADWeightAs[j]).first;

                Vector3 d_theta_w_now{ d_theta_w_r - d_theta_w_l , d_theta_w_u - d_theta_w_d , 0 };

                d_theta_w_now = -d_theta_w_now;

                d_theta_grad_plane[idx] = d_theta_w_now;

                nowval += 2.0 * dot(d_theta_w_now, Grad_plane[idx]);

            }
        }
        FEM_d_theta(k) = nowval;
    }
    
    
    
    double fx = 0;
    std::vector<Vector3> grad_p(tot, Vector3{0,0,0}), grad_grad_p(tot), grad_p_minus(tot,  Vector3{ 0,0,0 });
    std::vector<Vector3> d_grad_p(tot, Vector3{ 0, 0, 0 });
    std::vector<Vector3> grad_p_FEM(tot, Vector3{ 0, 0, 0 }), d_theta_grad_p_FEM(tot, Vector3{0, 0, 0});
    std::vector<Vector3> grad_p_direct(tot, Vector3{ 0, 0, 0 });
    std::vector<double> W_plus(tot), W_minus(tot);
    

    


    //std::cout << "here \n";

    std::vector<Vector3> Pplus(tot), Pminus(tot);
    for (int i = 0; i < tot; i++) {
        Vector3 nowN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Pplus[i] = rawPoints[i] + epsD * nowN;
        Pminus[i] = rawPoints[i] - epsD * nowN;
    }
    
    for (int i = 0; i < tot; i++) {
        Vector3 ni{ 0, 0, 0 }, d_theta_grad_p_ni{0, 0, 0};
        //std::cout << "i = " << i << "\n";
        for (int j = 0; j < CPs[i].size(); j++) {
            ni += Grad_plane[CPs[i][j]];

            d_theta_grad_p_ni += d_theta_grad_plane[CPs[i][j]];

            //std::cout << "Grad_plane = " << Grad_plane[j] << ", ";
        }
        //std::cout << ", ni = " << ni << "\n";
        assert(CPs[i].size() > 0);

        ni /= (1.0 * CPs[i].size());
        grad_p_FEM[i] = ni;

        d_theta_grad_p_ni /= (1.0 * CPs[i].size());
        d_theta_grad_p_FEM[i] = d_theta_grad_p_ni;
        //std::cout << "i = " << i << ", grad = " << grad_p[i] << "\n";
    }
    
    
    for (int i = 0; i < tot; i++) {

        for (int j = 0; j < tot; j++) {
            Vector3  pN{ normals(j, 0), normals(j, 1), 0 };
            auto now_val_and_grad = calcPossionKernel(rawPoints[j], pN, Pplus[i], ADWeightAs[j]) ;
            grad_p[i] += now_val_and_grad.second * area;
            grad_p_minus[i] += calcPossionKernel(rawPoints[j], pN, Pminus[i], ADWeightAs[j]).second;

            grad_p_direct[i] += calcPossionKernel(rawPoints[j], pN, rawPoints[i], ADWeightAs[j]).second;

        }
        //grad_p[i] /= (1.0 * tot);
    }
    

    Eigen::VectorXd midBalance(tot), mid_grad(tot);
    Eigen::VectorXd jumpTerm(tot), jump_grad(tot);
    Eigen::VectorXd grad_dirE(tot);
    Eigen::VectorXd grad_boundary(tot);
    double totdiff = 0;



//#pragma omp parallel for
    for (int i = 0; i < tot; i++) {
        Vector3 nowN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Vector3 PminusN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Vector3 PplusN = -1.0 * PminusN;
        double val = 0;
//#pragma omp parallel for
        for (int j = 0; j < tot; j++) {
            Vector3 pN{ normals(j, 0), normals(j, 1), 0 };
            val += calcPossionKernel(rawPoints[j], pN, Pplus[i], ADWeightAs[j]).first;
        }
        //W_plus[i] = val;
        W_plus[i] = calcPossionKernel(rawPoints[i], nowN, Pplus[i], ADWeightAs[i]).first;
        val = 0;
//#pragma omp parallel for

        double midTerm = 0, mid_grad_now = 0;

        for (int j = 0; j < tot; j++) {
            Vector3 pN{ normals(j, 0), normals(j, 1), 0 };
            val += calcPossionKernel(rawPoints[j], pN, Pminus[i], ADWeightAs[j]).first;
            //
            midTerm += calcPossionKernel(rawPoints[j], pN, rawPoints[i], ADWeightAs[j]).first;
            if (i == 10) {
                //std::cout << "j = " << j << ", val = " << calcPossionKernel(rawPoints[j], pN, rawPoints[i], ADWeightAs[j]).first << "\n";
            }
            //mid_grad_now += calcPossionKernel(rawPoints[j], Vector3{-pN.y, pN.x, 0}, rawPoints[i], ADWeightAs[j]).first;
            //
        }



        //W_minus[i] = val;
        W_minus[i] = calcPossionKernel(rawPoints[i], nowN, Pminus[i], ADWeightAs[i]).first;
        midBalance(i) = midTerm - 0.5;



        //mid_grad(i) = mid_grad_now;

        //std::cout << "i = " << i << ", w_plus = " << W_plus[i] << ", w_minus = " << W_minus[i] << ", diff = " << W_minus[i] - W_plus[i] << "\n";
        totdiff += (W_minus[i] - W_plus[i] - 1) * (W_minus[i] - W_plus[i] - 1);
        //fx += W_plus[i] * (dot(grad_p_FEM[i], PplusN)) + W_minus[i] * (dot(grad_p_FEM[i], PminusN)) /* + 10.0 * (midBalance(i)) * (midBalance(i))*/; // 
        
        jumpTerm(i) = (W_minus[i] - W_plus[i] - 1);
        //fx += 15.0 * jumpTerm(i) * jumpTerm(i)  + 10.0 * (midBalance(i)) * (midBalance(i));
        //d_jumpTerm(i) = 
        
        // mid!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //fx += /*10.0 * (W_minus[i] - W_plus[i] - 1) * (W_minus[i] - W_plus[i] - 1) +*/ 1000.0 * (midBalance(i)) * (midBalance(i));

        //fx += -2 * PI * dot(grad_p[i], PplusN); // 
        /*
        std::cout << "i = " << i << ", plus = " << W_plus[i] * (dot(grad_p[i], PplusN)) << ", minus = " 
            << W_minus[i] * (dot(grad_p[i], PminusN)) << ", mid = " 
            << 10.0 * (midBalance(i) ) * (midBalance(i) ) <<"\n";
        */
    }

    std::vector<double > vtest(tot);
    double direct_boundary_val = 0;
    { // direction 
        //#pragma omp parallel for
        for (int i = 0; i < tot; i++) {

            Vector3 Npi{ normals(i, 0), normals(i, 1), 0 };
            //#pragma omp parallel for
            for (int j = 0; j < tot; j++) {
                Vector3 Npj{ normals(j, 0), normals(j, 1), 0 };
                //#pragma omp parallel for
                for (int k = 0; k < tot; k++) {
                    Vector3 Npk{ normals(k, 0), normals(k, 1), 0 };
                    auto tmp1 = calcPossionKernel(rawPoints[i], Npi, rawPoints[k], ADWeightAs[i]);
                    auto tmp2 = calcPossionKernel(rawPoints[j], Npj, rawPoints[k], ADWeightAs[j]);
                    direct_boundary_val += tmp1.first * dot(tmp2.second, Npk);

                    if (k % 20 == 0) {
                        vtest[k] = direct_boundary_val;
                    }
                }
            }
        }
        /*
        for (int i = 0; i < tot; i++) {
            if (i % 20 == 0) {
                std::cout << "i = " << i << ", val = " << vtest[i] << "\n";
            }
        }
        */
        std::cout << "**************************************** Direct_boundary_val = " << direct_boundary_val << "\n";
    }


    //fx /= 10.0;
    std::cout << "**************************energy 0.5 = " << fx << "\n";

    // calc mid d_theta_j
    for (int j = 0; j < tot; j++) {
        Vector3 pN{ normals(j, 0), normals(j, 1), 0 };
        double nowgrad = 0;
        for (int i = 0; i < tot; i++) {
            nowgrad += 2.0 * midBalance(i) * calcPossionKernel(rawPoints[j], Vector3{ -pN.y, pN.x, 0 }, rawPoints[i], ADWeightAs[j]).first;
        }
        mid_grad(j) = nowgrad;
    }
    // calc jump d_theta_j
    for (int j = 0; j < tot; j++) {
        Vector3 pN{ normals(j, 0), normals(j, 1), 0 };
        double nowgrad = 0;
        for (int i = 0; i < tot; i++) {
            nowgrad += 2.0 * jumpTerm(i) * ( calcPossionKernel(rawPoints[j], Vector3{ -pN.y, pN.x, 0 }, Pminus[i], ADWeightAs[j]).first 
                                                - calcPossionKernel(rawPoints[j], Vector3{ -pN.y, pN.x, 0 }, Pplus[i], ADWeightAs[j]).first);
        }
        jump_grad(j) = nowgrad;
    }

    // 直接dir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    fx += DirichletEnery_deri;

    Eigen::VectorXd ggg(x.size());
//#pragma omp parallel for
    for (int k = 0; k < x.size(); k++) {
        double now = 0;
        Vector3 pkN{ normals(k, 0), normals(k, 1), 0 };
        //Vector3 
//#pragma omp parallel for
        for (int i = 0; i < sz * sz; i++) {
            
            Vector3 q{ plane(i, 0), plane(i, 1), 0 };
            auto nowki = calcPossionKernel(rawPoints[k], Vector3{-pkN.y, pkN.x, 0}, q, ADWeightAs[k]);
            
            if ((rawPoints[k] - q).norm2() < epsP) {
                continue;
            }
            
            now += 2.0 * dot(Grad_plane_deri[i], nowki.second);
        }
        ggg(k) = now;
    }
    /// *********************************************************************
    double boundary = 0;
    for (int i = 0; i < tot; i++) {
        Vector3 npi{ normals(i, 0), normals(i, 1), 0 };
        Vector3 tmp{ 0, 0, 0 };
        for (int j = 0; j < tot; j++) {
            
            if ((rawPoints[i] - rawPoints[j]).norm2() < epsP) {
                continue;
            }
            
            auto nowji = calcPossionKernel(rawPoints[j], Vector3{ normals(j, 0), normals(j, 1), 0 }, rawPoints[i], ADWeightAs[j]);
            tmp += nowji.second;
        }
        boundary += (-W_plus[i] + W_minus[i]) * dot(npi, tmp);
    }

    fx += - 1.0 * boundary;

    std::cout << "**************************boundary val = " << boundary << "\n";

    // calc boundary integral's grad
    // four cases
    
    for (int k = 0; k < x.size(); k++) {
        // grad_p_FEM, d_theta_grad_p_FEM : 
        // grad_dirE
        Vector3 Nk = Vector3{ normals(k, 0), normals(k, 1), 0 };
        Vector3 d_Nk = Vector3{ -Nk.y, Nk.x, 0 };

        //
        
        // d_theta_w
        double d_w_k_pk_plus = 0, d_w_k_pk_minus = 0;
        d_w_k_pk_plus = calcPossionKernel(rawPoints[k], d_Nk, Pplus[k], ADWeightAs[k]).first;
        d_w_k_pk_minus = calcPossionKernel(rawPoints[k], d_Nk, Pminus[k], ADWeightAs[k]).first;
        // w
        double w_k_pk_plus = 0, w_k_pk_minus = 0;
        w_k_pk_plus = calcPossionKernel(rawPoints[k], Nk, Pplus[k], ADWeightAs[k]).first;
        w_k_pk_minus = calcPossionKernel(rawPoints[k], Nk, Pminus[k], ADWeightAs[k]).first;

        double case1_1 = 0, case1_2 = 0, case2_1 = 0, case2_2 = 0;
        // case 1.1
        auto tmp1 = calcPossionKernel(rawPoints[k], d_Nk, rawPoints[k], ADWeightAs[k]);
        case1_1 = (-d_w_k_pk_plus + d_w_k_pk_minus) * (dot(Nk, grad_p[k])) +
            (-w_k_pk_plus + w_k_pk_minus) * (dot(d_Nk, grad_p[k])) +
            (-w_k_pk_plus + w_k_pk_minus) * (dot(Nk, tmp1.second));

        // case 1.2
        Vector3 sigma_grad{ 0, 0, 0 };
        for (int j = 0; j < tot; j++) {
            if (j == k) continue;
            Vector3 Nj{ normals(j, 0), normals(j, 1), 0 };
            sigma_grad += calcPossionKernel(rawPoints[j], Nj, rawPoints[k], ADWeightAs[j]).second;
        }
        case1_2 = (-d_w_k_pk_plus + d_w_k_pk_minus) * (dot(Nk, sigma_grad)) +
            (-w_k_pk_plus + w_k_pk_minus) * (dot(d_Nk, sigma_grad));

        // case 2.1

        for (int i = 0; i < tot; i++) {
            if (i == k) continue;
            Vector3 Npi{ normals(i, 0), normals(i, 1), 0 };
            Vector3 d_Npi{ -Npi.y, Npi.x, 0 };
            double w_pi_pi_plus = calcPossionKernel(rawPoints[i], Npi, Pplus[i], ADWeightAs[i]).first;
            double w_pi_pi_minus = calcPossionKernel(rawPoints[i], Npi, Pminus[i], ADWeightAs[i]).first;
            Vector3 grad_w_pk_pi_plus = calcPossionKernel(rawPoints[k], d_Npi, rawPoints[i], ADWeightAs[i]).second;
            case2_1 += (-w_pi_pi_plus + w_pi_pi_minus) * dot(Npi, grad_w_pk_pi_plus);
        }

        // case 2.2
        case2_2 += 0;
        grad_boundary(k) = case1_1 + case1_2 + case2_1 + case2_2;
    }
    std::cout << "**************************grad_boundary.norm = " << grad_boundary.norm() << "\n";


    for (int i = 0; i < x.size(); i++) {
        Vector3 nowN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Vector3 PminusN{ normals(i, 0), normals(i, 1), normals(i, 2) };
        Vector3 PplusN = -1.0 * PminusN;
        double grad_Pplus = 0, grad_Pminus = 0, grad_midTerm = 0;
        //p plus
        
        Vector3 dn_plus{ std::sin(x(i)), -std::cos(x(i)), 0 };
        grad_Pplus += calcPossionKernel(rawPoints[i], -dn_plus, Pplus[i], ADWeightAs[i]).first * (dot(grad_p[i], PplusN));

        /*
        grad_Pplus += calcPossionKernel(rawPoints[i], nowN, Pplus[i], ADWeightAs[i]).first * 
                                            (dot(calcPossionKernel(rawPoints[i], Vector3{ -nowN.y, nowN.x, 0 }, Pplus[i], ADWeightAs[i]).second * area, PplusN));
        */
        grad_Pplus += calcPossionKernel(rawPoints[i], nowN, Pplus[i], ADWeightAs[i]).first * (dot(grad_p[i], dn_plus));
        //p minus
        
        Vector3 dn_minus{ -std::sin(x(i)), std::cos(x(i)), 0 };
        grad_Pminus += calcPossionKernel(rawPoints[i], dn_minus, Pminus[i], ADWeightAs[i]).first * (dot(grad_p[i], PminusN));
        /*
        grad_Pminus += calcPossionKernel(rawPoints[i], nowN, Pminus[i], ADWeightAs[i]).first *
            (dot(calcPossionKernel(rawPoints[i], Vector3{ -nowN.y, nowN.x, 0 }, Pminus[i], ADWeightAs[i]).second * area, PminusN));
        */
        grad_Pminus += calcPossionKernel(rawPoints[i], nowN, Pminus[i], ADWeightAs[i]).first * (dot(grad_p[i], dn_minus));

        // mid term
        grad_midTerm += 10.0 * mid_grad(i);
        //


        //grad(i) = grad_Pplus + grad_Pminus + 10.0 * grad_midTerm;
        //grad(i) = /*15.0 * jump_grad(i)  +*/ 1000.0 * mid_grad(i);

        //grad(i) = dot(grad_p[i] , -dn_minus);
        //std::cout << "i = " << i << ", grad = " << grad(i) << "\n";
        /*
        std::cout << "grad_Pplus = " << grad_Pplus << ", grad_Pminus = " << grad_Pminus << ", poi= " << calcPossionKernel(rawPoints[i], nowN, Pplus[i], ADWeightAs[i]) <<
            ", r = " << dot(grad_p[i], dn_minus) << ", poi = " << calcPossionKernel(rawPoints[i], nowN, Pminus[i], ADWeightAs[i]) << "\n";
        */
        /*
        std::cout << "i = " << i << ", grad = " << grad(i) << ", l = " << dot(grad_p[i], dn_plus) << ", poi= " << calcPossionKernel(rawPoints[i], nowN, Pplus[i], ADWeightAs[i]) <<
                        ", r = " << dot(grad_p[i], dn_minus) << ", poi = " << calcPossionKernel(rawPoints[i], nowN, Pminus[i], ADWeightAs[i]) <<  "\n";
        */
    }
    //grad /= 10.0;
    //std::cout << "**************************enery 0.5 grad.norm = " << grad.norm() << "\n";
    
    grad =  ggg;
    grad += -1.0 * grad_boundary; // ！！！！！！！！！！！！！！！！！！！！！！！！！！！

    /*
    std::vector<Vector3> test_grad(sz * sz, Vector3{ 0, 0, 0 });
    std::vector<double> test_val(sz * sz, 0);
    {

        int specifyit = tot / 2;
        Vector3 Ns{ normals(specifyit, 0), normals(specifyit, 1), 0 };
        for (int i = 0; i < sz * sz; i++) {
            Vector3 q{ plane(i, 0), plane(i, 1), 0 };
            auto now = calcPossionKernel(rawPoints[specifyit], Ns, q, ADWeightAs[specifyit]);
            
            test_grad[i] = now.second;
            test_val[i] = now.first;
            if (test_grad[i].norm() > 1e5) {
                test_grad[i] = Vector3{0, 0, 0};
            }
            
            for (int j = 0; j < CPs[i].size(); j++) {

            }
            
        }
        polyscope::getPointCloud("plane")->addVectorQuantity("test_grad " + std::to_string(iter), test_grad);
        polyscope::getPointCloud("plane")->addScalarQuantity("test_val", test_val);
    }
    */
    
    //fx = -fx;
    
    //fx = DirichletEnery;
    //grad = grad_alldomain;
    //grad = FEM_d_theta;


    //fx = -fx;
    //grad = -grad;

    std::cout << "iter : " << iter << "\n";
    std::cout << "fx = " << fx << ", grad.norm = " << grad.norm() << ", totdiff = " << totdiff << "\n";
    std::cout << "DirichletEnery = " << DirichletEnery << "\n";
    std::cout << "DirichletEnery_deri = " << DirichletEnery_deri << "\n";


    polyscope::getPointCloud("plane")->addScalarQuantity("W " + std::to_string(iter), W);
    polyscope::getPointCloud("plane")->addVectorQuantity("grad " + std::to_string(iter), Grad_plane);

    polyscope::getPointCloud("plane")->addVectorQuantity("grad_deri " + std::to_string(iter), Grad_plane_deri);
    
    polyscope::getPointCloud("poly1")->addVectorQuantity("normal " + std::to_string(iter), normals);
    polyscope::getPointCloud("poly1")->addVectorQuantity("grad_p_plus " + std::to_string(iter), grad_p);
    polyscope::getPointCloud("poly1")->addVectorQuantity("grad_p_minus " + std::to_string(iter), grad_p_minus);
    polyscope::getPointCloud("poly1")->addScalarQuantity("W_p" + std::to_string(iter), midBalance + Eigen::VectorXd::Constant(midBalance.size(), 0.5));
    polyscope::getPointCloud("poly1")->addScalarQuantity("W_Pplus" + std::to_string(iter), W_plus );
    polyscope::getPointCloud("poly1")->addScalarQuantity("W_Pminus" + std::to_string(iter), W_minus);
    polyscope::getPointCloud("poly1")->addVectorQuantity("grad_direct" + std::to_string(iter), grad_p_direct);
    
    /*
    if(iter % 7 == 0){// hodge decom
                
        std::vector<float> vx, vy;
        {
            for (int i = 0; i < Grad_plane_deri.size(); i++) {
                vx.push_back((float)Grad_plane_deri[i].x);
                vy.push_back((float)Grad_plane_deri[i].y);
            }
        }

        VF gradientField(sz, sz, spacing, spacing, vx, vy);
        gradientField.RunHelmholtzDecomposition();
        std::vector<Vector3> nowd, nowr, nowh, calcfinal;
        double totDnorm = 0, totRnorm = 0, totHnorm = 0, totNoHnorm = 0;
        for (int i = 0; i < gradientField.d.sz; i++) {
            nowd.push_back(Vector3{ gradientField.d.u[i], gradientField.d.v[i], 0});
            nowr.push_back(Vector3{ gradientField.r.u[i], gradientField.r.v[i], 0 });
            nowh.push_back(Vector3{ gradientField.h.u[i], gradientField.h.v[i], 0 });
            calcfinal.push_back(Vector3{ gradientField.d.u[i] + gradientField.h.u[i], gradientField.d.v[i] + gradientField.h.v[i], 0 });
            totDnorm += nowd[i].norm();
            totRnorm += nowr[i].norm();
            totHnorm += nowh[i].norm();
            totNoHnorm += calcfinal[i].norm();
        }
        std::cout << "totDnorm = " << totDnorm << ", totRnorm = " << totRnorm << ", totHnorm = " << totHnorm << ", totNoHnorm = " << totNoHnorm << "\n";
        polyscope::getPointCloud("plane")->addVectorQuantity("d " + std::to_string(iter), nowd);
        polyscope::getPointCloud("plane")->addVectorQuantity("r " + std::to_string(iter), nowr);
        polyscope::getPointCloud("plane")->addVectorQuantity("h " + std::to_string(iter), nowh);
        polyscope::getPointCloud("plane")->addVectorQuantity("no r " + std::to_string(iter), calcfinal);
        
    }
    */
    

    iter++;
    return fx;
}

#include <igl/read_triangle_mesh.h>
void lwz_disk() {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh("../../../../data/disk.obj", V, F);
    polyscope::registerSurfaceMesh("disk", V, F);
}

#include "geometrycentral/surface/geodesic_centroidal_voronoi_tessellation.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

void show_Vor() {
    //Eigen::MatrixXd V;
    //Eigen::MatrixXi F;
    //igl::read_triangle_mesh("../../../../data/disk.obj", V, F);

    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
    std::tie(mesh, geometry) = readManifoldSurfaceMesh("../../../../data/disk.obj");
    //polyscope::registerSurfaceMesh("disk", V, F);
    std::cout << "here \n";
    VoronoiOptions opt;
    geometry->requireVertexPositions();

    VoronoiResult Vor = computeGeodesicCentroidalVoronoiTessellation(*mesh, *geometry, opt);
    std::cout << "here 2 \n";
    auto disk = polyscope::registerSurfaceMesh("disk", geometry->vertexPositions, mesh->getFaceVertexList());
    std::vector<VertexData<double>> siteDistributions = Vor.siteDistributions;
    std::vector<SurfacePoint> siteLocations = Vor.siteLocations;
    std::vector<Vector3> site3D;
    
    for (int i = 0; i < siteLocations.size(); i++) {
        Vector3 nowP = siteLocations[i].interpolate(geometry->vertexPositions);
        site3D.emplace_back(nowP);
    }
    polyscope::registerPointCloud("site", site3D);
    for (int i = 0; i < siteDistributions.size(); i++) {
        disk->addVertexScalarQuantity("site_" + std::to_string(i), siteDistributions[i]);
    }
}



void DataPre() {
    //int sz = 200;
    double spacing = 2.0 / (1.0 * sz);
    isThinkBand.resize(sz * sz + 5, -1);
    
    /*
    plane : 格点个数
    上面那个函数中的W，每个格点的值
    二维格点转一维，调用idx = getIndex（i, j）
    例如(x, y)， W(getIndex(x, y))
    */


    plane.resize(sz * sz, 3);
    {

        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                int idx = i * sz + j;
                plane(idx, 0) = -1.0 + 1.0 * i * spacing;
                plane(idx, 1) = -1.0 + 1.0 * j * spacing;
                plane(idx, 2) = 0;
            }
        }
    }
    std::cout << "begin calc thinkBand \n";

    int tot = rawPoints.size();
    std::cout << "tot = " << tot << "\n";
    CPs.resize(tot);
#pragma omp parallel for
    for (int i = 0; i < sz * sz; i++) {
        if (isThinkBand[i] != -1) continue;
        for (int j = 0; j < tot; j++) {
            double dist = (rawPoints[j].x - plane(i, 0)) * (rawPoints[j].x - plane(i, 0)) + (rawPoints[j].y - plane(i, 1)) * (rawPoints[j].y - plane(i, 1));
            dist = std::sqrt(dist);
            if (std::abs(dist) < 1.1 * spacing) {
                isThinkBand[i] = j;
             
            }
        }
    }
    for (int i = 0; i < sz * sz; i++) {
        if (isThinkBand[i] == -1) continue;

        for (int j = 0; j < tot; j++) {
            double dist = (rawPoints[j].x - plane(i, 0)) * (rawPoints[j].x - plane(i, 0)) + (rawPoints[j].y - plane(i, 1)) * (rawPoints[j].y - plane(i, 1));
            dist = std::sqrt(dist);
            if (std::abs(dist) < 1.1 * spacing) {
                CPs[j].emplace_back(i);
            }
        }
    }
    /*
    for (int i = 0; i < tot; i++) {
        std::cout << "i = " << i << "CP.size = " << CPs[i].size() << "\n";
    }
    */
    polyscope::registerPointCloud("plane", plane)->setPointRadius(0.001);
    lwz_disk();
}


void testshapeGen() {
    CircleGen2D O1(Vector3{ 0, 0, 0 }, 0.7, 200, 1, -1);
    //CircleGen2D O2;
    //CircleGen2D O3(Vector3{ 0, 0, 0 }, 0.65, 200, 0.75, -1);
    //CircleGen2D O4(Vector3{ 0, 0, 0 }, 0.5, 200, 0.75, -2);
    //polyscope::registerPointCloud("O1", O1.ps);
    //polyscope::registerPointCloud("O2", O2.ps);
    //polyscope::registerPointCloud("O3", O3.ps);
    //polyscope::registerPointCloud("O4", O4.ps);

    //polyscope::getPointCloud("O1")->addVectorQuantity("n1", O1.normals);
    //polyscope::getPointCloud("O2")->addVectorQuantity("n2", O2.normals);
    //polyscope::getPointCloud("O3")->addVectorQuantity("n3", O3.normals);
    //polyscope::getPointCloud("O4")->addVectorQuantity("n4", O4.normals);

    //printf("O1.weightA = %.6f, %.6f, %.6f, %.6f \n", O1.weightA, O2.weightA, O3.weightA, O4.weightA);
    
    /*
    Polygon2D poly1(std::vector<Vector3>{Vector3{-0.5, 0.5, 0}, 
                                            Vector3{ 0.5, 0.5, 0 },
                                            Vector3{ 0.25, 0, 0 },
                                            Vector3{ 0.5, -0.5, 0 },
                                            Vector3{ -0.5, -0.5, 0 },
                                            Vector3{ -0.25, 0, 0 }},
                                                        250,
                                                        1,
                                                        0);
    */
    /*
    Polygon2D poly1(std::vector<Vector3>    {Vector3{0, 0.2}, Vector3{ 0.7, 0.2 }, Vector3{ 0.7, 0 }, Vector3{ -0.7, 0 }, Vector3{ -0.7, 0.2 }},
        250,
        0,
        0);
    */
    Polygon2D poly1(std::vector<Vector3>    {Vector3{ -0.7, 0.4 }, Vector3{ 0.7, 0.4 }, Vector3{ 0.7, 0 }, Vector3{ -0.7, 0 }, Vector3{ -0.7, 0.1 }, Vector3{ 0.6, 0.1 }, Vector3{ 0.6, 0.3 }, Vector3{-0.7, 0.3}},
        250,
        0,
        0);
    
    

    LBFGSParam<double> param;  // New parameter class
    param.epsilon = 1e-6;
    param.max_iterations = 25;
    //param.max_linesearch = 10;
    // Create solver and function object
    LBFGSSolver<double> solver(param);  // New solver class
    // Bounds

    //Eigen::VectorXd lb = VectorXd::Constant(O1.ps.size(), -0.01);
    //Eigen::VectorXd ub = VectorXd::Constant(O1.ps.size(), 2.0 * PI + 0.01);
    //Eigen::VectorXd x(O1.ps.size());
    /*
    std::vector<Vector3> tmpps;
    Vector3 Center{ 0, 0, 0 };
    {
        std::string filename = "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\shape_A.txt";
        std::ifstream in;
        in.open(filename);
        int cnt = 0;
        for (std::string line; std::getline(in, line);) {
            std::istringstream lineStream(line);
            double x, y, z, nx, ny, nz;
            lineStream >> x >> y >> z ;
            tmpps.push_back(Vector3{ x, y, z });
            Center += Vector3{ x, y, z };
            cnt++;
        }
        Center /= 1.0 * cnt;
        double scale = 1.0;
        for (int i = 0; i < tmpps.size(); i++) {
            tmpps[i] -= Center;
            scale = std::max(1.0 * tmpps[i].norm(), scale);
        }
        for (int i = 0; i < tmpps.size(); i++) {
            tmpps[i] /= 1.2 * scale;
        }
        std::cout << "point cloud cnt = " << cnt << "\n";
        in.close();
    }
    */

    Eigen::VectorXd x(O1.ps.size());
    //Eigen::VectorXd x(tmpps.size());
    // data pre
    //rawPoints = O1.ps;
    rawPoints = O1.ps;
    //rawPoints = tmpps;
    std::cout << "rawPoints.size = " << rawPoints.size() << "\n";
    //ADWeightAs = O1.weightAs;
    ADWeightAs = O1.weightAs;
    std::fill(ADWeightAs.begin(), ADWeightAs.end(), 0.05);
    //
    DataPre();
    //
    std::cout << "over datapre \n";
    int rawPointsSZ = rawPoints.size();
    ADrawPoints.resize(rawPointsSZ * 3);
    for (int i = 0; i < rawPointsSZ; i++) {
        ADrawPoints(i * 3 + 0) = autodiff::real(rawPoints[i].x);
        ADrawPoints(i * 3 + 1) = autodiff::real(rawPoints[i].y);
        ADrawPoints(i * 3 + 2) = autodiff::real(rawPoints[i].z);

    }
    // end
    std::random_device rd;  // 将用于获得随机数引擎的种子
    std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0);
    /*
    for (int i = 0; i < poly1.allVs.size(); i++) {
        double u = dis(gen);
        x(i) = u;
        //x(i) = O1.thetas[i];
        
        x(i) = poly1.thetas[i];
        if (i > 126) x(i) = x(i) + PI;
    }
    */
    for (int i = 0; i < O1.ps.size(); i++) {
        double u = dis(gen);
        x(i) = u;
        x(i) = O1.thetas[i];

        //x(i) = poly1.thetas[i];
        //if (i > 126) x(i) = x(i) + PI;
    }
    //Eigen::VectorXd _(poly1.allVs.size());
    Eigen::VectorXd _(O1.ps.size());

    std::cout << "here ! \n";
    polyscope::registerPointCloud("poly1", O1.ps);
    igl::read_triangle_mesh("../../../../data/disk.obj", Vd, Fd);
    polyscope::registerSurfaceMesh("disk", Vd, Fd);
    //polyscope::getPointCloud("poly1")->addVectorQuantity("n1", poly1.allNormals);

    std::cout << "begin foo2 \n";
    foo2(x, _);

    /*
    double fx;
    int niter = solver.minimize(foo2, x, fx);

    std::cout << niter << " iterations" << std::endl;
    //std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    */
}


#include "test.h"
void test_sth() {
    //test3D::Datapremesh("../../../../data/108chair.obj");
    
    //test3D::DataprePointCloud("../../../../data/cowhead.xyz");    
    //test3D::readVDQs("../../../../data/cowhead_VDQS.txt");
    
    //test3D::DataprePointCloud("../../../../data/108chair.xyz");
    //test3D::readVDQs("../../../../data/108chair_ALLVDQueryPoints.txt");

    //test3D::DataprePointCloud("../../../../data/108chair6000poisson.xyz");
    
    //test3D::readVDQs("../../../../data/108chair6000poisson_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/bird7000.xyz");
    //test3D::readVDQs("../../../../data/bird7000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/bird_blue_7000.xyz");
    //test3D::DataprePointCloud("../../../../data/bird_blue_7000.xyz");
    //test3D::readVDQs("../../../../data/bird_blue_7000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/thinBand2.xyz");
    //test3D::readVDQs("../../../../data/thinBand2_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/dress_poisson6000.xyz");
    //test3D::readVDQs("../../../../data/dress_poisson6000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/WireFrame_trebol.xyz");
    //test3D::readVDQs("../../../../data/WireFrame_trebol_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/cup_poisson7000.xyz");
    //test3D::readVDQs("../../../../data/cup_poisson7000_VDQS.txt");


    //test3D::DataprePointCloud("../../../../data/skull_poisson6000.xyz");
    //test3D::readVDQs("../../../../data/skull_poisson6000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/genus250_poisson60000.xyz");
    //test3D::readVDQs("../../../../data/genus250_poisson60000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/BunnyPeel_blue_7000.xyz", "../../../../data/BunnyPeel_blue_7000_Areas_new.txt");
    //test3D::DataprePointCloud("../../../../data/xurui/BunnyPeel_blue_7000136_Double_End.xyz");
    //test3D::readVDQs("../../../../data/BunnyPeel_blue_7000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/linkCupTop_blue_7000.xyz");
    //test3D::DataprePointCloud("../../../../data/xurui/linkCupTop_blue_700081_Double_End.xyz");
    //test3D::readVDQs("../../../../data/linkCupTop_blue_7000_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/41glass.xyz");
    // 41glass_PCANormal
    //test3D::DataprePointCloud("../../../../data/41glass_PCANormal.xyz");
    //test3D::readVDQs("../../../../data/41glass_VDQS.txt");

    //397horse_blue_7000

    //test3D::DataprePointCloud("../../../../data/397horse_blue_7000.xyz", "../../../../data/397horse_blue_7000_Areas.txt");

    //test3D::readVDQs("../../../../data/397horse_blue_7000_VDQS.txt");

    //shellfish2_lr_blue_9200
    //test3D::DataprePointCloud("../../../../data/shellfish2_lr_blue_9200.xyz");
    //test3D::readVDQs("../../../../data/shellfish2_lr_blue_9200_VDQS.txt");

    //test3D::DataprePointCloud("../../../../data/kitten_blue_500_PCANormal.xyz");
    //test3D::readVDQs("../../../../data/kitten_blue_500_VDQS.txt");

    // Art-Institute-Chicago-Lion_blue_11000
    //test3D::DataprePointCloud("../../../../data/Art-Institute-Chicago-Lion_blue_11000.xyz");
    //test3D::DataprePointCloud("../../../../data/xurui/Art-Institute-Chicago-Lion_blue_11000205_Double_End.xyz");
    //test3D::DataprePointCloud("../../../../data/xs/result/Art-Institute-Chicago-Lion_blue_11000_real_4_70_modify.xyz");
    //test3D::DataprePointCloud("../../../../data/Art-Institute-Chicago-Lion_blue_11000_PCANormals.xyz");

    //test3D::readVDQs("../../../../data/Art-Institute-Chicago-Lion_blue_11000_VDQS.txt");

    //steampunk_gear_cube_small_corner_2_blue_11000
    //test3D::DataprePointCloud("../../../../data/steampunk_gear_cube_small_corner_2_blue_11000.xyz");
    //test3D::readVDQs("../../../../data/steampunk_gear_cube_small_corner_2_blue_11000_VDQS.txt");

    //WS0.5_4000_torus_End
    //test3D::DataprePointCloud("../../../../data/WS0.5_4000_torus_PCANormals.xyz");
    //test3D::readVDQs("../../../../data/WS0.5_4000_torus_VDQS.txt");

    // 108chair6000poisson_0.5
    //test3D::DataprePointCloud("../../../../data/noisydata/108chair6000poisson_0.5_PCANormal.xyz");
    //test3D::readVDQs("../../../../data/noisydata/108chair6000poisson_0.5_VDQS.txt");

    //108chair
    //test3D::DataprePointCloud("../../../../data/108chair.xyz");
    //test3D::addNoisy(0.005, "108chair_0.5");
    //test3D::readVDQs("../../../../data/108chair_VDQS.txt");

    //Art-Institute-Chicago-Lion_blue_11000_0.5
    //test3D::DataprePointCloud("../../../../data/noisydata/Art-Institute-Chicago-Lion_blue_11000_0.5.xyz");
    //test3D::addNoisy(0.005, "Art-Institute-Chicago-Lion_blue_11000_0.5");
    //test3D::readVDQs("../../../../data/noisydata/Art-Institute-Chicago-Lion_blue_11000_0.5_VDQS.txt");

    //30cup_blue_9000_PCANormal_0.5
    //test3D::DataprePointCloud("../../../../data/noisydata/30cup_blue_9000_PCANormal_0.5.xyz");
    //test3D::addNoisy(0.005, "30cup_blue_9000_PCANormal_0.5");
    //test3D::readVDQs("../../../../data/noisydata/30cup_blue_9000_PCANormal_0.5_VDQS.txt");//Arrayed_Vase_blue_12000_VDQS

    //candle_blue_10000_0.5
    //test3D::DataprePointCloud("../../../../data/candle_blue_10000.xyz");
    //test3D::addNoisy(0.005, "candle_blue_10000_0.5");

    //pulley_blue_13000
    //test3D::DataprePointCloud("../../../../data/pulley_blue_13000.xyz");
    //test3D::addNoisy(0.0075, "pulley_blue_13000_0.75");

    //vase_line_blue_11000
    //test3D::DataprePointCloud("../../../../data/vase_line_blue_11000.xyz");
    //test3D::addNoisy(0.0075, "vase_line_blue_11000_0.75");

    //botijo_blue_9000
    //test3D::DataprePointCloud("../../../../data/botijo_blue_9000.xyz");
    //test3D::addNoisy(0.0075, "botijo_blue_9000_0.75");

    {// zhouciyang's 
        Eigen::MatrixXd Vz;
        Eigen::MatrixXi Fz;
        igl::read_triangle_mesh("D:/liuweizhou/zhouciyang/example/crtpoints0.obj", Vz, Fz);
        polyscope::registerSurfaceMesh("zcy", Vz, Fz);

        std::cout << "Vz.rows = " << Vz.rows() << ", Fz.rows = " << Fz.rows() << "\n";

        int nFaces = Fz.rows();
        std::vector<std::array<double, 3>> fColor(nFaces);
        std::ifstream in;
        in.open("D:/liuweizhou/zhouciyang/example/rgb.txt");
        int cnt = 0;
        for (std::string line; std::getline(in, line);) {
            std::istringstream lineStream(line);
            double x, y, z, nx, ny, nz;
            lineStream >> x >> y >> z;
            fColor[cnt] = { x, y, z };
            cnt++;
        }
        std::cout << "cnt = " << cnt << "\n";
        polyscope::getSurfaceMesh("zcy")->addFaceColorQuantity("fcolor", fColor);
        std::cout << "point cloud cnt = " << cnt << "\n";
        in.close();

    }


    // show sth
    //test3D::DataprePointCloud("../../../../data/shape_As_Up_42.xyz");
    //test3D::DataprePointCloud("D:\\liuweizhou\\models\\famousShape\\deep_geometric_prior_data\\deep_geometric_prior_data\\xyz\\lord_quas.xyz");
    //D:\liuweizhou\models\famousShape\deep_geometric_prior_data\deep_geometric_prior_data\xyz\\lord_quas.xyz
    //test3D::readVDQs("../../../../data/108chair6000poisson_VDQS.txt");//
    //test3D::addNoisy(0.0075, "108chair6000poisson_0.75");

    //test3D::DataprePointCloud("../../../../data/xurui/108chair119_Double_End.xyz");
    

    //test3D::generateNearGrid();
    //test3D::run_nearGrid();
    //test3D::run_VDQS();
    //test3D::run_boundary();
    //test3D::run_nearPoints();
    //test3D::run();
    //test3D::getWNscalarFieldANDvector();
}

int main() {
    //testautodiff();
    //testLBFGSpp();
    
    polyscope::init();
    //show_Vor();
    testshapeGen();
    //test_sth();
    polyscope::state::userCallback = functionCallback;
    polyscope::show();
	return 0;
}
/*
{Vector3{ -0.7, 0, 0 },
Vector3{ -0.7, -0.7, 0 },
Vector3{ 0.7, -0.7, 0 },
Vector3{ 0.7, 0, 0 },
Vector3{ -0.65, 0, 0 },
Vector3{ -0.65, 0.7, 0 },
Vector3{ 0.65, 0.7, 0 },
Vector3{ 0.65, 0, 0 }}
//

*/