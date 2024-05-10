#include "utils.h"

using namespace geometrycentral;
namespace test3D {

#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"
#include <igl/read_triangle_mesh.h>
#include <igl/random_points_on_mesh.h>

//#include <nanoflann/nanoflann.hpp>

// number of vertices on the largest side
const int s = 100;
const double epsD = 1e-3;
const double eps = 1e-10;
const double INF = 1e12;

struct myVec3 {
    const double Veps = 1e-10;
    Vector3 V;
    myVec3() :V{0, 0, 0} {}
    myVec3(Vector3 vec3):V(vec3){}
    int Vsgn(double len) const {
        double diff = std::fabs(len);
        if (diff < Veps) return 0;
        else if (len >= Veps) return 1;
        else if (len <= -Veps) return -1;
    }
    bool operator ==(const myVec3& other) const {
        if ((V - other.V).norm() < Veps) {
            return true;
        }
        return false;
    }
    bool operator < (const myVec3& other) const {
        if (V == other.V)
            return false;
        if (Vsgn(V.x - other.V.x) == -1)
            return true;
        else if (Vsgn(V.x - other.V.x) == 0) {
            if (Vsgn(V.y - other.V.y) == -1) {
                return true;
            }
            else if (Vsgn(V.y - other.V.y) == 0) {
                if (Vsgn(V.z - other.V.z) == -1) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                return false;
            }
        }
        else {
            return false;
        }
    }
};
class myPointCloud {
public:
    std::vector<myVec3> points;
    // 返回数据集中的点的数量
    inline size_t kdtree_get_point_count() const
    {
        return points.size();
    }
    // 返回给定索引处的点的指定维度的值
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return points[idx].V.x;
        else if (dim == 1)
            return points[idx].V.y;
        else if (dim == 2)
            return points[idx].V.z;
        return 0.0;
    }
    // 估计数据集的边界框
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }

};

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
Eigen::MatrixXd Ps, Ns;
std::vector<Vector3> Vec3_Ps, Vec3_Ns;
std::vector<Vector3> randomNs;
std::vector<Vector3> nearPoints;

const double nearGrid_spacing = 0.01; // 0.008
const double nearGrid_radii = 4e-3;
const double gamma = 3e-3;
std::vector<Vector3> nearGrid;
std::vector<double> Dis_nearGrid;
std::vector<Vector3> thinBand;
std::vector<double> Dis_thinBand;

Eigen::VectorXd As;
Eigen::VectorXd As_xurui;
Eigen::VectorXd As_circle;

std::vector<Vector3> grad_Ps;

std::vector<Vector3> VDQS;
std::vector<std::vector<Vector3> > VD_Ps;
std::vector<std::vector<Vector3> > VDQS_neibor;
std::vector<double> Dis_VDQS;
std::vector<Vector3> far_VDQS;

autodiff::ArrayXreal ADPs, ADNs;
autodiff::ArrayXreal Areas;
autodiff::ArrayXreal ADxs;

Eigen::MatrixXd Pplus, Pminus;
Eigen::MatrixXd GV;
Eigen::RowVector3i res;
 

Eigen::VectorXd ws_space, ws_Pplus, ws_Pminus;
Eigen::MatrixXd grad_ws_space, grad_ws_Pplus, grad_ws_Pminus;

// Build octree
std::vector<std::vector<int > > O_PI;
Eigen::MatrixXi O_CH;
Eigen::MatrixXd O_CN;
Eigen::VectorXd O_W;
// build wn
Eigen::MatrixXd O_CM;
Eigen::VectorXd O_R;
Eigen::MatrixXd O_EC;

// knn k = 15
Eigen::MatrixXi knn_idx;

// scale
double scale = 0;
Vector3 Center{0, 0, 0};



int iter;

autodiff::ArrayXreal Eigen_to_AD(Eigen::VectorXd x) {
    autodiff::ArrayXreal ADx(x.size());
    for (int i = 0; i < x.size(); i++) {
        ADx(i) = (autodiff::real)(x(i));
    }
    return ADx;
}

autodiff::ArrayXreal Eigen_to_AD(Eigen::MatrixXd x) {
    assert(x.cols() == 3);
    autodiff::ArrayXreal ADx(x.rows() * 3);
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            ADx(i * 3 + j) = x(i, j);
        }
    }
    return ADx;
}

Eigen::MatrixXd AD_to_Eigen(autodiff::ArrayXreal x) {
    assert((int)x.size() % 3 == 0);
    int tot = (int)x.size() / 3;
    Eigen::MatrixXd EigenX(tot, 3);
    for (int i = 0; i < tot; i++) {
        for (int j = 0; j < 3; j++) {
            EigenX(i, j) = (double)x(i * 3 + j);
        }
    }
    return EigenX;
}

Eigen::VectorXd AD_to_Eigen_likesize(autodiff::ArrayXreal x) {
    int tot = x.size();
    Eigen::VectorXd EigenX(tot);
    for (int i = 0; i < tot; i++) {
        EigenX(i) = (double)x(i);
    }
    return EigenX;
}


std::pair<double, Vector3> W_pi_q(Vector3 p, Vector3 Np, Vector3 q, double WeightA) {
    // 返回 w 的值和 w 的梯度
    double val = 0;
    Vector3 G{ 0, 0, 0 };
    Vector3 L = (p - q);
    double up = dot(L, Np);
    double down = std::pow(L.norm2(), 1.5) + eps / (4.0 * PI);
    up *= WeightA;
    //up /= (4.0 * PI);
    down *= (4.0 * PI);
    val = up / down;
    double L2 = L.norm2();
    double downl = std::pow(L2, 2.5), downr = std::pow(L2, 1.5);
    double dx = -3.0 * L.x * (Np.x * L.x + Np.y * L.y + Np.z * L.z) * WeightA / (4.0 * PI * downl) + Np.x * WeightA / (4.0 * PI * downr);
    double dy = -3.0 * L.y * (Np.x * L.x + Np.y * L.y + Np.z * L.z) * WeightA / (4.0 * PI * downl) + Np.y * WeightA / (4.0 * PI * downr);
    double dz = -3.0 * L.z * (Np.x * L.x + Np.y * L.y + Np.z * L.z) * WeightA / (4.0 * PI * downl) + Np.z * WeightA / (4.0 * PI * downr);

    G = Vector3{ dx, dy, dz };
    return std::make_pair(val, G);
}


/*
void writePointCloudAsXYZ(std::pair<std::vector<Vector3>, std::vector<Vector3> > pairPN, std::string filepath) {
    std::ofstream out;
    out.open(filepath);

    std::vector<Vector3> ps = pairPN.first;
    std::vector<Vector3> normals = pairPN.second;

    for (int i = 0; i < ps.size(); i++) {
        double x, y, z, nx, ny, nz;
        x = ps[i].x;
        y = ps[i].y;
        z = ps[i].z;
        nx = normals[i].x;
        ny = normals[i].y;
        nz = normals[i].z;
        out << x << " " << y << " " << z << " " << nx << " " << ny << " " << nz << "\n";
    }
    out.close();
}
*/

void writePointCloudAsXYZEigen(std::pair<Eigen::MatrixXd, Eigen::MatrixXd > pairPN, std::string filepath) {
    std::ofstream out;
    out.open(filepath);

    Eigen::MatrixXd ps = pairPN.first;
    Eigen::MatrixXd normals = pairPN.second;

    for (int i = 0; i < ps.rows(); i++) {
        double x, y, z, nx, ny, nz;
        x = ps(i, 0);
        y = ps(i, 1);
        z = ps(i, 2);
        nx = normals(i, 0);
        ny = normals(i, 1);
        nz = normals(i, 2);
        out << x << " " << y << " " << z << " " << nx << " " << ny << " " << nz << "\n";
    }
    out.close();
}

void writePointCloudAsXYZVector(std::vector<Vector3> PC, std::vector<Vector3> PCN, std::string filepath) {
    std::ofstream out;
    out.open(filepath);
    for (int i = 0; i < PC.size(); i++) {
        double x, y, z, nx, ny, nz;
        x = PC[i].x;
        y = PC[i].y;
        z = PC[i].z;
        nx = PCN[i].x;
        ny = PCN[i].y;
        nz = PCN[i].z;
        out << x << " " << y << " " << z << " " << nx << " " << ny << " " << nz << "\n";
    }
    out.close();
}

void writeuvs(Eigen::VectorXd xs, std::string filepath) {
    std::ofstream out;
    out.open(filepath);
    for (int i = 0; i < xs.size(); i+=2) {
        double u = xs(i);
        double v = xs(i + 1);
        out << u << " " << v << "\n";
    }
    out.close();
}

Eigen::VectorXd readuvs(std::string filepath) {
    std::ifstream in;
    in.open(filepath);
    int cnt = 0;
    std::vector<double> uvs;
    for (std::string line; std::getline(in, line);) {
        std::istringstream lineStream(line);
        double u, v;
        lineStream >> u >> v;
        uvs.emplace_back(u);
        uvs.emplace_back(v);
        cnt++;
    }
    in.close();
    Eigen::VectorXd Euvs = Eigen::Map<Eigen::VectorXd>(uvs.data(), uvs.size());
    return Euvs;
}

std::pair<std::vector<Vector3>, std::vector<Vector3> > readPointCloud_and_Normals(std::string filename) {
    std::vector<Vector3> tmpps, tmpNormals;
    std::ifstream in;
    in.open(filename);
    int cnt = 0;
    for (std::string line; std::getline(in, line);) {
        std::istringstream lineStream(line);
        double x, y, z, nx, ny, nz;
        lineStream >> x >> y >> z >> nx >> ny >> nz;
        tmpps.push_back(Vector3{ x, y, z });
        tmpNormals.push_back(Vector3{ nx, ny, nz });
        Center += Vector3{ x, y, z };
        cnt++;
    }
    std::cout << "point cloud cnt = " << cnt << "\n";
    in.close();
    Center /= (1.0 * cnt);
    
    for (int i = 0; i < tmpps.size(); i++) {
        tmpps[i] = tmpps[i] - Center;
        //tmpps[i] = tmpps[i];
        scale = std::max(1.0 * tmpps[i].norm(), scale);
    }
    std::cout << "scale = " << scale << "\n";
    //scale = scale * 1.2;
    scale *= 0.9;
    //scale *= 0.5; // for fish
    //scale = scale * 0.1; // steampunk_gear_cube_small_corner_2_blue_11000
    for (int i = 0; i < tmpps.size(); i++) {
        tmpps[i] /= scale;
    }

    VD_Ps.resize(cnt);

    polyscope::registerPointCloud("init point clouds", tmpps);
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("init normal", tmpNormals);
    return std::make_pair(tmpps, tmpNormals);
}

/*
autodiff::real calc_w_p_q(autodiff::ArrayXreal P_flatten,
                            autodiff::ArrayXreal N_flatten,
                            autodiff::ArrayXreal Q_flatten,
                            autodiff::real Aera) {
    // 求解w_p_i(q)
    Eigen::MatrixXd EigenP = AD_to_Eigen(P_flatten);
    Eigen::MatrixXd EigenN = AD_to_Eigen(N_flatten);
    Eigen::MatrixXd EigenQ = AD_to_Eigen(Q_flatten);
    Eigen::VectorXd EigenA(1);
    EigenA(0) = (double)(Aera);

    Eigen::MatrixXd O_CM;
    Eigen::VectorXd O_R;
    Eigen::MatrixXd O_EC;
    igl::fast_winding_number(EigenP, EigenN, EigenA, O_PI, O_CH, 2, O_CM, O_R, O_EC);
    Eigen::VectorXd W;
    igl::fast_winding_number(EigenP, EigenN, EigenA, O_PI, O_CH, O_CM, O_R, O_EC, EigenQ, 2, W);
    assert(W.size() == EigenQ.rows());
    assert(W.size() == 1);

    return (autodiff::real)W(0);
}
*/

#include <nanoflann/nanoflann.hpp>

std::vector<Vector3> readVDQs(std::string filepath) {
    std::ifstream in;
    in.open(filepath);
    int cnt = 0;
    std::set<Vector3> set_VDQS;
    int flag = 0;
    for (std::string line; std::getline(in, line);) {
        std::istringstream lineStream(line);
        double x, y, z;
        int num;
        if (!flag) {
            lineStream >> num;
            flag = num;
            cnt++;
        }
        else {
            flag--;
            lineStream >> x >> y >> z;
            Vector3 now{ x, y ,z };
            now = (now - Center) / scale;
            VDQS.push_back(now);
            VD_Ps[cnt - 1].emplace_back(now);
        }

        //set_VDQS.insert(Vector3{ x, y, z });
        
    }
    std::cout << "cnt = " << cnt << "\n";//", real cnt = " << set_VDQS.size() << "\n";
    std::cout << "VD_Ps.size = " << VD_Ps.size() << "\n";
    int _ = 0;
    for (int i = 0; i < VD_Ps.size(); i++) {
        _ += VD_Ps[i].size();
    }
    std::cout << "sum VD_PS = " << _ << "\n";
    /*
    std::for_each(set_VDQS.begin(), set_VDQS.end(), [&](const auto& element) {
        //std::cout << element << " ";
        VDQS.push_back(element);
    });
    */
    // 去重 VDQS
    {
        std::vector<Vector3> _VDQS;
        myPointCloud myCloud;
        std::map<myVec3, bool> map_VDQS;

        for (int i = 0; i < VDQS.size(); i++) {
            myVec3 nowP(VDQS[i]);
            if (map_VDQS.count(nowP)) continue;
            map_VDQS[nowP] = true;
            _VDQS.emplace_back(VDQS[i]);
        }
        VDQS.clear();
        VDQS = _VDQS;
        Dis_VDQS.resize(VDQS.size());
        for (int i = 0; i < Vec3_Ps.size(); i++) {
            myCloud.points.emplace_back(Vec3_Ps[i]);
        }
        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, myPointCloud>,
            myPointCloud,
            3 /* dim */
        >
            my_kd_tree_t;

        my_kd_tree_t index(3, myCloud);

        // 加载数据到kd树
        index.buildIndex();
        std::vector<size_t> indices(1);
        std::vector<double> distances(1);
        for (int i = 0; i < VDQS.size(); i++) {
            myVec3 nowP(VDQS[i]);
            index.knnSearch(&nowP.V.x, 1, &indices[0], &distances[0]);
            Dis_VDQS[i] = distances[0];
            if (Dis_VDQS[i] >= 1.5e-3 && Dis_VDQS[i] < 5e-3) { // gamma
                far_VDQS.emplace_back(VDQS[i]);
            }
        }
        
    }
    
    

    polyscope::registerPointCloud("far_VDQS", far_VDQS);
    polyscope::getPointCloud("far_VDQS")->setPointRadius(0.001);
    polyscope::registerPointCloud("VDQS", VDQS);
    polyscope::getPointCloud("VDQS")->setPointRadius(0.001);
    polyscope::getPointCloud("VDQS")->addScalarQuantity("distance", Dis_VDQS);
    in.close();
    return VDQS;
}

void build_VDQS_knn() {
    VDQS_neibor.resize(Vec3_Ps.size());
    myPointCloud myCloud;
    for (int i = 0; i < VDQS.size(); i++) {
        myCloud.points.emplace_back(VDQS[i]);
    }

}

void DataprePointCloud(std::string filepath, std::string AreaFilePath = "") {
    auto pair_ps_and_ns = readPointCloud_and_Normals(filepath);
    int totPoints = pair_ps_and_ns.first.size();

    Vec3_Ps = pair_ps_and_ns.first;
    Vec3_Ns = pair_ps_and_ns.second;

    Ps.resize(totPoints, 3);
    Ns.resize(totPoints, 3);

    for (int i = 0; i < totPoints; i++) {
        Ps(i, 0) = pair_ps_and_ns.first[i].x;
        Ps(i, 1) = pair_ps_and_ns.first[i].y;
        Ps(i, 2) = pair_ps_and_ns.first[i].z;

        Ns(i, 0) = pair_ps_and_ns.second[i].x;
        Ns(i, 1) = pair_ps_and_ns.second[i].y;
        Ns(i, 2) = pair_ps_and_ns.second[i].z;
    }
    igl::voxel_grid(Ps, 0, s, 1, GV, res);
    {
        ws_space.resize(GV.rows());
        grad_ws_space.resize(GV.rows(), 3);
    }
    //polyscope::registerPointCloud("space points", GV);
    //polyscope::getPointCloud("space points")->setPointRadius(0.001);

    igl::octree(Ps, O_PI, O_CH, O_CN, O_W);
    std::cout << "here 1.5 \n";

    {
        Eigen::MatrixXi I;
        igl::knn(Ps, 15, O_PI, O_CH, O_CN, O_W, I); // 15, 10
        // CGAL is only used to help get point areas
        igl::copyleft::cgal::point_areas(Ps, I, Ns, As);
        for (int i = 0; i < totPoints; i++) {
            if (i % 200 == 0) {
                std::cout << "i = " << i << ", As = " << As(i) << "\n";
            }

        }
        knn_idx = I;
        std::cout << "I.rows = " << I.rows() << ", I.cols = " << I.cols() << "\n";
        As_xurui = As;
        if (AreaFilePath != "") {
            std::ifstream in;
            in.open(AreaFilePath);
            int idx = 0;
            As_xurui = As;
            for (std::string line; std::getline(in, line);) {
                std::istringstream lineStream(line);
                double _x;
                lineStream >> _x;
                As_xurui(idx) = _x / scale;
                idx++;
            }
            in.close();
            for (int i = 0; i < totPoints; i++) {
                if (i % 200 == 0) {
                    std::cout << "i = " << i << ", As_xurui = " << As_xurui(i) << "\n";
                }

            }
        }
        

    }
    std::cout << "DataprePointCloud \n";
}

void getRightPole() {
    std::vector<std::vector<int> > knn_PC(Vec3_Ps.size());
    myPointCloud myCloud;
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        myCloud.points.emplace_back(Vec3_Ps[i]);
    }
    // 创建kd树
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, myPointCloud>,
        myPointCloud,
        3 /* dim */
    >
        my_kd_tree_t;

    my_kd_tree_t index(3, myCloud);

    // 加载数据到kd树
    index.buildIndex();

    // 执行最邻近点搜索
    int k_nearst = 8;
    std::vector<size_t> indices(k_nearst);
    std::vector<double> distances(k_nearst);
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        myVec3 nowP(Vec3_Ps[i]);
        index.knnSearch(&nowP.V.x, k_nearst, &indices[0], &distances[0]);
        for (int j = 0; j < k_nearst; j++) {
            knn_PC[i].emplace_back((int)indices[j]);
        }
        
    }
    std::vector<Vector3> poles;
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        double finalWeight = INF;
        Vector3 nowPole{ 0, 0, 0 };
        for (Vector3 q : VD_Ps[i]) {
            Vector3 vec = (q - Vec3_Ps[i]).normalize();
            double weight = 0;
            for (auto j : knn_PC[i]) {
                if (j == i) continue;
                //std::cout << Vec3_Ps[i] << ", " << Vec3_Ps[j] << "\n";
                double _ = dot((Vec3_Ps[j] - Vec3_Ps[i]).normalize(), vec);
                weight += _ * _;
            }
            //std::cout << "final weight = " << finalWeight << ", weight = " << weight << "\n";
            if (finalWeight > weight) {
                finalWeight = weight;
                nowPole = vec;
                //std::cout << "weight = " << weight << "\n";
                //std::cout << "vec = " << vec << "\n";
            }
        }
        //poles.emplace_back(nowPole);
        poles.emplace_back(Vec3_Ns[i]);
    }
    std::ofstream out;
    out.open("../../../../data/BunnyPeel_blue_7000_poles.txt");
    for (int i = 0; i < poles.size(); i++) {
        double x, y, z;
        x = poles[i].x;
        y = poles[i].y;
        z = poles[i].z;
        out << x << " " << y << " " << z << "\n";
    }
    out.close();
}

Eigen::VectorXd getRandomNs(int totPoints) {
    std::random_device rd;  // 将用于获得随机数引擎的种子
    std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0), dis3(PI / 2.0, 3.0 * PI / 2.0);

    Eigen::VectorXd uvs(totPoints * 2);

    for (int i = 0; i < totPoints; i++) {
        double u = dis(gen), v = dis(gen);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        randomNs.emplace_back(nowN);
        uvs(i * 2 + 0) = u;
        uvs(i * 2 + 1) = v;
    }
    return uvs;
}

double generateGaussianNoise(double mu, double sigma) {
    //定义小值
    const double epsilon = std::numeric_limits<double>::min();
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量X
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量
    double PI = std::acos(-1.0);
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2 * PI * u2);
    return z0 * sigma + mu;
}

void addNoisy(double level = 0.0025, std::string modelname = "") {
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        Vector3 deltaD{ generateGaussianNoise(0, 1.0), generateGaussianNoise(0, 1.0), generateGaussianNoise(0, 1.0) };
        deltaD = deltaD * level;
        Vec3_Ps[i] += deltaD;
    }
    polyscope::registerPointCloud("noisy ps", Vec3_Ps);
    writePointCloudAsXYZVector(Vec3_Ps, Vec3_Ns, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\noisydata\\" + modelname + ".xyz");
}

void getNearPoints(std::vector<Vector3> initPs) {
    int totPoints = initPs.size();
    nearPoints.resize(totPoints);
    for (int i = 0; i < totPoints; i++) {
        double cao1 = generateGaussianNoise(0.0, 0.2);
        double cao2 = generateGaussianNoise(0.0, 0.2);
        double cao3 = generateGaussianNoise(0.0, 0.2);
        Vector3 noisy = { cao1, cao2, cao3 };
        nearPoints[i] = initPs[i] + 0.1 * noisy;
    }
}

std::pair<double, double>  parameterizeVec3(Vector3 vec3) {
    // n = (sin u * cos v, sin u sin v, cos u)
    vec3 = vec3.normalize();
    double u = std::acos(vec3.z);
    double sinu = std::sin(u), cosu = std::cos(u);
    //Vector3 tmpV {vec3.x }
    double v = std::acos(vec3.x / sinu);
    double sinv = std::sin(v), cosv = std::cos(v);
    Vector3 _{ sinu * cosv, sinu * sinv, cosu };
    if ((_ - vec3).norm() >= 1e-6) {
        std::cout << "vec3 = " << vec3 << ", _ = " << _ << "\n";
    }
    assert((_ - vec3).norm() < 1e-6);
    return std::make_pair(u, v);
}

std::pair<double, double>  V3toV2(Eigen::Vector3d nor)
{
    double u = 0.0, v = 0.0;
    u = acos(nor.z());
    if (u == 0)
    {
        v = 0;
    }
    else
    {
        double tmp1 = abs(acos(nor.x() / sin(acos(nor.z()))));
        double tmp2 = abs(asin(nor.y() / sin(acos(nor.z()))));
        if (isnan(tmp1))
        {
            if (isnan(tmp2))
            {
                v = 0.0;
                std::pair<double, double> p(u, v);
                return p;
            }
            else
            {
                v = tmp2;
                std::pair<double, double> p(u, v);
                return p;
            }

        }
        if (isnan(tmp2))
        {
            if (isnan(tmp1))
            {
                v = 0.0;
                std::pair<double, double> p(u, v);
                return p;
            }
            else
            {
                v = tmp1;
                std::pair<double, double> p(u, v);
                return p;
            }
        }
        Eigen::Vector3d n11(sin(u) * cos(tmp1), sin(u) * sin(tmp1), cos(u));
        Eigen::Vector3d n22(sin(u) * cos(tmp2), sin(u) * sin(tmp2), cos(u));
        double tot1, tot2;
        tot1 = (n11 - nor).x() * (n11 - nor).x() + (n11 - nor).y() * (n11 - nor).y() + (n11 - nor).z() * (n11 - nor).z();
        tot2 = (n22 - nor).x() * (n22 - nor).x() + (n22 - nor).y() * (n22 - nor).y() + (n22 - nor).z() * (n22 - nor).z();
        if (tot1 < tot2)
        {
            v = tmp1;
        }
        else
        {
            v = tmp2;
        }
        if (abs(sin(u) * cos(v) - nor.x()) > 0.1)
        {
            u = -1.0 * u;
        }
        if (abs(sin(u) * sin(v) - nor.y()) > 0.1)
        {
            v = -1.0 * v;
        }
    }
    std::pair<double, double> p(u, v);
    return p;
}


std::vector<Vector3> uniquePlusMinus(std::vector<Vector3> Pplus, std::vector<Vector3> Pminus) {
    std::map<myVec3, bool> _mp;
    for (auto now : Pplus) {
        myVec3 _(now);
        _mp[_] = true;
    }
    for (auto now : Pminus) {
        myVec3 _(now);
        _mp[_] = true;
    }
    std::vector<Vector3> res;
    for (auto _ : _mp) {
        res.emplace_back(_.first.V);
    }
    return res;
}

std::map<myVec3, bool> hasVec3;

void generateNearGrid() {
    std::cout << "begin generate near grid \n";
    myPointCloud myCloud;
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        myCloud.points.emplace_back(Vec3_Ps[i]);
    }
    // 创建kd树
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, myPointCloud>,
        myPointCloud,
        3 /* dim */
    >
    my_kd_tree_t;

    my_kd_tree_t index(3, myCloud);

    // 加载数据到kd树
    index.buildIndex();
    myVec3 stP(Vector3{ myCloud.points[0].V.x + nearGrid_spacing / 2.0,
                                    myCloud.points[0].V.y + nearGrid_spacing / 2.0 ,
                                    myCloud.points[0].V.z + nearGrid_spacing / 2.0 });
    nearGrid.emplace_back(stP.V);
    // 执行最邻近点搜索
    std::vector<size_t> indices(1);
    std::vector<double> distances(1);

    std::vector<size_t> indices_2(2);
    std::vector<double> distances_2(2);

    As_circle = As;
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        myVec3 nowP(Vec3_Ps[i]);
        index.knnSearch(&nowP.V.x, 2, &indices_2[0], &distances_2[0]);
        if (indices_2[0] == i) {
            As_circle(i) = PI * distances_2[1] * distances_2[1] / (4.0);
        }
        else {
            As_circle(i) = PI * distances_2[0] * distances_2[0] / 4.0;
        }
    }
    std::cout << "circle areas \n";
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        if (i % 200 == 0) {
            std::cout << "i = " << i << ", As_circle = " << As_circle(i) << "\n";
        }

    }

    index.knnSearch(&stP.V.x, 1, &indices[0], &distances[0]);
    Dis_nearGrid.emplace_back(distances[0]);
    std::queue<myVec3> que;
    que.push(stP);
    std::cout << "begin bfs \n";
    while (que.size() > 0) {
        myVec3 nowP = que.front();
        myVec3 nxtP = nowP;
        que.pop();
        for (int x = -1; x <= 1; x += 2) {
            for (int y = -1; y <= 1; y += 2) {
                for (int z = -1; z <= 1; z += 2) {
                    Vector3 dir{ 1.0 * x, 1.0 * y, 1.0 * z };
                    dir *= nearGrid_spacing;
                    nxtP.V = nowP.V + dir;
                    if (hasVec3.count(nxtP))
                        continue;

                    index.knnSearch(&nxtP.V.x, 1, &indices[0], &distances[0]);
                    if (distances[0] < nearGrid_radii) {
                        hasVec3[nxtP] = true;
                        nearGrid.emplace_back(nxtP.V);
                        Dis_nearGrid.emplace_back(distances[0]);
                        que.push(nxtP);
                    }
                }
            }
        }
    }
    std::cout << "end bfs \n";
    std::cout << "nearGrid cnt = " << nearGrid.size() << "\n";
    for (int i = 0; i < nearGrid.size(); i++) {
        if (Dis_nearGrid[i] < gamma) continue;
        thinBand.emplace_back(nearGrid[i]);
        Dis_thinBand.emplace_back(Dis_nearGrid[i]);
    }
    int tot = 0;
    std::vector<Vector3> fuck;
    for (int i = 0; i < nearGrid.size(); i++) {
        if (Dis_nearGrid[i] < 2e-3) {
            fuck.emplace_back(nearGrid[i]);
        }
    }
    std::cout << "thinBand.size = " << thinBand.size() << "\n";

    polyscope::registerPointCloud("thinBand", thinBand);
    polyscope::getPointCloud("thinBand")->addScalarQuantity("distance", Dis_thinBand);
    polyscope::getPointCloud("thinBand")->setPointRadius(0.001);
    /*
    polyscope::registerPointCloud("near grid points", nearGrid);
    polyscope::getPointCloud("near grid points")->addScalarQuantity("distance", Dis_nearGrid);
    polyscope::getPointCloud("near grid points")->setPointRadius(0.001);
    */
}

Vector3 FEM_grad_w_pi_q(Vector3 p, Vector3 Np, Vector3 q, double areaP) {
    // f(x + h, y, z) - f(x - h, y, z) / 2h
    // f(x, y + h, z) - f(x, y - h, z) / 2h
    // f(x, y, z + h) - f(x, y, z - h) / 2h
    Vector3 grad{ 0, 0, 0 };
    double h = 0.001;
    for (int i = 0; i < 3; i++) {
        Vector3 d{ 0, 0, 0 };
        d[i] = h;
        Vector3 ql = q - d, qr = q + d;
        double w_l = W_pi_q(p, Np, ql, areaP).first;
        double w_r = W_pi_q(p, Np, qr, areaP).first;
        grad[i] = (w_r - w_l);
    }
    return grad;
}

//void build_thinBand
 
#include <LBFGS.h>
using namespace LBFGSpp;

double func(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    /*
    * x : 2 * n: (u1,v1, u2, v2 , ……)
    * grad : 2 * n
    */
    std::cout << "iter = ********************" << iter << "\n";
    const double epsD = 0.1;
    int totPoints = Vec3_Ps.size();
    std::vector<Vector3> grad_p_direct(totPoints);

    std::vector<Vector3> nowNs(totPoints);
    for (int i = 0; i < totPoints; i++) {

        double u = x(i * 2 + 0);
        double v = x(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = nowN.normalize();
    }

    std::vector<Vector3> Pplus(totPoints), Pminus(totPoints);
    Eigen::VectorXd W_plus(totPoints), W_minus(totPoints);
    for (int i = 0; i < totPoints; i++) {
        Vector3 nowN = nowNs[i];
        Pplus[i] = Vec3_Ps[i] + epsD * nowN;
        Pminus[i] = Vec3_Ps[i] - epsD * nowN;
    }
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 qPlus = Pplus[i], qMinus = Pminus[i];
        double _valplus = 0, _valminus = 0;
        for (int j = 0; j < totPoints; j++) {
            _valplus += W_pi_q(Vec3_Ps[j], nowNs[j], qPlus, As(j)).first;
            _valminus += W_pi_q(Vec3_Ps[j], nowNs[j], qMinus, As(j)).first;
        }
        W_plus(i) = _valplus;
        W_minus(i) = _valminus;
    }

    double boundary_diri = 0, totE = 0, totCharge = 0;
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 tmp{ 0, 0, 0 };
        for (int j = 0; j < totPoints; j++) {
            //if (i == j) continue;
            
            if ((Vec3_Ps[i] - Vec3_Ps[j]).norm2() < 1e-4) {
                continue;
            }
            
            tmp += W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j)).second;
        }
        grad_p_direct[i] = tmp;
        boundary_diri += (-W_plus[i] + W_minus[i]) * dot(nowNs[i], tmp) * As(i);
        totE += tmp.norm2();

        totCharge += dot(nowNs[i], tmp) * As(i);
    }





    Eigen::VectorXd grad_boundary(totPoints * 2), grad_totE(totPoints * 2);
    Eigen::VectorXd grad_totNearE(totPoints * 2);

    // boundary grad
#pragma omp parallel for
    for (int k = 0; k < totPoints; k++) {
        double u = x(k * 2 + 0);
        double v = x(k * 2 + 1);
        double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
        Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

        Vector3 d_u_nowNk{cosu * cosv, cosu * sinv, -sinu};
        Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

        double w_pk_pkplus = W_pi_q(Vec3_Ps[k], nowNk, Pplus[k], As(k)).first;
        double w_pk_pkminus = W_pi_q(Vec3_Ps[k], nowNk, Pminus[k], As(k)).first;

        double d_u_w_pk_pkplus = W_pi_q(Vec3_Ps[k], d_u_nowNk, Pplus[k], As(k)).first;
        double d_u_w_pk_pkminus = W_pi_q(Vec3_Ps[k], d_u_nowNk, Pminus[k], As(k)).first;

        double d_v_w_pk_pkplus = W_pi_q(Vec3_Ps[k], d_v_nowNk, Pplus[k], As(k)).first;
        double d_v_w_pk_pkminus = W_pi_q(Vec3_Ps[k], d_v_nowNk, Pminus[k], As(k)).first;

        double case1_1_u = 0, case1_2_u = 0, case2_1_u = 0, case2_2_u = 0;
        double case1_1_v = 0, case1_2_v = 0, case2_1_v = 0, case2_2_v = 0;
        // case 1.1
        Vector3 grad_pk_pkplus = W_pi_q(Vec3_Ps[k], nowNs[k], Pplus[k], As(k)).second;
        Vector3 d_u_grad_pk_pkplus = W_pi_q(Vec3_Ps[k], d_u_nowNk, Pplus[k], As(k)).second;
        Vector3 d_v_grad_pk_pkplus = W_pi_q(Vec3_Ps[k], d_v_nowNk, Pplus[k], As(k)).second;
        // d_u
        case1_1_u += ( (-d_u_w_pk_pkplus + d_u_w_pk_pkminus) * dot(nowNk, grad_pk_pkplus) +
                        (-w_pk_pkplus + w_pk_pkminus) * dot(d_u_nowNk, grad_pk_pkplus) +
                        (-w_pk_pkplus + w_pk_pkminus) * dot(nowNk, d_u_grad_pk_pkplus) ) * As(k);
        

        case1_1_v += ( (-d_v_w_pk_pkplus + d_v_w_pk_pkminus) * dot(nowNk, grad_pk_pkplus) +
                        (-w_pk_pkplus + w_pk_pkminus) * dot(d_v_nowNk, grad_pk_pkplus) +
                        (-w_pk_pkplus + w_pk_pkminus) * dot(nowNk, d_v_grad_pk_pkplus) ) * As(k);
        // d_v
        // case 1.2
        Vector3 sigma_grad{ 0, 0, 0 };
        for (int i = 0; i < totPoints; i++) {
            //if (i == k) continue;
            
            if ((Vec3_Ps[k] - Vec3_Ps[i]).norm2() < 1e-3) {
                continue;
            }
            
            sigma_grad += W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_Ps[k], As(i)).second;
        }
        // d_u
        case1_2_u += ((-d_u_w_pk_pkplus + d_u_w_pk_pkminus) * dot(nowNk, sigma_grad) +
                        (-w_pk_pkplus + w_pk_pkminus) * dot(d_u_nowNk, sigma_grad) ) * As(k);
        // d_v
        case1_2_v += ((-d_v_w_pk_pkplus + d_v_w_pk_pkminus) * dot(nowNk, sigma_grad) +
                        (-w_pk_pkplus + w_pk_pkminus) * dot(d_v_nowNk, sigma_grad) ) * As(k);
        // case 2.1
        for (int i = 0; i < totPoints; i++) {
            //if (i == k) continue;
            
            if ((Vec3_Ps[k] - Vec3_Ps[i]).norm2() < 1e-3) {
                continue;
            }
            
            // d_u
            auto _d_u = W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[i], As(k)).second;
            case2_1_u += ( (-w_pk_pkplus + w_pk_pkminus) * dot(nowNs[i], _d_u) ) * As(i);
            // d_v
            auto _d_v = W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[i], As(k)).second;
            case2_1_v += ( (-w_pk_pkplus + w_pk_pkminus) * dot(nowNs[i], _d_v) ) * As(i);
        }
        // case 2.2

        grad_boundary(k * 2 + 0) = (case1_1_u + case1_2_u + case2_1_u + case2_2_u);
        grad_boundary(k * 2 + 1) = (case1_1_v + case1_2_v + case2_1_v + case2_2_v);
    }

    // totE grad
#pragma omp parallel for
    for (int k = 0; k < totPoints; k++) {
        double u = x(k * 2 + 0);
        double v = x(k * 2 + 1);
        double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
        Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

        Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
        Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

        Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

        double d_u = 0, d_v = 0;

        for (int j = 0; j < totPoints; j++) {
            //if (k == j) continue;

            if ((Vec3_Ps[k] - Vec3_Ps[j]).norm2() < 1e-3) {
                continue;
            }

            d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[j], As(k)).second;
            d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[j], As(k)).second;

            d_u += 2.0 * dot(grad_p_direct[j], W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[j], As(k)).second);
            d_v += 2.0 * dot(grad_p_direct[j], W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[j], As(k)).second);

        }
        grad_totE(k * 2 + 0) = d_u;

        grad_totE(k * 2 + 1) = d_v;
    }


    int totNearPoints = nearPoints.size();
    std::vector<Vector3> grad_nearPoints(totNearPoints);
    double totnearPointsE = 0;
    // near points
    {

#pragma omp parallel for
        for (int i = 0; i < totNearPoints; i++) {
            Vector3 grad_i{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                if ((Vec3_Ps[j] - nearPoints[i]).norm2() < 1e-4) continue;
                grad_i += W_pi_q(Vec3_Ps[j], nowNs[j], nearPoints[i], As(j)).second;
            }
            grad_nearPoints[i] = grad_i;
            totnearPointsE += grad_i.norm2();
        }
        // totnearPointsE grad
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

            double d_u = 0, d_v = 0;

            for (int j = 0; j < totNearPoints; j++) {
                if ((Vec3_Ps[k] - nearPoints[j]).norm2() < 1e-4) continue;
                d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, nearPoints[j], As(k)).second;
                d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, nearPoints[j], As(k)).second;

                d_u += 2.0 * dot(grad_nearPoints[j], W_pi_q(Vec3_Ps[k], d_u_nowNk, nearPoints[j], As(k)).second);
                d_v += 2.0 * dot(grad_nearPoints[j], W_pi_q(Vec3_Ps[k], d_v_nowNk, nearPoints[j], As(k)).second);

            }
            grad_totNearE(k * 2 + 0) = d_u;

            grad_totNearE(k * 2 + 1) = d_v;
        }
    }
    // end near points


    // mid 0.5
    double totMid = 0;
    Eigen::VectorXd W_midTerm(totPoints);
    Eigen::VectorXd grad_midTerm(totPoints * 2);
    {
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            double W_i = 0;
            for (int j = 0; j < totPoints; j++) {
                auto nowW = W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j));
                W_i += nowW.first;
            }
            W_midTerm(i) = W_i;
            totMid += (W_i - 0.5) * (W_i - 0.5);
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {

            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            double d_u = 0, d_v = 0;
            for (int i = 0; i < totPoints; i++) {
                auto d_u_w_pk_pi = W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[i], As(k)).first;
                auto d_v_w_pk_pi = W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[i], As(k)).first;
                d_u += 2.0 * (W_midTerm(i) - 0.5) * d_u_w_pk_pi;
                d_v += 2.0 * (W_midTerm(i) - 0.5) * d_v_w_pk_pi;
            }
            grad_midTerm(k * 2 + 0) = d_u;
            grad_midTerm(k * 2 + 1) = d_v;
        }
        
    }
    // end mid 0.5

    // tot charge
    Eigen::VectorXd grad_totCharge(totPoints * 2);
    {
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            double d_u = 0, d_v = 0;
            // case 1.1
            for (int j = 0; j < totPoints; j++) {
                if (j == k) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[j]).norm2() < 1e-4) {
                    continue;
                }
                Vector3 _g = W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[k], As(j)).second;
                d_u += dot(d_u_nowNk, _g) * As(k);
                d_v += dot(d_v_nowNk, _g) * As(k);
            }
            // case 2.2
            for (int i = 0; i < totPoints; i++) {
                if (i == k) continue;
                if ((Vec3_Ps[i] - Vec3_Ps[k]).norm2() < 1e-4) {
                    continue;
                }
                Vector3 _g_u = W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[i], As(k)).second;
                Vector3 _g_v = W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[i], As(k)).second;
                d_u += dot(nowNs[i], _g_u) * As(i);
                d_v += dot(nowNs[i], _g_v) * As(i);
            }

            grad_totCharge(k * 2 + 0) = d_u ;
            grad_totCharge(k * 2 + 1) = d_v ;
        }
    }

    std::cout << "******* mid = " << totMid << ", grad_midTerm.norm = " << grad_midTerm.norm() << "\n";
    std::cout << "********boundary_diri = " << std::setprecision(10) << -1.0 * boundary_diri << ", grad_boundary.norm = " << (-1.0 * grad_boundary).norm() << "\n";
    std::cout << "********totE = " << std::setprecision(10) << totE << ", grad_totE.norm = " << grad_totE.norm() << "\n";
    std::cout << "********totnearPointsE = " << totnearPointsE << ", grad_totNearE.norm = " << grad_totNearE.norm() << "\n";
    std::cout << "************totCharge = " << -1e6 * totCharge << ", grad_totcharge.norm = " << ( - 1e6 * grad_totCharge).norm() << "\n";
    //std::cout << "********grad_totNearE.norm = " << grad_totNearE.norm() << "\n";
    double fx = 0;

    //fx += 1000.0 * totMid;
    //fx += -1.0 * boundary_diri;
    //fx += totE;
    //fx +=  totnearPointsE;
    fx += -1e6 * totCharge;
    // 
    //grad = grad_totNearE;
    //grad = -1.0 * grad_boundary  /* + grad_totE*/;
    //grad = grad_totE;
    //grad = 1000.0 * grad_midTerm;
    grad = -1e6 * grad_totCharge;
    
    std::cout << "fx = " << fx << ", grad.norm = " << grad.norm() << "\n";
    //std::cout << "grad_boundary.norm = " << (-1.0 * grad_boundary).norm() << ", grad_totE.norm = " << grad_totE.norm() << "\n";
    if (iter % 1 == 0) {
        polyscope::getPointCloud("init point clouds")->addVectorQuantity("normal " + std::to_string(iter), nowNs);
        polyscope::getPointCloud("init point clouds")->addVectorQuantity("grad " + std::to_string(iter), grad_p_direct);
        polyscope::getPointCloud("init point clouds")->addScalarQuantity("W plus" + std::to_string(iter), W_plus);
        polyscope::getPointCloud("init point clouds")->addScalarQuantity("W minus" + std::to_string(iter), W_minus);

        polyscope::getPointCloud("init point clouds")->addScalarQuantity("W mid" + std::to_string(iter), W_midTerm);
    }


    iter++;
    return fx;
}

Eigen::VectorXd modify(Eigen::VectorXd x) {
    const double epsP = 1e-2;
    int totPoints = x.size() / 2;
    std::vector<Vector3> Ns(totPoints);
    std::vector<Vector3> grad_p(totPoints);
    for (int i = 0; i < totPoints; i++) {
        double u = x(i * 2 + 0);
        double v = x(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        Ns[i] = nowN.normalize();
        //nowNs[i] = Vec3_Ns[i];
    }
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        double _valplus = 0, _valminus = 0, _valWp = 0;
        Vector3 tmp{ 0, 0, 0 };
        // w
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[i] - Vec3_Ps[j]).norm() < epsP) continue;
            auto now = W_pi_q(Vec3_Ps[j], Ns[j], Vec3_Ps[i], As(j));
            _valWp += now.first;
            tmp += now.second;
        }
        grad_p[i] = tmp;
    }
    int hasModify = 0;
    int cnt = 0;
    do {
        hasModify = 0;
        for (int i = 0; i < totPoints; i++) {
            
            Vector3 avgGrad_i{ 0, 0, 0 };
            /*
            for (int j = 0; j < knn_idx.cols(); j++) {
                int nowId = knn_idx(i, j);
                avgGrad_i += grad_p[nowId] * As(nowId);
            }
            */
            avgGrad_i = grad_p[i];
            avgGrad_i = avgGrad_i.normalize();
            //Ns[i] = avgGrad_i;
            double angle = std::acos(dot(Ns[i], avgGrad_i));
            if (angle > PI / 6.0) {
                Ns[i] = avgGrad_i;
                //Ns[i] = -Ns[i];
                hasModify = 1;
            }
            
        }
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            double _valplus = 0, _valminus = 0, _valWp = 0;
            Vector3 tmp{ 0, 0, 0 };
            // w
            for (int j = 0; j < totPoints; j++) {
                if ((Vec3_Ps[i] - Vec3_Ps[j]).norm() < epsP) continue;
                auto now = W_pi_q(Vec3_Ps[j], Ns[j], Vec3_Ps[i], As(j));
                _valWp += now.first;
                tmp += now.second;
            }
            grad_p[i] = tmp;
        }
        polyscope::getPointCloud("init point clouds")->addVectorQuantity("modify N_" + std::to_string(cnt), Ns);
        polyscope::getPointCloud("init point clouds")->addVectorQuantity("modify grad_p_" + std::to_string(cnt), grad_p);
        cnt++;
        std::cout << "iter = " << cnt << "\n";
        
    } while (hasModify && cnt < 60);
    //kitten_blue_500_PCANormal_real_4_
    //writePointCloudAsXYZVector(Vec3_Ps, Ns, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\kitten_blue_500_PCANormal_real_4_112_modify.xyz");
    // 
    writePointCloudAsXYZVector(Vec3_Ps, Ns, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\Art-Institute-Chicago-Lion_blue_11000_0.5_real_4_14_168_modify.xyz");
    return x;
}

void iterativeImplicitRec() {

}

void shape_A() {
    std::ifstream in;
    in.open("../../../../data/shape_As/shape_A.xyz");
    int cnt = 0;
    std::vector<Vector3> tmpps;
    for (std::string line; std::getline(in, line);) {
        std::istringstream lineStream(line);
        double x, y, z, nx, ny, nz;
        lineStream >> x >> y >> z >> nx >> ny >> nz;
        tmpps.push_back(Vector3{ x, y, z });
        Center += Vector3{ x, y, z };
        cnt++;
    }
    Center /= 1.0 * cnt;
    double tscale = 1.0;
    for (int i = 0; i < tmpps.size(); i++) {
        tmpps[i] -= Center;
        tscale = std::max(1.0 * tmpps[i].norm(), tscale);
    }
    for (int i = 0; i < tmpps.size(); i++) {
        tmpps[i] /= 1.0 * tscale;
    }

    polyscope::registerPointCloud("A", tmpps);

}

double func_run_nearGrid(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    std::cout << "iter = ********************" << iter << "\n";
    const double epsD = 0.1;
    const double epsP = 1e-2;
    int totPoints = Vec3_Ps.size();
    int totSpacePoints = thinBand.size();   
    int totFar_VDQSPoints = far_VDQS.size();
    std::vector<Vector3> nowNs(totPoints);
    std::vector<Vector3> grad_p(totPoints);
    std::vector<Vector3> grad_thinband(totSpacePoints);
    // far_VDQS
    std::vector<Vector3> grad_far_VDQS(totFar_VDQSPoints);
    std::vector<Vector3> grad_FEM_far_VDQS(totFar_VDQSPoints);
    // plus, minus
    std::vector<Vector3> grad_Pplus(totPoints);
    std::vector<Vector3> grad_Pminus(totPoints);

    Eigen::VectorXd W_p(totPoints);
    Eigen::VectorXd W_thinBand(totSpacePoints);
    Eigen::VectorXd Diri_p(totPoints);
    Eigen::VectorXd Diri_thinBand(totSpacePoints);
    // far_VDQS
    Eigen::VectorXd W_far_VDQS(totFar_VDQSPoints);
    Eigen::VectorXd Diri_far_VDQS(totFar_VDQSPoints);

    std::vector<Vector3> Vec_Pplus(totPoints);
    std::vector<Vector3> Vec_Pminus(totPoints);

    for (int i = 0; i < totPoints; i++) {
        double u = x(i * 2 + 0);
        double v = x(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = nowN.normalize();
        nowNs[i] = Vec3_Ns[i];
    }
    // plus minus
    std::vector<Vector3> Pplus(totPoints), Pminus(totPoints);
    Eigen::VectorXd W_plus(totPoints), W_minus(totPoints);
    for (int i = 0; i < totPoints; i++) {
        Vector3 nowN = nowNs[i];
        double mx = -INF, mi = INF;
        for (Vector3 q : VD_Ps[i]) {
            Vector3 pq = q - Vec3_Ps[i];
            pq = pq.normalize();
            double angle = std::acos(dot(pq, nowN));
            if (angle > mx ) {
                mx = angle;
                Pminus[i] = q;
            }
            if (angle < mi ) {
                mi = angle;
                Pplus[i] = q;
            }
        }

        //Pplus[i] = Vec3_Ps[i] + epsD * nowN;
        //Pminus[i] = Vec3_Ps[i] - epsD * nowN;
    }
    /*
    for (int i = 0; i < totPoints; i++) {
        Vector3 Vplus = Pplus[i] - Vec3_Ps[i];
        Vector3 Vminus = Pminus[i] - Vec3_Ps[i];
        if (Vplus.norm() > epsD) {
            Pplus[i] = Vec3_Ps[i] + epsD * (Vplus.normalize());
        }
        if (Vminus.norm() > epsD) {
            Pminus[i] = Vec3_Ps[i] + epsD * (Vminus.normalize());
        }
        Vec_Pplus[i] = (Pplus[i] - Vec3_Ps[i]);
        Vec_Pminus[i] = (Pminus[i] - Vec3_Ps[i]);
    }
    */
    polyscope::registerPointCloud("Pplus", Pplus);
    polyscope::registerPointCloud("Pminus", Pminus);

    double totPlusE = 0, totMinusE = 0;
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 qPlus = Pplus[i], qMinus = Pminus[i];
        double _valplus = 0, _valminus = 0, _valWp = 0;
        Vector3 tmp{ 0, 0, 0 }, grad_plus_i{ 0, 0, 0 }, grad_minus_i{0, 0, 0};
        // plus
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[j] - qPlus).norm() < epsP) continue;
            auto _Plus = W_pi_q(Vec3_Ps[j], nowNs[j], qPlus, As(j));
            _valplus += _Plus.first;
            grad_plus_i += _Plus.second;
        }
        // minus
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[j] - qMinus).norm() < epsP) continue;
            auto _Minus = W_pi_q(Vec3_Ps[j], nowNs[j], qMinus, As(j));
            _valminus += _Minus.first;
            grad_minus_i += _Minus.second;
        }
        // w
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[i] - Vec3_Ps[j]).norm() < epsP) continue;
            auto now = W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j));
            _valWp += now.first;
            tmp += now.second;
        }
        totPlusE += grad_plus_i.norm2();
        totMinusE += grad_minus_i.norm2();

        W_plus(i) = _valplus;
        W_minus(i) = _valminus;
        W_p(i) = _valWp;
        grad_p[i] = tmp;
    }
    std::cout << "tot near E = " << totPlusE + totMinusE << ", plus = " << totPlusE << ", minus = " << totMinusE << "\n";
    // plus minus end
    

    {
        /*
        std::vector<Vector3> newPs, newNs;
        for (int i = 0; i < totPoints; i++) {
            if (W_p[i] < 0.8) {
                newPs.emplace_back(Vec3_Ps[i]);
                newNs.emplace_back(Vec3_Ns[i]);
            }
        }
        //polyscope::registerPointCloud("new Ps", newPs);
        writePointCloudAsXYZVector(newPs, newNs, "D:\\liuweizhou\\models\\famousShape\\xyz\\Arrayed_Vase_blue_12000_clean");
        */
        
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::read_triangle_mesh("../../../../data/disk.obj", V, F);
        polyscope::registerSurfaceMesh("disk", V, F);
        
        std::vector<Vector3> Vs;
        
        
        Eigen::VectorXd W_disk(V.rows());
#pragma omp parallel for
        for (int i = 0; i < V.rows(); i++) {
            Vector3 nowV{ V(i, 0), V(i, 1), V(i, 2) };
            double _valWp = 0;
            for (int j = 0; j < totPoints; j++) {
                if ((nowV - Vec3_Ps[j]).norm() < epsP) continue;
                auto now = W_pi_q(Vec3_Ps[j], nowNs[j], nowV, As(j));
                _valWp += now.first;
            }
            W_disk(i) = _valWp;
        }
        polyscope::getSurfaceMesh("disk")->addVertexScalarQuantity("W", W_disk);
        
    }


    // plus, minus dirichlet energy
    
    std::vector<Vector3> PlusMinusPs;
    double plusMinusDirichletE = 0;
    Eigen::VectorXd grad_plusMinusDirichletE(totPoints * 2);
    PlusMinusPs = uniquePlusMinus(Pplus, Pminus);
    int totPlusMinusPoints = PlusMinusPs.size(); std::cout << "totPlusMinusPoints = " << totPlusMinusPoints << "\n";
    std::vector<Vector3> grad_PlusMinusPs(totPlusMinusPoints);
    /*
    {
        totPlusE = totMinusE = 0;
#pragma omp parallel for
        for (int i = 0; i < totPlusMinusPoints; i++) {
            double _valWp = 0;
            Vector3 tmp{ 0, 0, 0 }, grad_i{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                if ((Vec3_Ps[j] - PlusMinusPs[i]).norm() < epsD) continue;
                auto now = W_pi_q(Vec3_Ps[j], nowNs[j], PlusMinusPs[i], As(j));
                _valWp += now.first;
                grad_i += now.second;
            }
            grad_PlusMinusPs[i] = grad_i;
            plusMinusDirichletE += grad_i.norm2();
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

            double d_u = 0, d_v = 0;
#pragma omp parallel for
            for (int j = 0; j < totPoints; j++) {
                if ((Vec3_Ps[k] - PlusMinusPs[j]).norm() < epsD) continue;
                //d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)).second;
                //d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)).second;
                d_u += 2.0 * dot(grad_PlusMinusPs[j], W_pi_q(Vec3_Ps[k], d_u_nowNk, PlusMinusPs[j], As(k)).second);
                d_v += 2.0 * dot(grad_PlusMinusPs[j], W_pi_q(Vec3_Ps[k], d_v_nowNk, PlusMinusPs[j], As(k)).second);
            }
            grad_plusMinusDirichletE(k * 2 + 0) = d_u;
            grad_plusMinusDirichletE(k * 2 + 1) = d_v;
        }
    }
    */
    

    // dirichlet enery on far_VDQS
    double far_VDQSDirichletE = 0;
    Eigen::VectorXd grad_far_VDQSDirichletE(totPoints * 2);
    /*
    {

#pragma omp parallel for
        for (int k = 0; k < totFar_VDQSPoints; k++) {
            Vector3 Vec3_Q = far_VDQS[k];
            double val_k = 0;
            Vector3 grad_k = Vector3{ 0, 0, 0 };
            //if (Dis_thinBand[k] < gamma) continue;
            for (int i = 0; i < totPoints; i++) {
                //if ((Vec3_Q - Vec3_Ps[i]).norm2() < 1e-5) continue;
                //auto now = W_pi_q(Vec3_Ps[i], Vec3_Ns[i], Vec3_GVi, As(i));
                auto now = W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_Q, As(i));

                val_k += now.first;
                grad_k += now.second;
            }
            W_far_VDQS(k) = val_k;
            grad_far_VDQS[k] = grad_k;
            far_VDQSDirichletE += grad_k.norm2();
            Diri_far_VDQS(k) = grad_k.norm2();
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

            double d_u = 0, d_v = 0;
#pragma omp parallel for
            for (int j = 0; j < totFar_VDQSPoints; j++) {
                //if ((Vec3_Ps[k] - nearPoints[j]).norm2() < 1e-4) continue;
                //d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)).second;
                //d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)).second;
                d_u += 2.0 * dot(grad_far_VDQS[j], W_pi_q(Vec3_Ps[k], d_u_nowNk, far_VDQS[j], As(k)).second);
                d_v += 2.0 * dot(grad_far_VDQS[j], W_pi_q(Vec3_Ps[k], d_v_nowNk, far_VDQS[j], As(k)).second);

            }

            grad_far_VDQSDirichletE(k * 2 + 0) = d_u;
            grad_far_VDQSDirichletE(k * 2 + 1) = d_v;
        }
    }
    */

    // FEM ! dirichlet enery on far_VDQS
    double FEM_far_VDQSDirichletE = 0;
    Eigen::VectorXd grad_FEM_far_VDQSDirichletE(totPoints * 2);
    /*
    {
#pragma omp parallel for
        for (int k = 0; k < totFar_VDQSPoints; k++) {
            Vector3 Vec3_Q = far_VDQS[k];
            double val_k = 0;
            Vector3 grad_k = Vector3{ 0, 0, 0 };
            //if (Dis_thinBand[k] < gamma) continue;
            for (int i = 0; i < totPoints; i++) {
                //if ((Vec3_Q - Vec3_Ps[i]).norm2() < 1e-5) continue;
                //auto now = W_pi_q(Vec3_Ps[i], Vec3_Ns[i], Vec3_GVi, As(i));
                auto now = W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_Q, As(i));
                
                val_k += now.first;
                grad_k += FEM_grad_w_pi_q(Vec3_Ps[i], nowNs[i], Vec3_Q, As(i));
            }
            //W_far_VDQS(k) = val_k;
            grad_FEM_far_VDQS[k] = grad_k;
            FEM_far_VDQSDirichletE += grad_k.norm2();
            //Diri_far_VDQS(k) = grad_k.norm2();
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

            double d_u = 0, d_v = 0;
#pragma omp parallel for
            for (int j = 0; j < totFar_VDQSPoints; j++) {
                //if ((Vec3_Ps[k] - nearPoints[j]).norm2() < 1e-4) continue;
                //d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)).second;
                //d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)).second;
                d_u += 2.0 * dot(grad_FEM_far_VDQS[j], FEM_grad_w_pi_q(Vec3_Ps[k], d_u_nowNk, far_VDQS[j], As(k)));
                d_v += 2.0 * dot(grad_FEM_far_VDQS[j], FEM_grad_w_pi_q(Vec3_Ps[k], d_v_nowNk, far_VDQS[j], As(k)));

            }

            grad_FEM_far_VDQSDirichletE(k * 2 + 0) = d_u;
            grad_FEM_far_VDQSDirichletE(k * 2 + 1) = d_v;
        }
    }
    */

    // // space
    double spaceDirichletE = 0;
    Eigen::VectorXd grad_spaceDirichletE(totPoints * 2);
    // dirichlet energy on space
    /*
    {
#pragma omp parallel for
        for (int k = 0; k < totSpacePoints; k++) {
            Vector3 Vec3_Q = thinBand[k];
            double val_k = 0;
            Vector3 grad_k = Vector3{ 0, 0, 0 };
            //if (Dis_thinBand[k] < gamma) continue;
            for (int i = 0; i < totPoints; i++) {
                //if ((Vec3_Q - Vec3_Ps[i]).norm2() < 1e-5) continue;
                //auto now = W_pi_q(Vec3_Ps[i], Vec3_Ns[i], Vec3_GVi, As(i));
                auto now = W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_Q, As(i));

                val_k += now.first;
                grad_k += now.second;
            }
            W_thinBand(k) = val_k;
            grad_thinband[k] = grad_k;
            spaceDirichletE += grad_k.norm2();
            Diri_thinBand(k) = grad_k.norm2();
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

            double d_u = 0, d_v = 0;
#pragma omp parallel for
            for (int j = 0; j < totSpacePoints; j++) {
                //if ((Vec3_Ps[k] - nearPoints[j]).norm2() < 1e-4) continue;
                //d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)).second;
                //d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)).second;
                d_u += 2.0 * dot(grad_thinband[j], W_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)).second);
                d_v += 2.0 * dot(grad_thinband[j], W_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)).second);

            }

            grad_spaceDirichletE(k * 2 + 0) = d_u;
            grad_spaceDirichletE(k * 2 + 1) = d_v;
        }
    }
    */

    
    // dirichlet integral energy on boundary
    double boundaryintDirichletE = 0;
    Eigen::VectorXd grad_boundaryintDirichletE(totPoints * 2);
    /*
    {
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            Vector3 tmp{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                //if (i == j) continue;

                if ((Vec3_Ps[i] - Vec3_Ps[j]).norm2() < 1e-4) {
                    continue;
                }

                tmp += W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j)).second;
            }
            boundaryintDirichletE += (-W_plus[i] + W_minus[i]) * dot(nowNs[i], tmp) * As(i);
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            double w_pk_pkplus = W_pi_q(Vec3_Ps[k], nowNk, Pplus[k], As(k)).first;
            double w_pk_pkminus = W_pi_q(Vec3_Ps[k], nowNk, Pminus[k], As(k)).first;

            double d_u_w_pk_pkplus = W_pi_q(Vec3_Ps[k], d_u_nowNk, Pplus[k], As(k)).first;
            double d_u_w_pk_pkminus = W_pi_q(Vec3_Ps[k], d_u_nowNk, Pminus[k], As(k)).first;

            double d_v_w_pk_pkplus = W_pi_q(Vec3_Ps[k], d_v_nowNk, Pplus[k], As(k)).first;
            double d_v_w_pk_pkminus = W_pi_q(Vec3_Ps[k], d_v_nowNk, Pminus[k], As(k)).first;

            double case1_1_u = 0, case1_2_u = 0, case2_1_u = 0, case2_2_u = 0;
            double case1_1_v = 0, case1_2_v = 0, case2_1_v = 0, case2_2_v = 0;
            // case 1.1
            Vector3 grad_pk_pkplus = W_pi_q(Vec3_Ps[k], nowNs[k], Pplus[k], As(k)).second;
            Vector3 d_u_grad_pk_pkplus = W_pi_q(Vec3_Ps[k], d_u_nowNk, Pplus[k], As(k)).second;
            Vector3 d_v_grad_pk_pkplus = W_pi_q(Vec3_Ps[k], d_v_nowNk, Pplus[k], As(k)).second;
            // d_u
            case1_1_u += ((-d_u_w_pk_pkplus + d_u_w_pk_pkminus) * dot(nowNk, grad_pk_pkplus) +
                (-w_pk_pkplus + w_pk_pkminus) * dot(d_u_nowNk, grad_pk_pkplus) +
                (-w_pk_pkplus + w_pk_pkminus) * dot(nowNk, d_u_grad_pk_pkplus)) * As(k);


            case1_1_v += ((-d_v_w_pk_pkplus + d_v_w_pk_pkminus) * dot(nowNk, grad_pk_pkplus) +
                (-w_pk_pkplus + w_pk_pkminus) * dot(d_v_nowNk, grad_pk_pkplus) +
                (-w_pk_pkplus + w_pk_pkminus) * dot(nowNk, d_v_grad_pk_pkplus)) * As(k);
            // !!!!!!!!!!!!!!!!!!!!
            case1_1_u = 0;
            case1_1_v = 0;
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // d_v
            // case 1.2
            Vector3 sigma_grad{ 0, 0, 0 };
#pragma omp parallel for
            for (int i = 0; i < totPoints; i++) {
                //if (i == k) continue;

                if ((Vec3_Ps[k] - Vec3_Ps[i]).norm2() < 1e-4) {
                    continue;
                }

                sigma_grad += W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_Ps[k], As(i)).second;
            }
            // d_u
            case1_2_u += ((-d_u_w_pk_pkplus + d_u_w_pk_pkminus) * dot(nowNk, sigma_grad) +
                (-w_pk_pkplus + w_pk_pkminus) * dot(d_u_nowNk, sigma_grad)) * As(k);
            // d_v
            case1_2_v += ((-d_v_w_pk_pkplus + d_v_w_pk_pkminus) * dot(nowNk, sigma_grad) +
                (-w_pk_pkplus + w_pk_pkminus) * dot(d_v_nowNk, sigma_grad)) * As(k);
            // case 2.1
#pragma omp parallel for
            for (int i = 0; i < totPoints; i++) {
                //if (i == k) continue;

                if ((Vec3_Ps[k] - Vec3_Ps[i]).norm2() < 1e-4) {
                    continue;
                }

                // d_u
                auto _d_u = W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[i], As(k)).second;
                case2_1_u += ((-w_pk_pkplus + w_pk_pkminus) * dot(nowNs[i], _d_u)) * As(i);
                // d_v
                auto _d_v = W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[i], As(k)).second;
                case2_1_v += ((-w_pk_pkplus + w_pk_pkminus) * dot(nowNs[i], _d_v)) * As(i);
            }
            // case 2.2

            grad_boundaryintDirichletE(k * 2 + 0) = (case1_1_u + case1_2_u + case2_1_u + case2_2_u);
            grad_boundaryintDirichletE(k * 2 + 1) = (case1_1_v + case1_2_v + case2_1_v + case2_2_v);
        }
    }
    */
    
    double boundaryDirichletE = 0;
    Eigen::VectorXd grad_boundaryDirichletE(totPoints * 2);
    
    // dirichlet energy on boundary
    /*
    {
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            Vector3 tmp{ 0, 0, 0 };
            double val_i = 0;

            boundaryDirichletE += dot(nowNs[i], grad_p[i].normalize()) * As(i);
        }

#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            double d_u = 0, d_v = 0;
            // case 1.1
#pragma omp parallel for
            for (int j = 0; j < totPoints; j++) {
                if (j == k) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[j]).norm() < epsP) {
                    continue;
                }
                Vector3 _g = W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[k], As(j)).second;
                d_u += dot(d_u_nowNk, _g) * As(k);
                d_v += dot(d_v_nowNk, _g) * As(k);
            }
            // case 2.2
#pragma omp parallel for
            for (int i = 0; i < totPoints; i++) {
                if (i == k) continue;
                if ((Vec3_Ps[i] - Vec3_Ps[k]).norm() < epsP) {
                    continue;
                }
                Vector3 _g_u = W_pi_q(Vec3_Ps[k], d_u_nowNk, Vec3_Ps[i], As(k)).second;
                Vector3 _g_v = W_pi_q(Vec3_Ps[k], d_v_nowNk, Vec3_Ps[i], As(k)).second;
                d_u += dot(nowNs[i], _g_u) * As(i);
                d_v += dot(nowNs[i], _g_v) * As(i);
            }

            grad_boundaryDirichletE(k * 2 + 0) = d_u;
            grad_boundaryDirichletE(k * 2 + 1) = d_v;
        }
    }
    */

    // real dirichlet enery on boundary 
    // ?????????
    
    double realboundaryDirichletE = 0;
    Eigen::VectorXd grad_realboundaryDirichletE(totPoints * 2);
    /*
    {
        for (int k = 0; k < totPoints; k++) {

            realboundaryDirichletE += (W_plus[k] + W_minus[k]) * dot(grad_p[k], nowNs[k]) * As(k);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };
            
            double ucase1 = 0, ucase3 = 0, ucase6 = 0, ucase7 = 0, ucase8 = 0;
            double vcase1 = 0, vcase3 = 0, vcase6 = 0, vcase7 = 0, vcase8 = 0;
            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            // case 1 (k=l, i!=l, j!= l)
            _w = W_plus[l] + W_minus[l];
            _grad = grad_p[l];
            ucase1 += _w * dot(_grad, d_u_nowNl) * As(l);
            vcase1 += _w * dot(_grad, d_v_nowNl) * As(l);
            // case 3 (k=l, i=l, j!= l)
            
            ucase3 += ((d_u_w_pl_plplus + d_u_w_pl_plminus) * dot(grad_p[l], nowNl) + 
                        (w_pl_plplus + w_pl_plminus) * dot(grad_p[l], d_u_nowNl)) * As(l);
            vcase3 += ((d_v_w_pl_plplus + d_v_w_pl_plminus) * dot(grad_p[l], nowNl) +
                        (w_pl_plplus + w_pl_plminus) * dot(grad_p[l], d_v_nowNl)) * As(l);
            // 注意！！！！！！！！！
            //ucase3 = 0;
            //vcase3 = 0;

            // case 6 (k!=l, i!=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                //_w = (-W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _w = (W_plus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first) + (W_minus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase6 += _w * dot(d_u_grad_w_pl_pk, nowNs[k]) * As(k);
                vcase6 += _w * dot(d_v_grad_w_pl_pk, nowNs[k]) * As(k);
            }
            // case 7 (k!=l, i=l, j!=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                //if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() < epsP) continue;
                _grad = grad_p[k];
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() >= epsP) {
                    _grad -= W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                }
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                ucase7 += (d_u_w_pl_pk_plus + d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
                vcase7 += (d_v_w_pl_pk_plus + d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
            }
            // case 8 (k!=l, i=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _w = (W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _grad = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase8 += ( (d_u_w_pl_pk_plus + d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                            _w * dot(d_u_grad_w_pl_pk, nowNs[k]) ) * As(k);
                vcase8 += ((d_v_w_pl_pk_plus + d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                            _w * dot(d_v_grad_w_pl_pk, nowNs[k])) * As(k);
            }

            grad_realboundaryDirichletE(l * 2 + 0) = (ucase1 + ucase3 + ucase6 + ucase7 + ucase8);
            grad_realboundaryDirichletE(l * 2 + 1) = (vcase1 + vcase3 + vcase6 + vcase7 + vcase8);
        }
    }
    // end
    */
    
    // w * grad * n
    double realboundaryDirichletE_2 = 0;
    Eigen::VectorXd grad_realboundaryDirichletE_2(totPoints * 2);
    /*
    {
        for (int k = 0; k < totPoints; k++) {
            realboundaryDirichletE_2 += W_p(k) * dot(grad_p[k], nowNs[k]) * As(k);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double ucase1 = 0, ucase3 = 0, ucase6 = 0, ucase7 = 0, ucase8 = 0;
            double vcase1 = 0, vcase3 = 0, vcase6 = 0, vcase7 = 0, vcase8 = 0;
            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            // case 1 (k=l, i!=l, j!= l)
            _w = W_p[l];
            _grad = grad_p[l];
            ucase1 += _w * dot(_grad, d_u_nowNl) * As(l);
            vcase1 += _w * dot(_grad, d_v_nowNl) * As(l);
            // case 3 (k=l, i=l, j!= l)
            
            // case 6 (k!=l, i!=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _w = W_p[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase6 += _w * dot(d_u_grad_w_pl_pk, nowNs[k]) * As(k);
                vcase6 += _w * dot(d_v_grad_w_pl_pk, nowNs[k]) * As(k);
            }
            // case 7 (k!=l, i=l, j!=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _grad = grad_p[k];
                _grad -= W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;

                double d_u_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).first;
                double d_v_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).first;

                ucase7 += d_u_w_pl_pk * dot(_grad, nowNs[k]) * As(k);
                vcase7 += d_v_w_pl_pk * dot(_grad, nowNs[k]) * As(k);
            }
            // case 8 (k!=l, i=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _w = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).first;
                _grad = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;

                double d_u_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).first;
                double d_v_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).first;

                ucase8 += (d_u_w_pl_pk * dot(_grad, nowNs[k]) +
                    _w * dot(d_u_grad_w_pl_pk, nowNs[k])) * As(k);
                vcase8 += (d_v_w_pl_pk * dot(_grad, nowNs[k]) +
                    _w * dot(d_v_grad_w_pl_pk, nowNs[k])) * As(k);
            }

            grad_realboundaryDirichletE_2(l * 2 + 0) = (ucase1 + ucase3 + ucase6 + ucase7 + ucase8);
            grad_realboundaryDirichletE_2(l * 2 + 1) = (vcase1 + vcase3 + vcase6 + vcase7 + vcase8);
        }
    }
    */

    // test FEM grad
    std::vector<Vector3> FEM_grad_p(totPoints);
    std::vector<Vector3> FEM_grad_thinBand(totSpacePoints);
    double FEM_totdirichletE = 0;
    double boundary_FEM = 0;
    Eigen::VectorXd grad_FEM_totdirichletE(totPoints * 2);
    Eigen::VectorXd grad_FEM_boundary(totPoints * 2);
    
    {
        /*
        const double FEMepsP = 1e-2;
        // tot dirichlet
        double tot2_plus = 0, tot2_minus = 0;  
#pragma omp parallel for
        for (int i = 0; i < totSpacePoints; i++) {
            Vector3 grad_i{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                if ((thinBand[i] - Vec3_Ps[j]).norm() < FEMepsP) continue;
                grad_i += FEM_grad_w_pi_q(Vec3_Ps[j], nowNs[j], thinBand[i], As(j));
            }
            FEM_grad_thinBand[i] = grad_i;
            FEM_totdirichletE += grad_i.norm2();
        }
#pragma omp parallel for
        for (int k = 0; k < totPoints; k++) {
            double u = x(k * 2 + 0);
            double v = x(k * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNk{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNk{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNk{ -sinu * sinv, sinu * cosv, 0 };

            Vector3 d_u_grad_i{ 0, 0, 0 }, d_v_grad_i{ 0, 0, 0 };

            double d_u = 0, d_v = 0;
#pragma omp parallel for
            for (int j = 0; j < totSpacePoints; j++) {
                if ((Vec3_Ps[k] - thinBand[j]).norm() < FEMepsP) continue;
                //d_u_grad_i += W_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)).second;
                //d_v_grad_i += W_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)).second;
                d_u += 2.0 * dot(FEM_grad_thinBand[j], FEM_grad_w_pi_q(Vec3_Ps[k], d_u_nowNk, thinBand[j], As(k)));
                d_v += 2.0 * dot(FEM_grad_thinBand[j], FEM_grad_w_pi_q(Vec3_Ps[k], d_v_nowNk, thinBand[j], As(k)));

            }

            grad_FEM_totdirichletE(k * 2 + 0) = d_u;
            grad_FEM_totdirichletE(k * 2 + 1) = d_v;
        }
        */
        // end
        /*
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            Vector3 grad_p_i{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                if ((Vec3_Ps[i] - Vec3_Ps[j]).norm() < FEMepsP) continue;
                grad_p_i += FEM_grad_w_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j));
                
            }
            FEM_grad_p[i] = grad_p_i;
        }
        
        for (int i = 0; i < totPoints; i++) {
            boundary_FEM += (-W_plus[i] + W_minus[i]) * dot( FEM_grad_p[i], nowNs[i]) * As(i);
        }
        std::cout << "FEM boundary = " << boundary_FEM << "\n";
        std::vector<double> norm_FEM_grad_p(totPoints);
        for (int i = 0; i < totPoints; i++) {
            norm_FEM_grad_p[i] = FEM_grad_p[i].norm2();
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double ucase1 = 0, ucase3 = 0, ucase6 = 0, ucase7 = 0, ucase8 = 0;
            double vcase1 = 0, vcase3 = 0, vcase6 = 0, vcase7 = 0, vcase8 = 0;
            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;
            // FEM_grad_w_pi_q
            // case 1 (k=l, i!=l, j!= l)
            _w = -W_plus[l] + W_minus[l];
            _grad = FEM_grad_p[l];
            ucase1 += _w * dot(_grad, d_u_nowNl) * As(l);
            vcase1 += _w * dot(_grad, d_v_nowNl) * As(l);
            // case 3 (k=l, i=l, j!= l)

            ucase3 += ((-d_u_w_pl_plplus + d_u_w_pl_plminus) * dot(FEM_grad_p[l], nowNl) +
                (-w_pl_plplus + w_pl_plminus) * dot(FEM_grad_p[l], d_u_nowNl)) * As(l);
            vcase3 += ((-d_v_w_pl_plplus + d_v_w_pl_plminus) * dot(FEM_grad_p[l], nowNl) +
                (-w_pl_plplus + w_pl_plminus) * dot(FEM_grad_p[l], d_v_nowNl)) * As(l);
            // 注意！！！！！！！！！
            //ucase3 = 0;
            //vcase3 = 0;

            // case 6 (k!=l, i!=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < FEMepsP) continue;
                //_w = (-W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _w = -(W_plus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first) + (W_minus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                Vector3 d_u_grad_w_pl_pk = FEM_grad_w_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l));
                Vector3 d_v_grad_w_pl_pk = FEM_grad_w_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l));
                ucase6 += _w * dot(d_u_grad_w_pl_pk, nowNs[k]) * As(k);
                vcase6 += _w * dot(d_v_grad_w_pl_pk, nowNs[k]) * As(k);
            }
            // case 7 (k!=l, i=l, j!=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                //if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() < epsP) continue;
                _grad = FEM_grad_p[k];
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() >= FEMepsP) {
                    _grad -= FEM_grad_w_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l));
                }
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                ucase7 += (-d_u_w_pl_pk_plus + d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
                vcase7 += (-d_v_w_pl_pk_plus + d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
            }
            // case 8 (k!=l, i=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < FEMepsP) continue;
                _w = (-W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _grad = FEM_grad_w_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l));
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = FEM_grad_w_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l));
                Vector3 d_v_grad_w_pl_pk = FEM_grad_w_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l));
                ucase8 += ((-d_u_w_pl_pk_plus + d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                    _w * dot(d_u_grad_w_pl_pk, nowNs[k])) * As(k);
                vcase8 += ((-d_v_w_pl_pk_plus + d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                    _w * dot(d_v_grad_w_pl_pk, nowNs[k])) * As(k);
            }

            grad_FEM_boundary(l * 2 + 0) = (ucase1 + ucase3 + ucase6 + ucase7 + ucase8);
            grad_FEM_boundary(l * 2 + 1) = (vcase1 + vcase3 + vcase6 + vcase7 + vcase8);
        }
        */
        //polyscope::getPointCloud("init point clouds")->addVectorQuantity("FEM_grad " + std::to_string(iter), FEM_grad_p);
        //polyscope::getPointCloud("init point clouds")->addScalarQuantity("norm_FEM_grad " + std::to_string(iter), norm_FEM_grad_p);
        
    }
    

    // double-well function
    double val_double_well = 0;
    Eigen::VectorXd grad_val_double_well(totPoints * 2);
    /*
    {
        double D = 4.0;
        for (int i = 0; i < totPoints; i++) {
            val_double_well += 4.0 * std::pow(W_plus[i] - 0.5, 4.0) - 2.0 * std::pow(W_plus[i] - 0.5, 2) - W_plus[i] / D;
            val_double_well += 4.0 * std::pow(W_minus[i] - 0.5, 4.0) - 2.0 * std::pow(W_minus[i] - 0.5, 2) - W_minus[i] / D;
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double d_u = 0, d_v = 0;

            for (int j = 0; j < totPoints; j++) {
                d_u += (16.0 * std::pow(W_plus[j] - 0.5, 3.0) - 4.0 * (W_plus[j] - 0.5) - 1.0 / D) * W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[j], As(l)).first;
                d_u += (16.0 * std::pow(W_minus[j] - 0.5, 3.0) - 4.0 * (W_minus[j] - 0.5) - 1.0 / D) * W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[j], As(l)).first;

                d_v += (16.0 * std::pow(W_plus[j] - 0.5, 3.0) - 4.0 * (W_plus[j] - 0.5) - 1.0 / D) * W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[j], As(l)).first;
                d_v += (16.0 * std::pow(W_minus[j] - 0.5, 3.0) - 4.0 * (W_minus[j] - 0.5) - 1.0 / D) * W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[j], As(l)).first;
            }
            grad_val_double_well(l * 2 + 0) = d_u;
            grad_val_double_well(l * 2 + 1) = d_v;
        }
    }
    */

    // -w_plus^2 + w_minus^2
    Eigen::VectorXd E_3(totPoints);
    double boundaryDirichletE_3 = 0;
    Eigen::VectorXd grad_boundaryDirichletE_3(totPoints * 2);
    /*
    {
        for (int k = 0; k < totPoints; k++) {

            boundaryDirichletE_3 += (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k]) * dot(grad_p[k], nowNs[k]) * As(k);
            E_3(k) = (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k]) * dot(grad_p[k], nowNs[k]) * As(k);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;
            
            double ucase2 = 0, ucase3 = 0, ucase4 = 0;
            double vcase2 = 0, vcase3 = 0, vcase4 = 0;
            // case 2 (j!=l, k=l)
            if ((Vec3_Ps[l] - Pplus[l]).norm() < epsP) {
                d_u_w_pl_plplus = 0;
                d_v_w_pl_plplus = 0;
            }
            if ((Vec3_Ps[l] - Pminus[l]).norm() < epsP) {
                d_u_w_pl_plminus = 0;
                d_v_w_pl_plminus = 0;
            }
            ucase2 += ((-2.0 * W_plus[l] * d_u_w_pl_plplus + 2.0 * W_minus[l] * d_u_w_pl_plminus) * dot(grad_p[l], nowNs[l]) +
                        (-W_plus[l] * W_plus[l] + W_minus[l] * W_minus[l]) * dot(grad_p[l], d_u_nowNl)) * As(l);
            vcase2 += ((-2.0 * W_plus[l] * d_v_w_pl_plplus + 2.0 * W_minus[l] * d_v_w_pl_plminus) * dot(grad_p[l], nowNs[l]) +
                        (-W_plus[l] * W_plus[l] + W_minus[l] * W_minus[l]) * dot(grad_p[l], d_v_nowNl)) * As(l);
            //ucase2 += (-W_plus[l] * W_plus[l] + W_minus[l] * W_minus[l]) * dot(grad_p[l], d_u_nowNl) * As(l);
            //vcase2 += (-W_plus[l] * W_plus[l] + W_minus[l] * W_minus[l]) * dot(grad_p[l], d_v_nowNl) * As(l);
            // case 3 (j!=l, k!=l)
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                _grad = grad_p[k];
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() >= epsP) {
                    _grad -= W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                }
                double _w_plus = W_plus[k], _w_minus = W_minus[k];
                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }
                // !!!!!!!!!!!!!!!!
                ucase3 += (-2.0 * W_plus[k] * d_u_w_pl_pkplus + 2.0 * W_minus[k] * d_u_w_pl_pkminus) * dot(_grad, nowNs[k]) * As(k);
                vcase3 += (-2.0 * W_plus[k] * d_v_w_pl_pkplus + 2.0 * W_minus[k] * d_v_w_pl_pkminus) * dot(_grad, nowNs[k]) * As(k);
            }
            // case 4 (j=l, k!=l)
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _grad = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;

                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }

                ucase4 += ((-2.0 * W_plus[k] * d_u_w_pl_pkplus + 2.0 * W_minus[k] * d_u_w_pl_pkminus) * dot(_grad, nowNs[k]) +
                    (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k]) * dot(d_u_grad_w_pl_pk, nowNs[k])) * As(k);
                vcase4 += ((-2.0 * W_plus[k] * d_v_w_pl_pkplus + 2.0 * W_minus[k] * d_v_w_pl_pkminus) * dot(_grad, nowNs[k]) +
                    (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k]) * dot(d_v_grad_w_pl_pk, nowNs[k])) * As(k);
            }
            grad_boundaryDirichletE_3(l * 2 + 0) = (ucase2 + ucase3 + ucase4);
            grad_boundaryDirichletE_3(l * 2 + 1) = (vcase2 + vcase3 + vcase4);
        }
    }
    */
    
    // only W_minus
    double realboundaryonlyMinustE = 0;
    Eigen::VectorXd grad_realboundaryonlyMinustE(totPoints * 2);
    /*
    {
        for (int k = 0; k < totPoints; k++) {

            realboundaryonlyMinustE += (W_minus[k]) * dot(grad_p[k].normalize(), nowNs[k]) * As(k);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double ucase1 = 0, ucase3 = 0, ucase6 = 0, ucase7 = 0, ucase8 = 0;
            double vcase1 = 0, vcase3 = 0, vcase6 = 0, vcase7 = 0, vcase8 = 0;
            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            // case 1 (k=l, i!=l, j!= l)
            _w = W_plus[l] + W_minus[l];
            _grad = grad_p[l];
            ucase1 += _w * dot(_grad, d_u_nowNl) * As(l);
            vcase1 += _w * dot(_grad, d_v_nowNl) * As(l);
            // case 3 (k=l, i=l, j!= l)

            ucase3 += ((d_u_w_pl_plminus) * dot(grad_p[l], nowNl) +
                        (w_pl_plminus) * dot(grad_p[l], d_u_nowNl)) * As(l);
            vcase3 += ((d_v_w_pl_plminus) * dot(grad_p[l], nowNl) +
                        (w_pl_plminus) * dot(grad_p[l], d_v_nowNl)) * As(l);
            // 注意！！！！！！！！！
            //ucase3 = 0;
            //vcase3 = 0;

            // case 6 (k!=l, i!=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                //_w = (-W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _w = (W_minus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase6 += _w * dot(d_u_grad_w_pl_pk, nowNs[k]) * As(k);
                vcase6 += _w * dot(d_v_grad_w_pl_pk, nowNs[k]) * As(k);
            }
            // case 7 (k!=l, i=l, j!=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                //if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() < epsP) continue;
                _grad = grad_p[k];
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() >= epsP) {
                    _grad -= W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                }
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                ucase7 += (d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
                vcase7 += (d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
            }
            // case 8 (k!=l, i=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _w = W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first;
                _grad = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase8 += ( (d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                            _w * dot(d_u_grad_w_pl_pk, nowNs[k]) ) * As(k);
                vcase8 += ((d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                            _w * dot(d_v_grad_w_pl_pk, nowNs[k])) * As(k);
            }

            grad_realboundaryonlyMinustE(l * 2 + 0) = (ucase1 + ucase3 + ucase6 + ucase7 + ucase8);
            grad_realboundaryonlyMinustE(l * 2 + 1) = (vcase1 + vcase3 + vcase6 + vcase7 + vcase8);
        }
    }
    */
    
    // align term
    double alignE = 0;
    Eigen::VectorXd grad_alignE(totPoints * 2);
    /*
    {
        for (int i = 0; i < totPoints; i++) {
            alignE += W_plus[i] * dot(nowNs[i], Pplus[i] - Vec3_Ps[i]) + W_minus[i] * dot(nowNs[i], Pminus[i] - Vec3_Ps[i]);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            double ucase1 = 0, ucase2 = 0;
            double vcase1 = 0, vcase2 = 0;
            // case 1 (k = l)
            ucase1 += d_u_w_pl_plplus * dot(nowNs[l], Pplus[l] - Vec3_Ps[l]) +
                        W_plus[l] * dot(d_u_nowNl, Pplus[l] - Vec3_Ps[l]) +
                        d_u_w_pl_plminus * dot(nowNs[l], Pminus[l] - Vec3_Ps[l]) +
                        W_minus[l] * dot(d_u_nowNl, Pminus[l] - Vec3_Ps[l]);
            vcase1 += d_v_w_pl_plplus * dot(nowNs[l], Pplus[l] - Vec3_Ps[l]) +
                        W_plus[l] * dot(d_v_nowNl, Pplus[l] - Vec3_Ps[l]) +
                        d_v_w_pl_plminus * dot(nowNs[l], Pminus[l] - Vec3_Ps[l]) +
                        W_minus[l] * dot(d_v_nowNl, Pminus[l] - Vec3_Ps[l]);
            // case 2 (k != l)
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                ucase2 += d_u_w_pl_pkplus * dot(nowNs[k], Pplus[k] - Vec3_Ps[k]) +
                            d_u_w_pl_pkminus * dot(nowNs[k], Pminus[k] - Vec3_Ps[k]);
                vcase2 += d_v_w_pl_pkplus * dot(nowNs[k], Pplus[k] - Vec3_Ps[k]) +
                            d_v_w_pl_pkminus * dot(nowNs[k], Pminus[k] - Vec3_Ps[k]);
            }
            grad_alignE(l * 2 + 0) = (ucase1 + ucase2);
            grad_alignE(l * 2 + 1) = (vcase1 + vcase2);
        }
    }
    */

    // - \sum (p+ - p-)^2
    double jumpE = 0;
    Eigen::VectorXd grad_jumpE(totPoints * 2);
    /*
    {
        for (int i = 0; i < totPoints; i++) {
            jumpE += -(W_plus[i] - W_minus[i]) * (W_plus[i] - W_minus[i]);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            double d_u = 0, d_v = 0;
            for (int k = 0; k < totPoints; k++) {
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }
                d_u += -2.0 * (W_plus[k] - W_minus[k]) * (d_u_w_pl_pkplus - d_u_w_pl_pkminus);
                d_v += -2.0 * (W_plus[k] - W_minus[k]) * (d_v_w_pl_pkplus - d_v_w_pl_pkminus);
            }
            grad_jumpE(l * 2 + 0) = d_u;
            grad_jumpE(l * 2 + 1) = d_v;
        }
    }
    */

    // -w(p+)^2 + w(p-)^2 + 1, E4
    Eigen::VectorXd E_4(totPoints);
    double boundaryDirichletE_4 = 0;
    Eigen::VectorXd grad_boundaryDirichletE_4(totPoints * 2);
    /*
    {
        const double delta = 0;
        for (int k = 0; k < totPoints; k++) {

            boundaryDirichletE_4 += (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(grad_p[k].normalize(), nowNs[k]) * As(k);
            //E_4(k) = (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k] + delta) * dot(grad_p[k], nowNs[k]) * As(k);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            double ucase2 = 0, ucase3 = 0, ucase4 = 0;
            double vcase2 = 0, vcase3 = 0, vcase4 = 0;
            // case 2 (j!=l, k=l)
            
            if ((Vec3_Ps[l] - Pplus[l]).norm() < epsP) {
                d_u_w_pl_plplus = 0;
                d_v_w_pl_plplus = 0;
            }
            if ((Vec3_Ps[l] - Pminus[l]).norm() < epsP) {
                d_u_w_pl_plminus = 0;
                d_v_w_pl_plminus = 0;
            }
            // 此时grad_p[l]是常数

            ucase2 += ((-1.0 * std::sqrt(W_plus[l] * W_plus[l]) *  W_plus[l] * d_u_w_pl_plplus + 1.0 * std::sqrt(W_minus[l] * W_minus[l]) * W_minus[l] * d_u_w_pl_plminus) * dot(grad_p[l], nowNs[l]) / grad_p[l].norm() +
                        (-1.0 * std::sqrt(W_plus[l] * W_plus[l]) + std::sqrt(W_minus[l] * W_minus[l])) * dot(grad_p[l], d_u_nowNl) / grad_p[l].norm() ) * As(l);
            vcase2 += ((-1.0 * std::sqrt(W_plus[l] * W_plus[l]) * W_plus[l] * d_v_w_pl_plplus + 1.0 * std::sqrt(W_minus[l] * W_minus[l]) * W_minus[l] * d_v_w_pl_plminus) * dot(grad_p[l], nowNs[l]) / grad_p[l].norm() +
                        (-1.0 * std::sqrt(W_plus[l] * W_plus[l]) + std::sqrt(W_minus[l] * W_minus[l])) * dot(grad_p[l], d_v_nowNl) / grad_p[l].norm()) * As(l);
            //ucase2 += (-W_plus[l] * W_plus[l] + W_minus[l] * W_minus[l] + delta) * dot(grad_p[l], d_u_nowNl) * As(l);
            //vcase2 += (-W_plus[l] * W_plus[l] + W_minus[l] * W_minus[l] + delta) * dot(grad_p[l], d_v_nowNl) * As(l);
            // 
            // case 3 (j!=l, k!=l)
            // 此时grad_p[l]是常数
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                _grad = grad_p[k];
                double grad_pk_norm = grad_p[k].norm();
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() >= epsP) {
                    _grad -= W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                }
                double _w_plus = W_plus[k], _w_minus = W_minus[k];
                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) {
                    d_u_grad_w_pl_pk = Vector3{ 0, 0, 0 };
                    d_v_grad_w_pl_pk = Vector3{ 0, 0, 0 };
                }
                // !!!!!!!!!!!!!!!!
                
                {
    ucase3 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_u_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_u_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(_grad, nowNs[k]) * (-1.0 * std::pow(dot(grad_p[k], grad_p[k]), -1.5) * dot(grad_p[k], d_u_grad_w_pl_pk))) * As(k);
    vcase3 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_v_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_v_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(_grad, nowNs[k]) * (-1.0 * std::pow(dot(grad_p[k], grad_p[k]), -1.5) * dot(grad_p[k], d_v_grad_w_pl_pk))) * As(k);
                }
                
                
                // 有时这样效果更好
                
                {
    ucase3 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_u_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_u_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm) * As(k);
    vcase3 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_v_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_v_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm) * As(k);
                }
                
            }
            // case 4 (j=l, k!=l)
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm() < epsP) continue;
                _grad = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                double grad_pk_norm = grad_p[k].norm();

                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }
                
                {
    ucase4 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_u_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_u_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(d_u_grad_w_pl_pk, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(_grad, nowNs[k]) * (-1.0 * std::pow(dot(grad_p[k], grad_p[k]), -1.5) * dot(grad_p[k], d_u_grad_w_pl_pk)) )* As(k);
    vcase4 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_v_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_v_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(d_v_grad_w_pl_pk, nowNs[k]) / grad_pk_norm + 
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(_grad, nowNs[k]) * (-1.0 * std::pow(dot(grad_p[k], grad_p[k]), -1.5) * dot(grad_p[k], d_v_grad_w_pl_pk))) * As(k);
                }
                
                
                // 有时这样效果更好
                
                {
    ucase4 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_u_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_u_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(d_u_grad_w_pl_pk, nowNs[k]) / grad_pk_norm ) * As(k);
    vcase4 += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_v_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_v_w_pl_pkminus) * dot(_grad, nowNs[k]) / grad_pk_norm +
        (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])) * dot(d_v_grad_w_pl_pk, nowNs[k]) / grad_pk_norm ) * As(k);
                }
                
            }
            grad_boundaryDirichletE_4(l * 2 + 0) = (ucase2 + ucase3 + ucase4);
            grad_boundaryDirichletE_4(l * 2 + 1) = (vcase2 + vcase3 + vcase4);
        }
    }
    */

    // only (-w(p+)^2 + w(p-)^2 - 1)^2
    double jumpE_2 = 0;
    Eigen::VectorXd grad_jumpE_2(totPoints * 2);
    /*
    {
        auto sqrt2 = [=](double _) ->double {
            return _ * _;
        };
        for (int i = 0; i < totPoints; i++) {
            jumpE_2 += sqrt2(-W_plus[i] * W_plus[i] + W_minus[i] * W_minus[i] - 1);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            double d_u = 0, d_v = 0;
            for (int k = 0; k < totPoints; k++) {
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }
                //d_u += (-2.0 * W_plus[k] * d_u_w_pl_pkplus + 2.0 * W_minus[k] * d_u_w_pl_pkminus);
                //d_v += (-2.0 * W_plus[k] * d_v_w_pl_pkplus + 2.0 * W_minus[k] * d_v_w_pl_pkminus);

                d_u += (-2.0 * W_plus[k] * d_u_w_pl_pkplus + 2.0 * W_minus[k] * d_u_w_pl_pkminus) * 2.0 * (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k] - 1);
                d_v += (-2.0 * W_plus[k] * d_v_w_pl_pkplus + 2.0 * W_minus[k] * d_v_w_pl_pkminus) * 2.0 * (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k] - 1);
            }
            grad_jumpE_2(l * 2 + 0) = d_u;
            grad_jumpE_2(l * 2 + 1) = d_v;
        }
    }
    */
    
    // (|w(p-)| - |w(p+)| - 1) ^ 2, jumpE_3
    double jumpE_3 = 0;
    Eigen::VectorXd grad_jumpE_3(totPoints * 2);
    /*
    {
        auto sqrt2 = [=](double _) ->double {
            return _ * _;
            };
        for (int i = 0; i < totPoints; i++) {
            //-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k])
            jumpE_3 += sqrt2(-1.0 * std::sqrt(W_plus[i] * W_plus[i]) + std::sqrt(W_minus[i] * W_minus[i]) - 1);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            double d_u = 0, d_v = 0;
            for (int k = 0; k < totPoints; k++) {
                double d_u_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pkplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pkminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                if ((Vec3_Ps[l] - Pplus[k]).norm() < epsP) {
                    d_u_w_pl_pkplus = 0;
                    d_v_w_pl_pkplus = 0;
                }
                if ((Vec3_Ps[l] - Pminus[k]).norm() < epsP) {
                    d_u_w_pl_pkminus = 0;
                    d_v_w_pl_pkminus = 0;
                }
                //d_u += (-2.0 * W_plus[k] * d_u_w_pl_pkplus + 2.0 * W_minus[k] * d_u_w_pl_pkminus);
                //d_v += (-2.0 * W_plus[k] * d_v_w_pl_pkplus + 2.0 * W_minus[k] * d_v_w_pl_pkminus);
                //(-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_u_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_u_w_pl_pkminus)
d_u += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_u_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_u_w_pl_pkminus)) * 2.0 * (-1.0 * std::sqrt(W_plus[k] * W_plus[k]) + std::sqrt(W_minus[k] * W_minus[k]) - 1);
d_v += ((-1.0 * std::sqrt(W_plus[k] * W_plus[k]) * W_plus[k] * d_v_w_pl_pkplus + 1.0 * std::sqrt(W_minus[k] * W_minus[k]) * W_minus[k] * d_v_w_pl_pkminus)) * 2.0 * (-W_plus[k] * W_plus[k] + W_minus[k] * W_minus[k] - 1);
            }
            grad_jumpE_3(l * 2 + 0) = d_u;
            grad_jumpE_3(l * 2 + 1) = d_v;
        }
    }
    */

    // 0 - eps < w < 1 + eps, else e^-(x - a) - 1, e^(x - b) - 1
    double inner_term = 0;
    Eigen::VectorXd grad_inner_term(totPoints * 2);
    /*
    {
        const double e_val = 2.71828181845904523536;
        const double a = 0 - 0.05, b = 1 + 0.05;
        for (int i = 0; i < totPoints; i++) {
            if (W_plus[i] < a)
                inner_term += std::exp(-(W_plus[i] - a)) - 1;
            else if (W_plus[i] > b)
                inner_term += std::exp(W_plus[i] - b) - 1;
            if (W_minus[i] < a)
                inner_term += std::exp(-(W_minus[i] - a)) - 1;
            else if (W_minus[i] > b)
                inner_term += std::exp(W_minus[i] - b) - 1;
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            double d_u = 0, d_v = 0;

            for (int j = 0; j < totPoints; j++) {
                if (W_plus[j] < a && (Vec3_Ps[l] - Pplus[j]).norm() >= epsP) {
                    d_u += W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[j], As(l)).first * std::exp(-(W_plus[j] - a));
                    d_v += W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[j], As(l)).first * std::exp(-(W_plus[j] - a));
                }
                else if (W_plus[j] > b && (Vec3_Ps[l] - Pplus[j]).norm() >= epsP) {
                    d_u += W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[j], As(l)).first * std::exp(W_plus[j] - b);
                    d_v += W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[j], As(l)).first * std::exp(W_plus[j] - b);
                }

                if (W_minus[j] < a && (Vec3_Ps[l] - Pminus[j]).norm() >= epsP) {
                    d_u += W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[j], As(l)).first * std::exp(-(W_minus[j] - a));
                    d_v += W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[j], As(l)).first * std::exp(-(W_minus[j] - a));
                }
                else if (W_minus[j] > b && (Vec3_Ps[l] - Pminus[j]).norm() >= epsP) {
                    d_u += W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[j], As(l)).first * std::exp(W_minus[j] - b);
                    d_v += W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[j], As(l)).first * std::exp(W_minus[j] - b);
                }
            }
            grad_inner_term(l * 2 + 0) = d_u;
            grad_inner_term(l * 2 + 1) = d_v;
        }
    }
    */

    //spaceDirichletE *= nearGrid_spacing * nearGrid_spacing * nearGrid_spacing;
    //grad_spaceDirichletE *= nearGrid_spacing * nearGrid_spacing * nearGrid_spacing;

    //plusMinusDirichletE /= (1.0 * (Pplus.size() + Pminus.size()));
    //grad_plusMinusDirichletE /= (1.0 * (Pplus.size() + Pminus.size()));

    val_double_well /= (1.0 * (Pplus.size() + Pminus.size()));
    grad_val_double_well /= (1.0 * (Pplus.size() + Pminus.size()));

    // end
    std::cout << "boundaryDirichletE = " << boundaryDirichletE << "\n";
    //std::cout << "spaceDirichletE = " << spaceDirichletE << "\n";
    std::cout << "boundaryintDirichletE = " << boundaryintDirichletE << "\n";
    std::cout << "realboundaryDirichletE = " << realboundaryDirichletE << "\n";
    std::cout << "realboundaryDirichletE_2 = " << realboundaryDirichletE_2  << "\n";
    std::cout << "far_VDQSDirichletE = " << far_VDQSDirichletE << "\n";
    //std::cout << "plusMinusDirichletE = " << plusMinusDirichletE << "\n";
    std::cout << "val_double_well = " << val_double_well << "\n";
    std::cout << "FEM_totdirichletE = " << FEM_totdirichletE << "\n";
    std::cout << "real_boundary_3 = " << boundaryDirichletE_3 << "\n";
    std::cout << "alignE = " << alignE << "\n";
    std::cout << "plusMinusDirichletE = " << plusMinusDirichletE << "\n";
    std::cout << "realboundaryonlyMinustE = " << realboundaryonlyMinustE << "\n";
    std::cout << "FEM_far_VDQSDirichletE = " << FEM_far_VDQSDirichletE << "\n";
    std::cout << "jumpE = " << jumpE << "\n";
    std::cout << "real_boundary_4 = " << boundaryDirichletE_4 << "\n";
    std::cout << "jumpE_2 = " << jumpE_2 << "\n";
    std::cout << "jumpE_3 = " << jumpE_3 << "\n";

    double fx = 0;
    // tot charge
    //fx += -1000.0 * boundaryDirichletE + spaceDirichletE;
    //grad = -1000.0 * grad_boundaryDirichletE + grad_spaceDirichletE;
    // boundary int
    //fx += 1000.0 * boundaryintDirichletE + spaceDirichletE;
    //grad = 1000.0 * grad_boundaryintDirichletE + grad_spaceDirichletE;
    // real
    //fx += -1.0 * boundaryDirichletE;
    //grad = -1.0 * grad_boundaryDirichletE;

    // boundaryintDirichletE
    //fx += 1.0 * boundaryintDirichletE;
    //grad = 1.0 * grad_boundaryintDirichletE;

    // grad_far_VDQSDirichletE

    //fx += -1000.0 * realboundaryDirichletE + far_VDQSDirichletE;
    //grad = -1000.0 * grad_realboundaryDirichletE + grad_far_VDQSDirichletE;
    // real _ 2 !!!!!!!!!!!!!!!!!!!!!!
    //fx += -100.0 * realboundaryDirichletE_2;
    //grad = -100.0 * grad_realboundaryDirichletE_2;
    // only space dirichlet E
    //fx += spaceDirichletE;
    //grad = grad_spaceDirichletE;
    
    // real
    //fx += -1.0 * realboundaryDirichletE;
    //grad = -1.0 * grad_realboundaryDirichletE;
    // 
    // plus, minus dirichlet E
    //fx += plusMinusDirichletE;
    //grad = grad_plusMinusDirichletE;
    // plus, minus + real
    //fx += -10.0 * realboundaryDirichletE + 0.2 * plusMinusDirichletE;
    //grad = -10.0 * grad_realboundaryDirichletE + 0.2 * grad_plusMinusDirichletE;
    
    // double-well + real
    //fx += -1.0 * realboundaryDirichletE + 5.0 * val_double_well;
    //grad = -1.0 * grad_realboundaryDirichletE + 5.0 * grad_val_double_well;

    // FEM dirichlet
    //fx += 1000.0 * FEM_totdirichletE;
    //grad = 1000.0 * grad_FEM_totdirichletE;

    // FEM boundary
    std::cout << "boundary_FEM = " << boundary_FEM << "\n";
    //fx += 100.0 * boundary_FEM;
    //grad = 100.0 * grad_FEM_boundary;

    // real boundary 3
    //fx += -8000.0 * boundaryDirichletE_3 + far_VDQSDirichletE;
    //grad = -8000.0 * grad_boundaryDirichletE_3 + grad_far_VDQSDirichletE;

    // real boundary 3 + boundaryDirichletE
    //fx += -3.0 * boundaryDirichletE_3 + alignE;
    //grad = -3.0 * grad_boundaryDirichletE_3 + grad_alignE;

    // plusMinusDirichletE + real _ 3
    //fx += -1000.0 * boundaryDirichletE_3 + plusMinusDirichletE;
    //grad = -1000.0 * grad_boundaryDirichletE_3 + grad_plusMinusDirichletE;

    // alignE + realboundaryDirichletE_2
    //fx += -1.0 * realboundaryDirichletE_2 + alignE;
    //grad = -1.0 * grad_realboundaryDirichletE_2 + grad_alignE;

    // real_3 + double-well
    //fx += -1.0 * boundaryDirichletE_3 + val_double_well;
    //grad = -1.0 * grad_boundaryDirichletE_3 + grad_val_double_well;

    // only minus
    //fx += -1.0 * realboundaryonlyMinustE + 10.0 * val_double_well;
    //grad = -1.0 * grad_realboundaryonlyMinustE + 10.0 * grad_val_double_well;

    // real_3
    //fx += -100.0 * boundaryDirichletE_3;
    //grad = -100.0 * grad_boundaryDirichletE_3;

    // FEM_far_VDQSDirichletE
    //fx += 100.0 * FEM_far_VDQSDirichletE;
    //grad = 100.0 * grad_FEM_far_VDQSDirichletE;

    // FEM_totdirichletE + real_3
    //fx += -1.0 * boundaryDirichletE_3 + 100.0 * FEM_totdirichletE;
    //grad = -1.0 * grad_boundaryDirichletE_3 + 100.0 * grad_FEM_totdirichletE;

    // real_3 + jumpE
    //fx += -5.0 * boundaryDirichletE_3 + jumpE;
    //grad = -5.0 * grad_boundaryDirichletE_3 + grad_jumpE;

    // only jumpE_2
    //fx += jumpE_2;
    //grad = grad_jumpE_2;

    // real_4
    fx += -100.0 * boundaryDirichletE_4;
    grad = -100.0 * grad_boundaryDirichletE_4;

    // real_3 + jumpE_2
    //fx += -500.0 * boundaryDirichletE_3 + (1.0 * jumpE_3);
    //grad = -500.0 * grad_boundaryDirichletE_3 + (1.0 * grad_jumpE_3);

    // real_3 + align
    //fx += -100.0 * boundaryDirichletE_3 + alignE;
    //grad = -100.0 * grad_boundaryDirichletE_3 + grad_alignE;

    // real_3 + plusMinusDirichletE

    // align + jumpE_2
    //fx += alignE + jumpE_2;
    //grad = grad_alignE + grad_jumpE_2;

    // jumpE_3 + real_4
    //fx += -1000.0 * boundaryDirichletE_4 + 1.0 * jumpE_3;
    //grad = -1000.0 * grad_boundaryDirichletE_4 + 1.0 * grad_jumpE_3;

    // jumpE_2 + real_4
    //fx += -10000.0 * boundaryDirichletE_4 + 1.0 * jumpE_3;
    //grad = -10000.0 * grad_boundaryDirichletE_4 + 1.0 * grad_jumpE_3;

    std::cout << "fx = " << fx << ", grad.norm = " << grad.norm() << "\n";
    Eigen::VectorXd consistency(totPoints);
    for (int i = 0; i < totPoints; i++) {
        if (dot(nowNs[i], Vec3_Ns[i]) >= 0) {
            consistency(i) = 1.0;
        }
        else consistency(i) = 0;
    }

    double tot_grad_p = 0;
    Eigen::VectorXd norm_grad_p(totPoints);
    for (int i = 0; i < totPoints; i++) {
        tot_grad_p += grad_p[i].norm();
        norm_grad_p(i) = grad_p[i].norm();
    }
    std::cout << "tot_grad_p = " << tot_grad_p << "\n";
    {
        //writeuvs(x, "../../../../data/xs/108chair/realboundaryDirichletE_2_modify_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/real_3+alignE_1+1_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/real_3_modify_2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/real_3_far_VDQSDirichletE_8e3_64_" + std::to_string(iter) + ".txt"); // 
        //writeuvs(x, "../../../../data/xs/108chair/realboundaryDirichletE_without_near_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/boundaryDirichletE_3_without_near_align_3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/realboundaryonlyMinustE_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/boundaryDirichletE_3+double-well_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/FEM_far_VDQSDirichletE_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/108chair/FEM_totdirichletE+real_3_" + std::to_string(iter) + ".txt");// 比较好
        //writeuvs(x, "../../../../data/xs/108chair/108chair_6000_real_3_" + std::to_string(iter) + ".txt");
        // bird7000
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e0_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e1_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e2_72_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e2_72_try3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e2_72_try3_43_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_3_63_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_4_" + std::to_string(iter) + ".txt"); // here
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_4_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_4_try2_52_" + std::to_string(iter) + ".txt");


        // thinBand
        //writeuvs(x, "../../../../data/xs/thinBand2/thinBand2_scaled_real_3_64_" + std::to_string(iter) + ".txt");
        // dress
        //writeuvs(x, "../../../../data/xs/dress_poisson6000/dress_poisson6000_scaled_real_3_" + std::to_string(iter) + ".txt");
        //WireFrame_trebol
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3_try3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_4_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_4_try3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_4_try4_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try4_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try4_210_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+align_try1_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+align_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3_1e2+jumpE_2_1_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_4_5e2+jumpE_3_1_near_0.1_" + std::to_string(iter) + ".txt");
        
        //cup_poisson7000
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_try1_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3+align_try1_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2+align_try2_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2+jumpE_2_try2_71_20_22_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2+jumpE_2_1e2_try3_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2_20_90_146_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_4_modify_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/cup_poisson7000/cup_poisson7000_real_4_49_continue_16_11_" + std::to_string(iter) + ".txt");
        // skull_poisson6000
        //writeuvs(x, "../../../../data/xs/skull_poisson6000/skull_poisson6000_real_4_" + std::to_string(iter) + ".txt");
        
        //BunnyPeel_blue_7000
        //writeuvs(x, "../../../../data/xs/BunnyPeel_blue_7000/BunnyPeel_blue_7000_real_4_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/BunnyPeel_blue_7000/BunnyPeel_blue_7000_real_4_xuruiAreas_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/BunnyPeel_blue_7000/BunnyPeel_blue_7000_real_4_xuruiAreas_63_initAreas_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/BunnyPeel_blue_7000/BunnyPeel_blue_7000_real_4_PCANormal_" + std::to_string(iter) + ".txt");

        // linkCupTop_blue_7000
        //writeuvs(x, "../../../../data/xs/linkCupTop_blue_7000/linkCupTop_blue_7000_real_4_" + std::to_string(iter) + ".txt");

        // 41glass
        //writeuvs(x, "../../../../data/xs/41glass/41glass_real_4_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/41glass/41glass_real_4_PCANormal_" + std::to_string(iter) + ".txt");

        // 397horse_blue_7000
        //writeuvs(x, "../../../../data/xs/397horse_blue_7000/397horse_blue_7000_real_4_" + std::to_string(iter) + ".txt");

        //shellfish2_lr_blue_9200
        //writeuvs(x, "../../../../data/xs/shellfish2_lr_blue_9200/shellfish2_lr_blue_9200_real_4_" + std::to_string(iter) + ".txt");

        //kitten_blue_500_PCANormal
        //writeuvs(x, "../../../../data/xs/kitten_blue_500_PCANormal/kitten_blue_500_PCANormal_real_4_" + std::to_string(iter) + ".txt");

        //Art-Institute-Chicago-Lion_blue_11000
        //writeuvs(x, "../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000/Art-Institute-Chicago-Lion_blue_11000_real_4_" + std::to_string(iter) + ".txt");
        //writeuvs(x, "../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000/Art-Institute-Chicago-Lion_blue_11000_PCANormals_real_4_" + std::to_string(iter) + ".txt");

        //steampunk_gear_cube_small_corner_2_blue_11000
        //writeuvs(x, "../../../../data/xs/steampunk_gear_cube_small_corner_2_blue_11000/steampunk_gear_cube_small_corner_2_blue_11000_real_4_" + std::to_string(iter) + ".txt");

        //WS0.5_4000_torus
        //writeuvs(x, "../../../../data/xs/WS0.5_4000_torus/WS0.5_4000_torus_PCANormals_real_4_" + std::to_string(iter) + ".txt");

        //108chair6000poisson_0.5
        //writeuvs(x, "../../../../data/xs/108chair6000poisson_0.5/108chair6000poisson_0.5_PCANormal_real_4_" + std::to_string(iter) + ".txt");

        //108chair
        //writeuvs(x, "../../../../data/xs/108chair/108chair_real_4_" + std::to_string(iter) + ".txt");

        //Art-Institute-Chicago-Lion_blue_11000_0.5
        //writeuvs(x, "../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000_0.5/Art-Institute-Chicago-Lion_blue_11000_0.5_real_4_14_" + std::to_string(iter) + ".txt");

        //30cup_blue_9000_PCANormal_0.5
        //writeuvs(x, "../../../../data/xs/30cup_blue_9000_PCANormal_0.5/30cup_blue_9000_PCANormal_0.5_real_4_" + std::to_string(iter) + ".txt");
        
        // write XYZ
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\cup_poisson7000.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\BunnyPeel_blue_7000_real_4_49.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\linkCupTop_blue_7000_real_4_52.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\41glass_real_4_58.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\41glass_real_4_72.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\397horse_blue_7000_real_4_62.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\kitten_blue_500_PCANormal_real_4_112.xyz");.
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\Art-Institute-Chicago-Lion_blue_11000_real_4_70.xyz");
        
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\steampunk_gear_cube_small_corner_2_blue_11000_real_4_79.xyz");
        //WS0.5_4000_torus/WS0.5_4000_torus_real_4_
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\WS0.5_4000_torus_PCANormals_real_4_65.xyz");
        //
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\108chair6000poisson_0.5_PCANormal_real_4_74.xyz");

        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\108chair_real_4_84.xyz");
        //writePointCloudAsXYZVector(Vec3_Ps, nowNs, "D:\\liuweizhou\\mycodes\\DiriNormal\\data\\xs\\result\\Art-Institute-Chicago-Lion_blue_11000_0.5_real_4_14_168.xyz");
    }

    //polyscope::getPointCloud("init point clouds")->addScalarQuantity("E_3  " + std::to_string(iter), E_3);
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("w  " + std::to_string(iter), W_p);
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("w_plus " + std::to_string(iter), W_plus);
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("w_minus " + std::to_string(iter), W_minus);
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("consistency " + std::to_string(iter), consistency);
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("norm_grad_p " + std::to_string(iter), norm_grad_p);

    polyscope::getPointCloud("init point clouds")->addVectorQuantity("normal " + std::to_string(iter), nowNs);
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("grad " + std::to_string(iter), grad_p);
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("Vec_Pplus " + std::to_string(iter), Vec_Pplus);
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("Vec_Pminus " + std::to_string(iter), Vec_Pminus);
    /*
    polyscope::getPointCloud("far_VDQS")->addScalarQuantity("W " + std::to_string(iter), W_far_VDQS);
    polyscope::getPointCloud("far_VDQS")->addVectorQuantity("grad " + std::to_string(iter), grad_far_VDQS);
    polyscope::getPointCloud("far_VDQS")->addScalarQuantity("dirichlet E " + std::to_string(iter), Diri_far_VDQS);
    */
    /*
    polyscope::getPointCloud("thinBand")->addScalarQuantity("W " + std::to_string(iter), W_thinBand);
    polyscope::getPointCloud("thinBand")->addVectorQuantity("grad_W " + std::to_string(iter), grad_thinband);
    polyscope::getPointCloud("thinBand")->addScalarQuantity("dirichlet E " + std::to_string(iter), Diri_thinBand);
    */

    {
        //modify(x);
    }

    iter++;
    return fx;
}
// FEM
/*
double FEM_f(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    std::cout << "******************* iter = " << iter << "\n";
    const double epsD = 5e-2;
    const double epsP = 1e-2;
    const double epsFEM = 2e-2;
    int totPoints = Vec3_Ps.size();
    int totSpacePoints = thinBand.size();
    int totFar_VDQSPoints = far_VDQS.size();
    std::vector<Vector3> nowNs(totPoints);
    std::vector<Vector3> grad_p(totPoints);
    std::vector<Vector3> grad_thinband(totSpacePoints);
    // far_VDQS
    std::vector<Vector3> grad_far_VDQS(totFar_VDQSPoints);
    // plus, minus
    std::vector<Vector3> grad_Pplus(totPoints);
    std::vector<Vector3> grad_Pminus(totPoints);

    Eigen::VectorXd W_p(totPoints);
    Eigen::VectorXd W_thinBand(totSpacePoints);
    Eigen::VectorXd Diri_p(totPoints);
    Eigen::VectorXd Diri_thinBand(totSpacePoints);
    // far_VDQS
    Eigen::VectorXd W_far_VDQS(totFar_VDQSPoints);
    Eigen::VectorXd Diri_far_VDQS(totFar_VDQSPoints);

    std::vector<Vector3> Vec_Pplus(totPoints);
    std::vector<Vector3> Vec_Pminus(totPoints);

    for (int i = 0; i < totPoints; i++) {
        double u = x(i * 2 + 0);
        double v = x(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = nowN.normalize();
        nowNs[i] = Vec3_Ns[i];
    }
    // plus minus
    std::vector<Vector3> Pplus(totPoints), Pminus(totPoints);
    Eigen::VectorXd W_plus(totPoints), W_minus(totPoints);
    for (int i = 0; i < totPoints; i++) {
        Vector3 nowN = nowNs[i];
        double mx = -INF, mi = INF;
        for (Vector3 q : VD_Ps[i]) {
            Vector3 pq = q - Vec3_Ps[i];
            pq = pq.normalize();
            double angle = std::acos(dot(pq, nowN));
            if (angle > mx) {
                mx = angle;
                Pminus[i] = q;
            }
            if (angle < mi) {
                mi = angle;
                Pplus[i] = q;
            }
        }

        //Pplus[i] = Vec3_Ps[i] + epsD * nowN;
        //Pminus[i] = Vec3_Ps[i] - epsD * nowN;
    }
    for (int i = 0; i < totPoints; i++) {
        Vector3 Vplus = Pplus[i] - Vec3_Ps[i];
        Vector3 Vminus = Pminus[i] - Vec3_Ps[i];
        if (Vplus.norm() > epsD) {
            Pplus[i] = Vec3_Ps[i] + epsD * (Vplus.normalize());
        }
        if (Vminus.norm() > epsD) {
            Pminus[i] = Vec3_Ps[i] + epsD * (Vminus.normalize());
        }
        Vec_Pplus[i] = (Pplus[i] - Vec3_Ps[i]);
        Vec_Pminus[i] = (Pminus[i] - Vec3_Ps[i]);
    }

    polyscope::registerPointCloud("Pplus", Pplus);
    polyscope::registerPointCloud("Pminus", Pminus);

    double totPlusE = 0, totMinusE = 0;
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 qPlus = Pplus[i], qMinus = Pminus[i];
        double _valplus = 0, _valminus = 0, _valWp = 0;
        Vector3 tmp{ 0, 0, 0 }, grad_plus_i{ 0, 0, 0 }, grad_minus_i{ 0, 0, 0 };
        for (int j = 0; j < totPoints; j++) {
            auto _Plus = W_pi_q(Vec3_Ps[j], nowNs[j], qPlus, As(j));
            auto _Minus = W_pi_q(Vec3_Ps[j], nowNs[j], qMinus, As(j));
            _valplus += _Plus.first;
            _valminus += _Minus.first;
            grad_plus_i += _Plus.second;
            grad_minus_i += _Minus.second;

            if ((Vec3_Ps[i] - Vec3_Ps[j]).norm2() < epsP) continue;
            auto now = W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j));
            _valWp += now.first;
            tmp += now.second;
        }
        totPlusE += grad_plus_i.norm2();
        totMinusE += grad_minus_i.norm2();

        W_plus(i) = _valplus;
        W_minus(i) = _valminus;
        W_p(i) = _valWp;
        grad_p[i] = tmp;
    }
    std::cout << "tot near E = " << totPlusE + totMinusE << ", plus = " << totPlusE << ", minus = " << totMinusE << "\n";
    // pre end

    double realboundaryDirichletE = 0;
    Eigen::VectorXd grad_realboundaryDirichletE(totPoints * 2);
    {
        for (int k = 0; k < totPoints; k++) {

            realboundaryDirichletE += (-W_plus[k] + W_minus[k]) * dot(grad_p[k], nowNs[k]) * As(k);
        }
#pragma omp parallel for
        for (int l = 0; l < totPoints; l++) {
            double u = x(l * 2 + 0);
            double v = x(l * 2 + 1);
            double cosu = std::cos(u), sinu = std::sin(u), cosv = std::cos(v), sinv = std::sin(v);
            Vector3 nowNl{ sinu * cosv, sinu * sinv, cosu };

            Vector3 d_u_nowNl{ cosu * cosv, cosu * sinv, -sinu };
            Vector3 d_v_nowNl{ -sinu * sinv, sinu * cosv, 0 };

            double ucase1 = 0, ucase3 = 0, ucase6 = 0, ucase7 = 0, ucase8 = 0;
            double vcase1 = 0, vcase3 = 0, vcase6 = 0, vcase7 = 0, vcase8 = 0;
            double _w = 0;
            Vector3 _grad{ 0, 0, 0 };

            double d_u_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[l], As(l)).first;
            double d_u_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[l], As(l)).first;

            double d_v_w_pl_plplus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[l], As(l)).first;
            double d_v_w_pl_plminus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[l], As(l)).first;

            double w_pl_plplus = W_pi_q(Vec3_Ps[l], nowNl, Pplus[l], As(l)).first;
            double w_pl_plminus = W_pi_q(Vec3_Ps[l], nowNl, Pminus[l], As(l)).first;

            // case 1 (k=l, i!=l, j!= l)
            _w = -W_plus[l] + W_minus[l];
            _grad = grad_p[l];
            ucase1 += _w * dot(_grad, d_u_nowNl) * As(l);
            vcase1 += _w * dot(_grad, d_v_nowNl) * As(l);
            // case 3 (k=l, i=l, j!= l)

            ucase3 += ((-d_u_w_pl_plplus + d_u_w_pl_plminus) * dot(grad_p[l], nowNl) +
                (-w_pl_plplus + w_pl_plminus) * dot(grad_p[l], d_u_nowNl)) * As(l);
            vcase3 += ((-d_v_w_pl_plplus + d_v_w_pl_plminus) * dot(grad_p[l], nowNl) +
                (-w_pl_plplus + w_pl_plminus) * dot(grad_p[l], d_v_nowNl)) * As(l);
            // 注意！！！！！！！！！
            //ucase3 = 0;
            //vcase3 = 0;

            // case 6 (k!=l, i!=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() < epsP) continue;
                //_w = (-W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _w = -(W_plus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first) + (W_minus[k] - W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase6 += _w * dot(d_u_grad_w_pl_pk, nowNs[k]) * As(k);
                vcase6 += _w * dot(d_v_grad_w_pl_pk, nowNs[k]) * As(k);
            }
            // case 7 (k!=l, i=l, j!=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                //if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() < epsP) continue;
                _grad = grad_p[k];
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() >= epsP) {
                    _grad -= W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                }
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                ucase7 += (-d_u_w_pl_pk_plus + d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
                vcase7 += (-d_v_w_pl_pk_plus + d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) * As(k);
            }
            // case 8 (k!=l, i=l, j=l)
#pragma omp parallel for
            for (int k = 0; k < totPoints; k++) {
                if (k == l) continue;
                if ((Vec3_Ps[k] - Vec3_Ps[l]).norm2() < epsP) continue;
                _w = (-W_pi_q(Vec3_Ps[l], nowNs[l], Pplus[k], As(l)).first + W_pi_q(Vec3_Ps[l], nowNs[l], Pminus[k], As(l)).first);
                _grad = W_pi_q(Vec3_Ps[l], nowNs[l], Vec3_Ps[k], As(l)).second;
                double d_u_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pplus[k], As(l)).first;
                double d_u_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_u_nowNl, Pminus[k], As(l)).first;
                double d_v_w_pl_pk_plus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pplus[k], As(l)).first;
                double d_v_w_pl_pk_minus = W_pi_q(Vec3_Ps[l], d_v_nowNl, Pminus[k], As(l)).first;
                Vector3 d_u_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_u_nowNl, Vec3_Ps[k], As(l)).second;
                Vector3 d_v_grad_w_pl_pk = W_pi_q(Vec3_Ps[l], d_v_nowNl, Vec3_Ps[k], As(l)).second;
                ucase8 += ((-d_u_w_pl_pk_plus + d_u_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                    _w * dot(d_u_grad_w_pl_pk, nowNs[k])) * As(k);
                vcase8 += ((-d_v_w_pl_pk_plus + d_v_w_pl_pk_minus) * dot(_grad, nowNs[k]) +
                    _w * dot(d_v_grad_w_pl_pk, nowNs[k])) * As(k);
            }

            grad_realboundaryDirichletE(l * 2 + 0) = (ucase1 + ucase3 + ucase6 + ucase7 + ucase8);
            grad_realboundaryDirichletE(l * 2 + 1) = (vcase1 + vcase3 + vcase6 + vcase7 + vcase8);
        }
    }
    // test FEM grad
    std::vector<Vector3> FEM_grad_Pplus(totPoints), FEM_grad_Pminus(totPoints);
    std::vector<Vector3> FEM_grad_p(totPoints);
    double boundary_FEM = 0;
    Eigen::VectorXd 
    {
        double tot2_plus = 0, tot2_minus = 0;
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            Vector3 grad_plus_i{ 0, 0, 0 }, grad_minus_i{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                if ((Pplus[i] - Vec3_Ps[j]).norm() < epsFEM) continue;
                grad_plus_i += FEM_grad_w_pi_q(Vec3_Ps[j], nowNs[j], Pplus[i], As(j));
            }
            for (int j = 0; j < totPoints; j++) {
                if ((Pminus[i] - Vec3_Ps[j]).norm() < epsFEM) continue;
                grad_minus_i += FEM_grad_w_pi_q(Vec3_Ps[j], nowNs[j], Pminus[i], As(j));
            }
            tot2_plus += grad_plus_i.norm2();
            tot2_minus += grad_minus_i.norm2();
            FEM_grad_Pplus[i] = grad_plus_i;
            FEM_grad_Pminus[i] = grad_minus_i;
        }
        std::cout << "FEM : tot plus = " << tot2_plus << ", tot minus = " << tot2_minus << ", tot2 = " << tot2_plus + tot2_minus << "\n";
#pragma omp parallel for
        for (int i = 0; i < totPoints; i++) {
            Vector3 grad_p_i{ 0, 0, 0 };
            for (int j = 0; j < totPoints; j++) {
                if ((Vec3_Ps[i] - Vec3_Ps[j]).norm() < epsFEM) continue;
                grad_p_i += FEM_grad_w_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j));

            }
            FEM_grad_p[i] = grad_p_i;
        }
        
        for (int i = 0; i < totPoints; i++) {
            boundary_FEM += (-W_plus[i] + W_minus[i]) * dot(FEM_grad_p[i], nowNs[i]) * As(i);
        }



        std::cout << "FEM boundary = " << boundary_FEM << "\n";
        polyscope::getPointCloud("init point clouds")->addVectorQuantity("FEM_grad " + std::to_string(iter), FEM_grad_p);
        polyscope::getPointCloud("Pplus")->addVectorQuantity("FEM_grad_Pplus", FEM_grad_Pplus);
        polyscope::getPointCloud("Pminus")->addVectorQuantity("FEM_grad_Pminus", FEM_grad_Pminus);
    }

    double fx = 0;
    return fx;
}
*/
using namespace autodiff;
autodiff::real boundary_E(const autodiff::ArrayXreal& x, const autodiff::ArrayXreal ADps, const autodiff::ArrayXreal ADplus, const autodiff::ArrayXreal ADminus, const autodiff::ArrayXreal Areas) {
    
    const autodiff::real ADeps = autodiff::real(1e-8);
    const autodiff::real ADPI = autodiff::real(3.141592653589793238462643383279);
    const autodiff::real ADepsP = autodiff::real(1e-2);
    int totPoints = Vec3_Ps.size();
    
    auto AD_W_pi_q = [=](autodiff::Array3real p, autodiff::Array3real Np, autodiff::Array3real q, autodiff::real weightA)->autodiff::real {
        autodiff::real val = 0;
        autodiff::ArrayXreal L = p - q;
        autodiff::real up = (L * Np).sum();
        autodiff::real down = pow((L * L).sum(), autodiff::real(1.5)) + ADeps / (autodiff::real(4.0) * ADPI);
        up = up * weightA;
        down = down * (autodiff::real(4.0) * ADPI);
        val = up / down;
        return val;
    };
    auto AD_grad_W_pi_q = [=](autodiff::Array3real p, autodiff::Array3real Np, autodiff::Array3real q, autodiff::real weightA)->autodiff::Array3real {
        
        autodiff::ArrayXreal L = p - q;
        autodiff::real L2 = (L * L).sum();
        autodiff::real downl = pow(L2, autodiff::real(2.5)), downr = pow(L2, autodiff::real(1.5));
        autodiff::real dx = autodiff::real(-3.0) * L.x() * (Np.x() * L.x() + Np.y() * L.y() + Np.z() * L.z()) * weightA / (autodiff::real(4.0) * ADPI * downl) + Np.x() * weightA / (autodiff::real(4.0) * ADPI * downr);
        autodiff::real dy = autodiff::real(-3.0) * L.y() * (Np.x() * L.x() + Np.y() * L.y() + Np.z() * L.z()) * weightA / (autodiff::real(4.0) * ADPI * downl) + Np.y() * weightA / (autodiff::real(4.0) * ADPI * downr);
        autodiff::real dz = autodiff::real(-3.0) * L.z() * (Np.x() * L.x() + Np.y() * L.y() + Np.z() * L.z()) * weightA / (autodiff::real(4.0) * ADPI * downl) + Np.z() * weightA / (autodiff::real(4.0) * ADPI * downr);
        autodiff::Array3real G;
        assert(G.size() == 3);
        G << dx, dy, dz;
        return G;
    };
    autodiff::ArrayXreal nowNs(ADPs.size());
    autodiff::ArrayXreal W_plus(totPoints), W_minus(totPoints);
    autodiff::ArrayXreal grad_p(totPoints * 3);

    for (int i = 0; i < totPoints; i++) {
        autodiff::real u = x(i * 2 + 0);
        autodiff::real v = x(i * 2 + 1);
        autodiff::real sinu = sin(u), cosu = cos(u), sinv = sin(v), cosv = cos(v);
        nowNs(i * 3 + 0) = sinu * cosv;
        nowNs(i * 3 + 1) = sinu * sinv;
        nowNs(i * 3 + 2) = cosu;
    }
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        autodiff::Array3real piPlus, piMinus, pi;
        piPlus << ADplus(i * 3 + 0), ADplus(i * 3 + 1), ADplus(i * 3 + 2);
        piMinus << ADminus(i * 3 + 0), ADminus(i * 3 + 1), ADminus(i * 3 + 2);
        pi << ADps(i * 3 + 0), ADps(i * 3 + 1), ADps(i * 3 + 2);
        autodiff::real _W_plus_i = autodiff::real(0), _W_minus_i = autodiff::real(0);
        autodiff::Array3real _grad_pi;
        _grad_pi << autodiff::real(0), autodiff::real(0), autodiff::real(0);

        for (int j = 0; j < totPoints; j++) {
            autodiff::Array3real pj, Npj;
            pj << ADps(j * 3 + 0), ADps(j * 3 + 1), ADps(j * 3 + 2);
            Npj << nowNs(j * 3 + 0), nowNs(j * 3 + 1), nowNs(j * 3 + 2);

            autodiff::real _ = sqrt(((pj - piPlus) * (pj - piPlus)).sum());
            if (_ >= ADepsP) {
                _W_plus_i += AD_W_pi_q(pj, Npj, piPlus, Areas(j));
            }

            _ = sqrt(((pj - piMinus) * (pj - piMinus)).sum());
            if (_ >= ADepsP) {
                _W_minus_i += AD_W_pi_q(pj, Npj, piMinus, Areas(j));
            }

            _ = sqrt(((pj - pi) * (pj - pi)).sum());
            if (_ >= ADepsP) {
                _grad_pi += AD_grad_W_pi_q(pj, Npj, pi, Areas(j));
            }
        }
        W_plus(i) = _W_plus_i;
        W_minus(i) = _W_minus_i;
        grad_p(i * 3 + 0) = _grad_pi.x();
        grad_p(i * 3 + 1) = _grad_pi.y();
        grad_p(i * 3 + 2) = _grad_pi.z();
    }
    autodiff::real E_3 = autodiff::real(0);
    for (int k = 0; k < totPoints; k++) {
        autodiff::real term1, term2;
        term1 = (autodiff::real(-1.0) * W_plus(k) * W_plus(k) + W_minus(k) * W_minus(k));
        autodiff::Array3real grad_pk, nowNpk;
        grad_pk << grad_p(k * 3 + 0), grad_p(k * 3 + 1), grad_p(k * 3 + 2);
        nowNpk << nowNs(k * 3 + 0), nowNs(k * 3 + 1), nowNs(k * 3 + 2);
        autodiff::real grad_pk_norm = sqrt((grad_pk * grad_pk).sum());
        grad_pk.x() /= grad_pk_norm;
        grad_pk.y() /= grad_pk_norm;
        grad_pk.z() /= grad_pk_norm;
        term2 = (grad_pk * nowNpk).sum();
        E_3 += term1 * term2;
    }
    return E_3;
}
int iter_AD = 0;
double func_AD(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    std::cout << "iter_AD = " << iter_AD << "\n";
    const double epsD = 5e-2;
    const double epsP = 1e-2;
    int totPoints = Vec3_Ps.size();
    std::vector<Vector3> nowNs(totPoints);
    std::vector<Vector3> grad_p(totPoints);
    std::vector<Vector3> grad_Pplus(totPoints);
    std::vector<Vector3> grad_Pminus(totPoints);
    Eigen::VectorXd W_p(totPoints);

    for (int i = 0; i < totPoints; i++) {
        double u = x(i * 2 + 0);
        double v = x(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = nowN.normalize();
        //nowNs[i] = Vec3_Ns[i];
    }
    // plus minus
    
    std::vector<Vector3> Pplus(totPoints), Pminus(totPoints);
    Eigen::VectorXd W_plus(totPoints), W_minus(totPoints);
    autodiff::ArrayXreal ADPplus(totPoints * 3), ADPminus(totPoints * 3);
    for (int i = 0; i < totPoints; i++) {
        Vector3 nowN = nowNs[i];
        double mx = -INF, mi = INF;
        for (Vector3 q : VD_Ps[i]) {
            Vector3 pq = q - Vec3_Ps[i];
            pq = pq.normalize();
            double angle = std::acos(dot(pq, nowN));
            if (angle > mx) {
                mx = angle;
                Pminus[i] = q;
            }
            if (angle < mi) {
                mi = angle;
                Pplus[i] = q;
            }
        }

        //Pplus[i] = Vec3_Ps[i] + epsD * nowN;
        //Pminus[i] = Vec3_Ps[i] - epsD * nowN;
    }
    // ADplus, minus
    for (int i = 0; i < totPoints; i++) {
        for (int j = 0; j < 3; j++) {
            ADPplus(i * 3 + j) = (autodiff::real)Pplus[i][j];
            ADPminus(i * 3 + j) = (autodiff::real)Pminus[i][j];
        }
    }

    polyscope::registerPointCloud("Pplus", Pplus);
    polyscope::registerPointCloud("Pminus", Pminus);

    double totPlusE = 0, totMinusE = 0;
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 qPlus = Pplus[i], qMinus = Pminus[i];
        double _valplus = 0, _valminus = 0, _valWp = 0;
        Vector3 tmp{ 0, 0, 0 }, grad_plus_i{ 0, 0, 0 }, grad_minus_i{ 0, 0, 0 };
        // plus
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[j] - qPlus).norm() < epsP) continue;
            auto _Plus = W_pi_q(Vec3_Ps[j], nowNs[j], qPlus, As(j));
            _valplus += _Plus.first;
            grad_plus_i += _Plus.second;
        }
        // minus
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[j] - qMinus).norm() < epsP) continue;
            auto _Minus = W_pi_q(Vec3_Ps[j], nowNs[j], qMinus, As(j));
            _valminus += _Minus.first;
            grad_minus_i += _Minus.second;
        }
        // w
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[i] - Vec3_Ps[j]).norm() < epsP) continue;
            auto now = W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j));
            _valWp += now.first;
            tmp += now.second;
        }
        totPlusE += grad_plus_i.norm2();
        totMinusE += grad_minus_i.norm2();

        W_plus(i) = _valplus;
        W_minus(i) = _valminus;
        W_p(i) = _valWp;
        grad_p[i] = tmp;
    }
    std::cout << "tot near E = " << totPlusE + totMinusE << ", plus = " << totPlusE << ", minus = " << totMinusE << "\n";

    for (int i = 0; i < x.size(); i++) {
        ADxs(i) = (autodiff::real)x(i);
    }
    autodiff::real ADfx;
    Eigen::VectorXd grad_x = autodiff::gradient(boundary_E, wrt(ADxs), at(ADxs, ADPs, ADPplus, ADPminus, Areas), ADfx);
    double fx = (double)ADfx;
    assert(grad_x.size() == x.size());
    grad = grad_x;
    std::cout << "fx = " << fx << ", grad.norm = " << grad.norm() << "\n";
    iter_AD++;
    return fx;
}

void run_nearGrid() {

    // poles
    //getRightPole();

    int totPoints = Vec3_Ps.size();
    int totSpacePoints = thinBand.size();
    Eigen::VectorXd W(totSpacePoints);
    std::vector<Vector3> grad_W(totSpacePoints);
    Eigen::VectorXd Diri_thinBand(totSpacePoints);
    Eigen::VectorXd xs = getRandomNs(totPoints);
    std::vector<Vector3> nowNs(totPoints);
    double dirichlet_deri = 0;
    {
        //xs = readuvs("../../../../data/xs/108chair/boundary_fuck_44.txt");
        //xs = readuvs("../../../../data/xs/108chair/FEM_totdirichletE_thinBand_29.txt");
        //xs = readuvs("../../../../data/xs/108chair/real_2_42_20.txt");
        //xs = readuvs("../../../../data/xs/108chair/boundary_modify_44.txt");
        //xs = readuvs("../../../../data/xs/108chair/real_3+alignE_46.txt");
        //xs = readuvs("../../../../data/xs/108chair/real_3_far_VDQSDirichletE_8e3_64.txt");
        //xs = readuvs("../../../../data/xs/108chair/boundaryDirichletE_3_without_near_align_4715.txt");

        //xs = readuvs("../../../../data/xs/108chair/108chair_6000_real_3_60.txt");
        //xs = readuvs("../../../../data/xs/108chair/108chair_6000_real_3_142.txt"); // best!
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_61.txt");
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e0_53.txt");
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e1_49.txt");
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e2_72.txt");
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_real_3+jumpE_1e2_72_try3_43_63.txt");
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_3_63.txt");
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_4_71.txt"); // best!
        //xs = readuvs("../../../../data/xs/bird_blue_7000/bird_blue_7000_scaled_real_4_try2_52_61.txt");
        
        //xs = readuvs("../../../../data/xs/thinBand2/thinBand2_scaled_real_3_64_21.txt");
        
        //xs = readuvs("../../../../data/xs/dress_poisson6000/dress_poisson6000_scaled_real_3_70.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3_try2_183.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_4_try4_198.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try1_200.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try2_151.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+jumpE_2_try3_250.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+align_try1_226.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3+align_try2_306.txt");// not bad
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_4_63.txt");
        //xs = readuvs("../../../../data/xs/WireFrame_trebol/WireFrame_trebol_scaled_real_3_1e2+jumpE_2_1_268.txt"); // not bad

        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_try1_70.txt");
        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3+align_try1_164.txt");
        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2+jumpE_2_try1_61.txt");
        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2+jumpE_2_try2_71_20_5.txt");
        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2+jumpE_2_1e2_try3_142.txt");
        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_real_3_1e2_20_90_146.txt");
        //xs = readuvs("../../../../data/xs/cup_poisson7000/cup_poisson7000_real_4_49_continue_16_11_23.txt");

        //xs = readuvs("../../../../data/xs/skull_poisson6000/skull_poisson6000_real_4_55.txt");

        //xs = readuvs("../../../../data/xs/BunnyPeel_blue_7000/BunnyPeel_blue_7000_real_4_49.txt");

        //xs = readuvs("../../../../data/xs/linkCupTop_blue_7000/linkCupTop_blue_7000_real_4_52.txt");

        //xs = readuvs("../../../../data/xs/41glass/41glass_real_4_58.txt");
        //xs = readuvs("../../../../data/xs/41glass/41glass_real_4_PCANormal_65.txt");

        //xs = readuvs("../../../../data/xs/397horse_blue_7000/397horse_blue_7000_real_4_62.txt");

        //xs = readuvs("../../../../data/xs/shellfish2_lr_blue_9200/shellfish2_lr_blue_9200_real_4_11.txt");

        //xs = readuvs("../../../../data/xs/BunnyPeel_blue_7000/BunnyPeel_blue_7000_real_4_PCANormal_58.txt");

        //xs = readuvs("../../../../data/xs/shellfish2_lr_blue_9200/shellfish2_lr_blue_9200_real_4_61.txt");

        //xs = readuvs("../../../../data/xs/kitten_blue_500_PCANormal/kitten_blue_500_PCANormal_real_4_112.txt");
        // 
        //xs = readuvs("../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000/Art-Institute-Chicago-Lion_blue_11000_real_4_70.txt");
        //xs = readuvs("../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000/Art-Institute-Chicago-Lion_blue_11000_PCANormals_real_4_42.txt");
        //xs = readuvs("../../../../data/xs/steampunk_gear_cube_small_corner_2_blue_11000/steampunk_gear_cube_small_corner_2_blue_11000_real_4_79.txt");

        //xs = readuvs("../../../../data/xs/WS0.5_4000_torus/WS0.5_4000_torus_PCANormals_real_4_65.txt");
        //xs = readuvs("../../../../data/xs/108chair6000poisson_0.5/108chair6000poisson_0.5_PCANormal_real_4_74.txt");
        //xs = readuvs("../../../../data/xs/108chair/108chair_real_4_84.txt");
        //xs = readuvs("../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000_0.5/Art-Institute-Chicago-Lion_blue_11000_0.5_real_4_14.txt");
        
        //xs = readuvs("../../../../data/xs/Art-Institute-Chicago-Lion_blue_11000_0.5/Art-Institute-Chicago-Lion_blue_11000_0.5_real_4_14_168.txt");
    }

    for (int i = 0; i < totPoints; i++) {
        double u = xs(i * 2 + 0);
        double v = xs(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = nowN;
        nowNs[i] = Vec3_Ns[i];
        //nowNs[i] = nowNs[i].normalize();
    }
    /*
#pragma omp parallel for
    for (int k = 0; k < totSpacePoints; k++) {
        Vector3 Vec3_GVi = thinBand[k];
        double val_k = 0;
        Vector3 grad_k = Vector3{ 0, 0, 0 };
        //if (Dis_thinBand[k] < gamma) continue;
        for (int i = 0; i < totPoints; i++) {
            //auto now = W_pi_q(Vec3_Ps[i], Vec3_Ns[i], Vec3_GVi, As(i));
            auto now = W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_GVi, As(i));

            val_k += now.first;
            grad_k += now.second;
        }
        W(k) = val_k;
        grad_W[k] = grad_k;
    }

//#pragma omp parallel for
    int totFarPoints = 0;
    for (int k = 0; k < totSpacePoints; k++) {
        Vector3 Vec3_GVi = thinBand[k];
        if (Dis_thinBand[k] < gamma) {
            grad_W[k] = Vector3{ 0 ,0, 0 };
            totFarPoints++;            
        }

        dirichlet_deri += grad_W[k].norm2();
        Diri_thinBand(k) = grad_W[k].norm2();
    }
    /*    
    Eigen::VectorXd GTuvs(totPoints * 2);
    for (int i = 0; i < totPoints; i++) {
        auto now = parameterizeVec3(Vec3_Ns[i]);
        GTuvs(i * 2 + 0) = now.first;
        GTuvs(i * 2 + 1) = now.second;
    }
    */
    // pre AD
    Areas.resizeLike(As);
    for (int i = 0; i < As.size(); i++) {
        Areas(i) = As(i);
    }
    ADPs.resize(Vec3_Ps.size() * 3);
    ADNs.resizeLike(ADPs);
    ADxs.resize(xs.size());
    for (int i = 0; i < Vec3_Ps.size(); i++) {
        for (int j = 0; j < 3; j++) {
            ADPs(i * 3 + j) = (autodiff::real)Vec3_Ps[i][j];
        }
    }
    for (int i = 0; i < xs.size(); i++) {
        ADxs(i) = (autodiff::real)xs(i);
    }

    //std::cout << "totFarPoints = " << totSpacePoints - totFarPoints << "\n";
    std::cout << "dirichlet_deri = " << dirichlet_deri << "\n";

    polyscope::getPointCloud("init point clouds")->addVectorQuantity("now normal", nowNs);
    //As_circle
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("init Areas", As);
    //polyscope::getPointCloud("init point clouds")->addScalarQuantity("xurui Areas", As_xurui);
    //polyscope::getPointCloud("init point clouds")->addScalarQuantity("circle Areas", As_circle);

    //polyscope::getPointCloud("thinBand")->addScalarQuantity("init W", W);
    //polyscope::getPointCloud("thinBand")->addVectorQuantity("init grad_W", grad_W);
    //polyscope::getPointCloud("thinBand")->addScalarQuantity("dirichlet E ", Diri_thinBand);
    /*
    polyscope::getPointCloud("near grid points")->addScalarQuantity("W", W);
    polyscope::getPointCloud("near grid points")->addVectorQuantity("grad_W", grad_W);
    */
    omp_set_num_threads(28);
    {
        
        LBFGSParam<double> param;  // New parameter class
        param.epsilon = 1e-8;
        param.max_iterations = 60;
        //param.max_linesearch = 10;
        // Create solver and function object
        LBFGSSolver<double> solver(param);  // New solver class

        Eigen::VectorXd X = xs; 
        //Eigen::VectorXd X = GTuvs; // ground truth uvs

        Eigen::VectorXd grad;

        //shape_A();

        func_run_nearGrid(X, grad);
        //modify(X);
        
        double fx;
        int niter = solver.minimize(func_run_nearGrid, X, fx);

        std::cout << niter << " iterations" << std::endl;
        //std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;
        

        // AD func
        /*
        double fx;
        int niter = solver.minimize(func_AD, X, fx);

        std::cout << niter << " iterations" << std::endl;
        //std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;
        */
    }
}

void run_nearPoints() {
    getNearPoints(Vec3_Ps);
    polyscope::registerPointCloud("near Point", nearPoints);
    int totPoints = Vec3_Ps.size();
    int totNearPoints = nearPoints.size();
    std::vector<Vector3> grad_nearPoints(totNearPoints);

    Eigen::VectorXd xs = getRandomNs(totPoints);
    std::vector<Vector3> nowNs = Vec3_Ns;
    for (int i = 0; i < totPoints; i++) {
        double u = xs(i * 2 + 0);
        double v = xs(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = /*nowN +*/ Vec3_Ns[i];
        nowNs[i] = nowNs[i].normalize();
    }
    double totE = 0;
#pragma omp parallel for
    for (int i = 0; i < totNearPoints; i++) {
        Vector3 grad_i{ 0, 0, 0 };
        for (int j = 0; j < totPoints; j++) {
            if ((Vec3_Ps[j] - nearPoints[i]).norm2() < 1e-2) continue;
            grad_i += W_pi_q(Vec3_Ps[j], nowNs[j], nearPoints[i], As(j)).second;
        }
        grad_nearPoints[i] = grad_i;
        totE += grad_i.norm2();
    }
    std::cout << "totE = " << totE << "\n";
    polyscope::getPointCloud("near Point")->addVectorQuantity("now normal", nowNs);
    polyscope::getPointCloud("near Point")->addVectorQuantity("grad near Points", grad_nearPoints);

}

void run_boundary() {
    iter = 0;
    const double epsD = 0.1;
    int totPoints = Vec3_Ps.size();
    Eigen::VectorXd xs = getRandomNs(totPoints);
    
    grad_Ps.resize(totPoints);
    std::vector<Vector3> grad_Pplus(totPoints), grad_Pminus(totPoints);
    std::vector<Vector3> nowNs = Vec3_Ns;
    
    for (int i = 0; i < totPoints; i++) {
        double u = xs(i * 2 + 0);
        double v = xs(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = /*nowN +*/ Vec3_Ns[i];
        nowNs[i] = nowNs[i].normalize();
    }
    
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("now normal", nowNs);

    std::vector<Vector3> Pplus(totPoints), Pminus(totPoints);
    Eigen::VectorXd W_plus(totPoints), W_minus(totPoints);
    for (int i = 0; i < totPoints; i++) {
        Vector3 nowN = nowNs[i];
        Pplus[i] = Vec3_Ps[i] + epsD * nowN;
        Pminus[i] = Vec3_Ps[i] - epsD * nowN;
    }
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 qPlus = Pplus[i], qMinus = Pminus[i];
        double _valplus = 0, _valminus = 0;
        for (int j = 0; j < totPoints; j++) {
            //if (i == j) continue;
            _valplus += W_pi_q(Vec3_Ps[j], nowNs[j], qPlus, As(j)).first;
            _valminus += W_pi_q(Vec3_Ps[j], nowNs[j], qMinus, As(j)).first;
        }
        W_plus(i) = _valplus;
        W_minus(i) = _valminus;
    }

    double boundary_diri = 0;
    double totE = 0;
#pragma omp parallel for
    for (int i = 0; i < totPoints; i++) {
        Vector3 tmp{ 0, 0, 0 }, _grad_Pplusi{ 0, 0, 0 }, _grad_Pminus{0, 0, 0};
        for (int j = 0; j < totPoints; j++) {

            //if (i == j) continue;
            
            if ((Vec3_Ps[i] - Vec3_Ps[j]).norm2() < 1e-4) {
                continue;
            }
            
            tmp += W_pi_q(Vec3_Ps[j], nowNs[j], Vec3_Ps[i], As(j)).second;
        }
        grad_Ps[i] = tmp;
        boundary_diri += (-W_plus[i] + W_minus[i]) * dot(nowNs[i], tmp);
        totE += tmp.norm2();
    }

    {// test int arround
        double totCharge = 0;
        for (int i = 0; i < totPoints; i++) {
            totCharge += dot(grad_Ps[i], nowNs[i]) * As(i);
        }
        std::cout << "tot charge = " << totCharge << "\n";
    }

    // near points
    getNearPoints(Vec3_Ps);

    std::cout << "totE = " << std::setprecision(10) << totE << "\n";
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("W plus", W_plus);
    polyscope::getPointCloud("init point clouds")->addScalarQuantity("W minus", W_minus);
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("grad_W", grad_Ps);


    polyscope::registerPointCloud("near Point", nearPoints);

    std::cout << "boundary_diri = " << std::setprecision(10) << boundary_diri << "\n";

    {
        LBFGSParam<double> param;  // New parameter class
        param.epsilon = 1e-6;
        param.max_iterations = 25;
        //param.max_linesearch = 10;
        // Create solver and function object
        LBFGSSolver<double> solver(param);  // New solver class
        Eigen::VectorXd X = xs;
        
        double fx;
        int niter = solver.minimize(func, X, fx);

        std::cout << niter << " iterations" << std::endl;
        //std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;
        
    }

}

void run_VDQS() {
    int totPoints = Vec3_Ps.size();
    int totSpacePoints = far_VDQS.size();
    Eigen::VectorXd W(totSpacePoints);
    std::vector<Vector3> grad_W(totSpacePoints);
    double dirichlet_deri = 0;

    std::random_device rd;  // 将用于获得随机数引擎的种子
    std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0);
    
    Eigen::VectorXd xs = getRandomNs(totPoints);
    std::vector<Vector3> nowNs(totPoints, Vector3{ 0, 0, 0 });

    for (int i = 0; i < totPoints; i++) {
        double u = xs(i * 2 + 0);
        double v = xs(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        //nowNs[i] = Vec3_Ns[i];
        nowNs[i] = nowN;
        nowNs[i] = nowNs[i].normalize();
    }

#pragma omp parallel for
    for (int k = 0; k < totSpacePoints; k++) {
        Vector3 Vec3_GVi = far_VDQS[k];
        double val_k = 0;
        Vector3 grad_k = Vector3{ 0, 0, 0 };
        for (int i = 0; i < totPoints; i++) {
            /*
            if ((Vec3_GVi - Vec3_Ps[i]).norm2() < 2.25) {
                continue;
            }
            */
            Vector3 nowN = nowNs[i];
            //auto now = W_pi_q(Vec3_Ps[i], Vec3_Ns[i], Vec3_GVi, As(i));
            
            auto now = W_pi_q(Vec3_Ps[i], nowN, Vec3_GVi, As(i));

            val_k += now.first;
            grad_k += now.second;
        }
        W(k) = val_k;
        grad_W[k] = grad_k;
    }
    for (int k = 0; k < totSpacePoints; k++) {
        Vector3 Vec3_GVi = far_VDQS[k];
        dirichlet_deri += grad_W[k].norm2();
        for (int i = 0; i < totPoints; i++) {
            /*
            if ((Vec3_GVi - Vec3_Ps[i]).norm2() < 2.25) {
                grad_W[k] = Vector3{ 0, 0, 0 };
                break;
            }
            */
        }
    }
    std::cout << "dirichlet_deri = " << dirichlet_deri << "\n";
    polyscope::getPointCloud("far_VDQS")->addScalarQuantity("W", W);
    polyscope::getPointCloud("far_VDQS")->addVectorQuantity("grad_W", grad_W);

}

void run() {
    std::cout << "x spacing = " << GV(0, 0) - GV(1, 0) << "\n";
    std::cout << "y spacing = " << GV(0, 1) - GV(1, 1) << "\n";
    std::cout << "z spacing = " << GV(0, 2) - GV(1, 2) << "\n";
    //std::cout << "diff = " << (Vec3_Ps[1527] - Vector3{ GV(887, 0), GV(887, 1), GV(887, 2) }).norm2() << "\n";
    int totPoints = Vec3_Ps.size();
    int totSpacePoints = GV.rows();
    Eigen::VectorXd W(totSpacePoints);
    std::vector<Vector3> grad_W(totSpacePoints);
    Eigen::VectorXd xs = getRandomNs(totPoints);
    std::vector<Vector3> nowNs(totPoints);
    double dirichlet_deri = 0;

    for (int i = 0; i < totPoints; i++) {
        double u = xs(i * 2 + 0);
        double v = xs(i * 2 + 1);
        Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };
        nowNs[i] = nowN /* + Vec3_Ns[i]*/;
        nowNs[i] = nowNs[i].normalize();
    }

#pragma omp parallel for
    for (int k = 0; k < totSpacePoints; k++) {
        Vector3 Vec3_GVi{ GV(k, 0), GV(k, 1), GV(k, 2) };
        double val_k = 0;
        Vector3 grad_k = Vector3{ 0, 0, 0 };
        for (int i = 0; i < totPoints; i++) {
            if ((Vec3_GVi - Vec3_Ps[i]).norm2() < 0.0001) {
                continue;
            }
            //auto now = W_pi_q(Vec3_Ps[i], Vec3_Ns[i], Vec3_GVi, As(i));
            auto now = W_pi_q(Vec3_Ps[i], nowNs[i], Vec3_GVi, As(i));
            
            val_k += now.first;
            grad_k += now.second;
        }
        W(k) = val_k;
        grad_W[k] = grad_k;
    }

#pragma omp parallel for
    for (int k = 0; k < totSpacePoints; k++) {
        Vector3 Vec3_GVi{ GV(k, 0), GV(k, 1), GV(k, 2) };
        for (int i = 0; i < totPoints; i++) {
            
            if ((Vec3_GVi - Vec3_Ps[i]).norm2() < 0.0001) {
                grad_W[k] = Vector3{ 0, 0, 0 };
                break;
            }
            
        }
        dirichlet_deri += grad_W[k].norm2();
    }    
    std::cout << "dirichlet_deri = " << dirichlet_deri << "\n";

    polyscope::getPointCloud("init point clouds")->addVectorQuantity("now normal", nowNs);

    polyscope::getPointCloud("space points")->addScalarQuantity("W", W);
    polyscope::getPointCloud("space points")->addVectorQuantity("grad_W", grad_W);

}

void Datapremesh(std::string filepath) {
    igl::read_triangle_mesh(filepath, meshV, meshF);
    // Sample mesh for point cloud
    //Eigen::MatrixXd Ps, Ns;
    {
        Eigen::VectorXi I;
        Eigen::MatrixXd B;
        igl::random_points_on_mesh(6000, meshV, meshF, B, I, Ps);
        Eigen::MatrixXd FN;
        igl::per_face_normals(meshV, meshF, FN);
        Ns.resize(Ps.rows(), 3);
        for (int p = 0; p < I.rows(); p++)
        {
            Ns.row(p) = FN.row(I(p));
        }
        ws_space.resize(GV.rows());
        ws_Pplus.resize(Ps.rows());
        ws_Pminus.resize(Ps.rows());

        Pplus.resizeLike(Ps);
        Pminus.resizeLike(Ps);

        std::cout << "here \n";
        /*
        for (int i = 0; i < Ps.rows(); i++) {
            ws_Pplus.row(i) = Ps.row(i) + epsD * Ns.row(i);
            ws_Pminus.row(i) = Ps.row(i) - epsD * Ns.row(i);
        }
        */
        //Pplus = Ps + epsD * Ns;
        //Pminus = Ps - epsD * Ns;
        grad_ws_Pplus.resize(ws_Pplus.size(), 3);
        grad_ws_Pminus.resize(ws_Pminus.size(), 3);
    }
    writePointCloudAsXYZEigen(std::make_pair(Ps, Ns), "../../../../data/108chair.xyz");

    igl::voxel_grid(meshV, 0, s, 1, GV, res);
    {
        ws_space.resize(GV.rows());
        grad_ws_space.resize(GV.rows(), 3);
    }

    polyscope::registerPointCloud("init point clouds", Ps);
    polyscope::getPointCloud("init point clouds")->addVectorQuantity("init normal", Ns);

    polyscope::registerPointCloud("space points", GV);
    polyscope::getPointCloud("space points")->setPointRadius(0.001);

    igl::octree(Ps, O_PI, O_CH, O_CN, O_W);
    std::cout << "here 1.5 \n";

    {
        Eigen::MatrixXi I;
        igl::knn(Ps, 15, O_PI, O_CH, O_CN, O_W, I);
        // CGAL is only used to help get point areas
        igl::copyleft::cgal::point_areas(Ps, I, Ns, As);
    }

    // Eigen trans to AD
    {
        ADPs = Eigen_to_AD(Ps);
        ADNs = Eigen_to_AD(Ns);
        Areas = Eigen_to_AD(As);
    }
    std::cout << "data processoing end ! \n";

    igl::fast_winding_number(Ps, Ns, As, O_PI, O_CH, 2, O_CM, O_R, O_EC);
    Eigen::VectorXd W;
    igl::fast_winding_number(Ps, Ns, As, O_PI, O_CH, O_CM, O_R, O_EC, GV, 2, W);
    polyscope::getPointCloud("space points")->addScalarQuantity("ground truth W", W);
}


autodiff::real calc_w_allp_q(autodiff::ArrayXreal Q_flatten) {
    // 求解 \sum w_p_i(q) for all p_i
    //Eigen::MatrixXd EigenPs = AD_to_Eigen(Ps_flatten);
    //Eigen::MatrixXd EigenNs = AD_to_Eigen(Ns_flatten);
    Eigen::MatrixXd EigenQ = AD_to_Eigen(Q_flatten);
    //Eigen::VectorXd EigenAs = AD_to_Eigen_likesize(Aeras);


    
    Eigen::VectorXd W;
    igl::fast_winding_number(Ps, Ns, As, O_PI, O_CH, O_CM, O_R, O_EC, EigenQ, 2, W);
    assert(W.size() == EigenQ.rows());
    assert(W.size() == 1);
    return (autodiff::real)W(0);
}



void getWNscalarFieldANDvector() {

    igl::fast_winding_number(Ps, Ns, As, O_PI, O_CH, 2, O_CM, O_R, O_EC);

    std::cout << "begin calc \n";
    int nX = res(0), nY = res(1), nZ = res(2);
    int tot_space_points = nX * nY * nZ;
    int tot_Ps = Ps.size();
    for (int i = 0; i < tot_space_points; i++) {
        if(i % 100 == 0)
            std::cout << "i = " << i << "\n";
        Eigen::VectorXd nowPoints(3);
        nowPoints(0) = GV(i, 0);
        nowPoints(1) = GV(i, 1);
        nowPoints(2) = GV(i, 2);

        autodiff::ArrayXreal x = Eigen_to_AD(nowPoints);
        autodiff::real fx = 0;
        Eigen::VectorXd grad_x = autodiff::gradient(calc_w_allp_q, wrt(x), at(x), fx);
        assert(grad_x.size() == 3);
        ws_space(i) = (double)fx;
        grad_ws_space(i, 0) = grad_x(0);
        grad_ws_space(i, 1) = grad_x(1);
        grad_ws_space(i, 2) = grad_x(2);
    }
    polyscope::getPointCloud("space points")->addScalarQuantity("W", ws_space);
    polyscope::getPointCloud("space points")->addVectorQuantity("grad_w", grad_ws_space);
    /*
    for (int i = 0; i < tot_Ps; i++) {
        Eigen::VectorXd nowPplus = Eigen::VectorXd(GV(i, 0), GV(i, 1), GV(i, 2));
        autodiff::ArrayXreal x = Eigen_to_AD(nowPplus);
    }
    */
    /*
    // WN MC
    Eigen::MatrixXd O_CM;
    Eigen::VectorXd O_R;
    Eigen::MatrixXd O_EC;
    igl::fast_winding_number(Ps, Ns, A, O_PI, O_CH, 2, O_CM, O_R, O_EC);
    Eigen::VectorXd W;
    igl::fast_winding_number(Ps, Ns, A, O_PI, O_CH, O_CM, O_R, O_EC, GV, 2, W);
    */
}

}