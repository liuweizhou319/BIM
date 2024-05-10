#include "shapeGen.h"


CircleGen2D::CircleGen2D() {
	O = Vector3{ 0, 0, 0 };
	R = 1.0;
	numPs = 100;
	ArcRatio = 1.0;
	double theta = 2.0 * PI / (1.0 * numPs);
	weightAs.resize(numPs);
	for (int i = 0; i < numPs; i++) {
		weightAs[i] = (R * 2.0 * PI) / (1.0 * numPs);
	}
	for (int i = 0; i < numPs; i++) {
		double nowTheta = theta * i;
		ps.push_back(Vector3{ std::sqrt(R) * std::cos(nowTheta), std::sqrt(R) * std::sin(nowTheta), 0 });
		normals.push_back(Vector3{ std::cos(nowTheta), std::sin(nowTheta), 0 });
	}
}

CircleGen2D::CircleGen2D(Vector3 inputO, double inputR, int inputNumPs, double InputArcRatio, int isRandom) {
	O = inputO;
	R = inputR;
	numPs = inputNumPs;
	ArcRatio = InputArcRatio;
	weightAs.resize(numPs);
	for (int i = 0; i < numPs; i++) {
		weightAs[i] = (R * 2.0 * PI) / (1.0 * numPs);
	}
	std::random_device rd;  // 将用于获得随机数引擎的种子
	std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
	std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0);
	double theta = (2.0 * PI * ArcRatio) / (1.0 * numPs);
	double weightA = (R * 2.0 * PI) / (1.0 * numPs);
	for (int i = 0; i < numPs; i++) {
		double nowTheta = theta * i;
		thetas.emplace_back(nowTheta);
		ps.push_back(Vector3{ std::sqrt(R) * std::cos(nowTheta), std::sqrt(R) * std::sin(nowTheta), 0 });
		if(isRandom == 0) // 正确的法向
			normals.push_back(Vector3{ std::cos(nowTheta), std::sin(nowTheta), 0 });
		else if (isRandom == -1) { // 纯随机法向
			double u = dis(gen), v = dis(gen);
			normals.push_back(Vector3{ std::cos(v), std::sin(v), 0 }.normalize());
		}
		else if (isRandom == -2) { // 在原始法向半平面内随机
			double _ = dis2(gen);
			nowTheta += _;
			normals.push_back(Vector3{ std::cos(nowTheta), std::sin(nowTheta), 0 });
		}

	}
}

Polygon2D::Polygon2D() {
	PolygonVs = std::vector<Vector3>{ Vector3{-0.5, 0.5, 0}, Vector3{0.5, 0.5, 0}, Vector3{0.5, -0.5, 0}, Vector3{-0.5, -0.5, 0} };
	VnumOnEdge = 50;
	for (int i = 0, szPolygonVs = PolygonVs.size(); i < szPolygonVs; i++) {
		allVs.emplace_back(PolygonVs[i]);
		Vector3 vec = PolygonVs[(i + 1) % szPolygonVs] - PolygonVs[i];
		double vecNorm = vec.norm(), d = 0;
		d = (vecNorm) / (1.0 * VnumOnEdge + 1.0);
		for (int j = 0; j < VnumOnEdge; j++) {
			allVs.emplace_back(PolygonVs[i] + (1.0 * j + 1.0) * d * vec);
		}
	}
	//std::cout << "wtf ! \n";
	for (int i = 0, szAllVs = allVs.size(); i < szAllVs; i++) {
		//normal
		//std::cout << "i = " << i << ", szAllVs = " << szAllVs << " , (i - 1 + szAllVs) % szAllVs = " << (i - 1 + szAllVs) % szAllVs << "\n";
		Vector3 vec = allVs[(i + 1) % szAllVs] - allVs[i];
		allNormals.emplace_back(Vector3{ -vec.y, vec.x, 0 }.normalize());
		//allNormals.emplace_back(vec.normalize());

		

		Vector3 vecback = allVs[i] - allVs[ (i - 1 + szAllVs) % szAllVs ];
		weightAs.emplace_back(0.5 * (vec.norm() + vecback.norm()) );
	}


	std::random_device rd;  // 将用于获得随机数引擎的种子
	std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
	std::uniform_real_distribution<> dis(0, PI * 2.0), dis2(-PI / 2.0, PI / 2.0);
	for (int i = 0, szAllVs = allVs.size(); i < szAllVs; i++) {
		double u = dis(gen), v = dis2(gen);
		thetas.emplace_back(std::atan2(allNormals[i].y, allNormals[i].x));
		/*
		for (int j = 0; j < PolygonVs.size(); j++) {
			if (PolygonVs[j] == allVs[i]) {
				thetas[i] += PI / 4.0;
				break;
			}
		}
		*/
	}
	
}


Polygon2D::Polygon2D(std::vector<Vector3> InputPolygonVs, int InputVnumOnEdge, int isOpen, int isRandom) {
	PolygonVs = InputPolygonVs;
	VnumOnEdge = InputVnumOnEdge;
	std::vector<int> numseglength;
	double length = 0;
	int szPolygonVs = InputPolygonVs.size();
	//std::cout << "szPolygonVs = " << szPolygonVs << "\n";
	for (int i = 0; i < szPolygonVs - isOpen; i++) {
		length += (PolygonVs[(i + 1) % szPolygonVs] - PolygonVs[i]).norm();
	}
	int totps = 0;
	for (int i = 0; i < szPolygonVs - isOpen; i++) {
		double nowseglength = (PolygonVs[(i + 1) % szPolygonVs] - PolygonVs[i]).norm();
		numseglength.emplace_back(int((nowseglength / length) * (1.0 * VnumOnEdge)));
		totps += numseglength[i];
		//std::cout << "i = " << i << ", numseglength = " << numseglength[i] << "\n";
	}
	VnumOnEdge = totps + szPolygonVs;
	
	for (int i = 0; i < szPolygonVs - isOpen; i++) {
		allVs.emplace_back(PolygonVs[i]);
		Vector3 vec = PolygonVs[(i + 1) % szPolygonVs] - PolygonVs[i];
		double vecNorm = vec.norm(), d = 0;
		//std::cout << "i = " << i << ", vecnorm = " << vecNorm << "\n";
		d = (vecNorm) / (1.0 * numseglength[i] + 1.0);
		vec = vec.normalize();
		for (int j = 0; j < numseglength[i]; j++) {
			allVs.emplace_back(PolygonVs[i] + (1.0 * j + 1.0) * d * vec);
		}
	}
	for (int i = 0; i < allVs.size(); i++) {
		allVs[i] += Vector3{ 0.005, 0, 0 };
	}
	for (int i = 0, szAllVs = allVs.size(); i < szAllVs; i++) {
		//normal
		//std::cout << "i = " << i << ", szAllVs = " << szAllVs << " , (i - 1 + szAllVs) % szAllVs = " << (i - 1 + szAllVs) % szAllVs << "\n";
		Vector3 vec = allVs[(i + 1) % szAllVs] - allVs[i];
		allNormals.emplace_back(Vector3{ -vec.y, vec.x, 0 }.normalize());
		//allNormals.emplace_back(vec.normalize());



		Vector3 vecback = allVs[i] - allVs[(i - 1 + szAllVs) % szAllVs];
		weightAs.emplace_back(0.5 * (vec.norm() + vecback.norm()));
	}

	for (int i = 0, szAllVs = allVs.size(); i < szAllVs; i++) {
		thetas.emplace_back(std::atan2(allNormals[i].y, allNormals[i].x));
		
		for (int j = 0; j < PolygonVs.size(); j++) {
			if (PolygonVs[j] == allVs[i]) {
				thetas[i] += PI / 4.0;
				break;
			}
		}
		
	}

}
