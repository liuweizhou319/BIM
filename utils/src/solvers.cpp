#include "solvers.h"

/*
 * Compute the inverse of a sparse diagonal matrix.
 *
 * Input: A sparse diagonal matrix <M>.
 * Returns: The inverse of M, which is also a sparse diagonal matrix.
 */
SparseMatrix<double> sparseInverseDiagonal(SparseMatrix<double>& M) {

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    SparseMatrix<double> inv(M.rows(), M.cols());
    for (int i = 0; i < M.rows(); i++) {
        tripletList.push_back(T(i, i, 1.0 / M.coeffRef(i, i)));
    }
    inv.setFromTriplets(tripletList.begin(), tripletList.end());
    return inv;
}

/*
 * Computes the residual of Ax - ¦Ëx, where x has unit norm and ¦Ë = x.Ax.
 *
 * Input: <A>, the complex sparse matrix whose eigendecomposition is being computed; and <x>, the current guess for the
 * smallest eigenvector
 * Returns: The residual
 */
double residual(const SparseMatrix<std::complex<double>>& A, const Vector<std::complex<double>>& x) {

    // TODO
    double xnorm = x.norm();

    auto y = x * (std::complex<double>(1.0 / xnorm, 0.0));
    auto r1 = A * y;
    auto r2 = y * ((y.transpose().conjugate()) * r1);



    /*
        auto cpx = x;
        cpx = cpx * (std::complex<double>(1.0 / xnorm, 0.0));
        Vector<std::complex<double> > xstar = cpx.transpose().conjugate();
        double res = (A * cpx - cpx * (xstar * A * cpx)).norm();
        std::cout << "residual = " << res << "\n";
        */

    auto r = r1 - r2;
    return r.norm();
    //return res; // placeholder
}


#include <Eigen/LU>
#include<Eigen/SparseCholesky>
/*
 * Solves Ax = ¦Ëx, where ¦Ë is the smallest nonzero eigenvalue of A, and x is the corresponding eigenvector.
 *
 * Input: <A>, the complex positive definite sparse matrix whose eigendecomposition is being computed.
 * Returns: The smallest eigenvector of A.
 */
Vector<std::complex<double>> solveInversePowerMethod(const SparseMatrix<std::complex<double>>& A) {

    // TODO
    int n = A.rows();
    std::cout << "n = " << n << "\n";
    auto cpA = A;
    Vector<std::complex<double> > x = Vector<std::complex<double> >::Random(n);
    //std::cout << "x = " << x << "\n";
    Eigen::SimplicialLLT<Eigen::SparseMatrix<std::complex<double> > > choleskyA;
    choleskyA.compute(cpA);
    std::cout << "end cholesky\n";
    while (residual(A, x) > 1e-10) {

        x = choleskyA.solve(x);
        auto tx = x;
        //x = tx - Vector<std::complex<double> >::Constant(n, tx.sum() * (std::complex<double> (1.0 / (1.0 * n), 1.0 / (1.0 * n))));
        x = tx - Vector<std::complex<double> >::Constant(n, tx.sum() * (1.0 / (1.0 * n)));
        x = x * (std::complex<double>(1.0 / x.norm(), 0.0));
        //std::cout << "re x = " << x << std::endl;
    }

    return x;
}