/*
* HHD head files
*
*/
#include "RW.h"
#include "VectorField.h"
#include "GreensFunction.h"
#include "Poisson.h"
#include "HHD.h"

#define USE_MAP

class VF {
public:
	VectorField<float> d, r, h;
	RGrid rgrid;
	VectorField<float> vfield;
	VF(int nX, int nY, int nZ, float dx, float dy, float dz, std::vector<float> x, std::vector<float> y, std::vector<float> z) {
        RGrid tmprgrid(nX, nY, nZ, dx, dy, dz);
		rgrid = tmprgrid;
        VectorField<float> tmpvfield(x, y, z, rgrid.dim);
		vfield = tmpvfield;
        printf("sz = %d, dim = %d, \n", vfield.sz, vfield.dim);
        printf("fuck = %.6f, %.6f, %.6f\n", (vfield.u[0] ), (vfield.v[0] ), (vfield.w[0]));
		vfield.need_magnitudes(rgrid);
		vfield.need_divcurl(rgrid);
		vfield.show_stats("vfield");

	}
	VF(int nX, int nY, float dx, float dy, std::vector<float> x, std::vector<float> y) {
        RGrid tmprgrid(nX, nY, dx, dy);
        rgrid = tmprgrid;
        VectorField<float> tmpvfield(x, y, rgrid.dim);
        vfield = tmpvfield;

		vfield.need_magnitudes(rgrid);
		vfield.need_divcurl(rgrid);
		vfield.show_stats("vfield");
	}

	void RunHelmholtzDecomposition() {
        naturalHHD<float> nhhd(vfield, rgrid, 2);


        // d
        d.compute_as_gradient_field(nhhd.D, rgrid);

        // r
        if (rgrid.dim == 2) {
            r.compute_as_gradient_field(nhhd.Ru, rgrid);
            r.rotate_J();
        }
        else if (rgrid.dim == 3) {
            r.compute_as_curl_field(nhhd.Ru, nhhd.Rv, nhhd.Rw, rgrid);
        }

        // h
        h.compute_as_harmonic_field(vfield, d, r);

        // -------------------------------------------------------
        d.need_magnitudes(rgrid);
        r.need_magnitudes(rgrid);
        h.need_magnitudes(rgrid);

        d.need_divcurl(rgrid);
        r.need_divcurl(rgrid);
        h.need_divcurl(rgrid);

        d.show_stats("d");
        r.show_stats("r");
        h.show_stats("h");
	}
};
