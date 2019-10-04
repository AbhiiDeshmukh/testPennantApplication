/*
 * HydroGPU.cu
 *
 *  Created on: Aug 2, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include <CL/sycl.hpp>
#include <syclct/syclct.hpp>
#include "HydroGPU.hh"

#include <cmath>
#include <cstdio>
#include <algorithm>

#include <thrust/copy.h>
#include <dpstd/algorithm>
#include <dpstd/execution>
#include <syclct/syclct_dpstd_utils.hpp>



#include "Memory.hh"
#include "Vec2.hh"

using namespace std;


const int CHUNK_SIZE = 64;

syclct::constant_memory<int, 0> gpuinit;

syclct::constant_memory<int, 0> numsch;
syclct::constant_memory<int, 0> nump;
syclct::constant_memory<int, 0> numz;
syclct::constant_memory<int, 0> nums;
syclct::constant_memory<int, 0> numc;
syclct::constant_memory<double, 0> dt;
syclct::constant_memory<double, 0> pgamma;
syclct::constant_memory<double, 0> pssmin;
syclct::constant_memory<double, 0> talfa;
syclct::constant_memory<double, 0> tssmin;
syclct::constant_memory<double, 0> qgamma;
syclct::constant_memory<double, 0> q1;
syclct::constant_memory<double, 0> q2;
syclct::constant_memory<double, 0> hcfl;
syclct::constant_memory<double, 0> hcflv;
syclct::constant_memory<double2*, 0> vfixx;
syclct::constant_memory<double2*, 0> vfixy;
syclct::constant_memory<int, 0> numbcx;
syclct::constant_memory<int, 0> numbcy;
syclct::constant_memory<double, 1> bcx(2);
syclct::constant_memory<double, 1> bcy(2);

syclct::device_memory<int, 0> numsbad;
syclct::device_memory<double, 0> dtnext;
syclct::device_memory<int, 0> idtnext;

syclct::device_memory<int, 1> schsfirst;
syclct::device_memory<int, 1> schslast;
syclct::device_memory<int, 1> schzfirst;
syclct::device_memory<int, 1> schzlast;
syclct::device_memory<int, 1> mapsp1;
syclct::device_memory<int, 1> mapsp2;
syclct::device_memory<int, 1> mapsz;
syclct::device_memory<int, 1> mapss4;
syclct::device_memory<int, 1> mapspkey;
syclct::device_memory<int, 1> mapspval;
syclct::device_memory<int, 1> mappsfirst;
syclct::device_memory<int, 1> mapssnext;
syclct::device_memory<int, 1> znump;

syclct::device_memory<double2*, 1> px;
syclct::device_memory<double2*, 1> pxp;
syclct::device_memory<double2*, 1> px0;
syclct::device_memory<double2*, 1> zx;
syclct::device_memory<double2*, 1> zxp;
syclct::device_memory<double2*, 1> pu;
syclct::device_memory<double2*, 1> pu0;
syclct::device_memory<double2*, 1> pap;
syclct::device_memory<double2*, 1> ssurf;
syclct::device_memory<double, 1> zm;
syclct::device_memory<double, 1> zr;
syclct::device_memory<double, 1> zrp;
syclct::device_memory<double, 1> ze;
syclct::device_memory<double, 1> zetot;
syclct::device_memory<double, 1> zw;
syclct::device_memory<double, 1> zwrate;
syclct::device_memory<double, 1> zp;
syclct::device_memory<double, 1> zss;
syclct::device_memory<double, 1> smf;
syclct::device_memory<double, 1> careap;
syclct::device_memory<double, 1> sareap;
syclct::device_memory<double, 1> svolp;
syclct::device_memory<double, 1> zareap;
syclct::device_memory<double, 1> zvolp;
syclct::device_memory<double, 1> sarea;
syclct::device_memory<double, 1> svol;
syclct::device_memory<double, 1> zarea;
syclct::device_memory<double, 1> zvol;
syclct::device_memory<double, 1> zvol0;
syclct::device_memory<double, 1> zdl;
syclct::device_memory<double, 1> zdu;
syclct::device_memory<double, 1> cmaswt;
syclct::device_memory<double, 1> pmaswt;
syclct::device_memory<double2*, 1> sfp;
syclct::device_memory<double2*, 1> sft;
syclct::device_memory<double2*, 1> sfq;
syclct::device_memory<double2*, 1> cftot;
syclct::device_memory<double2*, 1> pf;
syclct::device_memory<double, 1> cevol;
syclct::device_memory<double, 1> cdu;
syclct::device_memory<double, 1> cdiv;
syclct::device_memory<double2*, 1> zuc;
syclct::device_memory<double, 1> crmu;
syclct::device_memory<double2*, 1> cqe;
syclct::device_memory<double, 1> ccos;
syclct::device_memory<double, 1> cw;

syclct::shared_memory<int, 1> dss3(CHUNK_SIZE);
syclct::shared_memory<int, 1> dss4(CHUNK_SIZE);
syclct::shared_memory<double, 1> ctemp(CHUNK_SIZE);
syclct::shared_memory<double2*, 1> ctemp2(CHUNK_SIZE);

static int numschH, numpchH, numzchH;
static int *schsfirstH, *schslastH, *schzfirstH, *schzlastH;
static int *schsfirstD, *schslastD, *schzfirstD, *schzlastD;
static int *mapsp1D, *mapsp2D, *mapszD, *mapss4D, *znumpD;
static int *mapspkeyD, *mapspvalD;
static int *mappsfirstD, *mapssnextD;
static double2* *pxD, *pxpD, *px0D, *zxD, *zxpD, *puD, *pu0D, *papD,
    *ssurfD, *sfpD, *sftD, *sfqD, *cftotD, *pfD, *zucD, *cqeD;
static double *zmD, *zrD, *zrpD,
    *sareaD, *svolD, *zareaD, *zvolD, *zvol0D, *zdlD, *zduD,
    *zeD, *zetot0D, *zetotD, *zwD, *zwrateD,
    *zpD, *zssD, *smfD, *careapD, *sareapD, *svolpD, *zareapD, *zvolpD;
static double *cmaswtD, *pmaswtD;
static double *cevolD, *cduD, *cdivD, *crmuD, *ccosD, *cwD;


/*int checkCudaError(const cudaError_t err, const char* cmd)
{
    if(err) {
        printf("CUDA error in command '%s'\n", cmd); 
        //printf("Error message: %s\n", cudaGetErrorString(err)); 
    }
    return err;
}*/

//#define CHKERR(cmd) checkCudaError(cmd, #cmd)


static void advPosHalf(const int p,const double2** __restrict__ px0,const double2* __restrict__ pu0,const double dt,double2* __restrict__ pxp) 
{

    pxp[p] = syclct_operator_overloading::operator+(px0[p] , syclct_operator_overloading::operator*(pu0[p] , dt));

}


static void calcZoneCtrs(
        const int s,
        const int s0,
        const int z,
        const int p1,
        const double2** __restrict__ px,
        double2** __restrict__ zx, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double2, syclct::shared, 1> ctemp2) {

    ctemp2[s0] = px[p1];
    item_ct1.barrier();

    double2* zxtot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zxtot  = syclct_operator_overloading::operator+=(zxtot , ctemp2[sn]);
        zct += 1.;
    }
    zx[z] = syclct_operator_overloading::operator/(zxtot , zct);

}


static void calcSideVols(
    const int s,
    const int z,
    const int p1,
    const int p2,
    const double2** __restrict__ px,
    const double2** __restrict__ zx,
    double* __restrict__ sarea,
    double* __restrict__ svol, syclct::syclct_accessor<int, syclct::device, 0> numsbad)
{
    const double third = 1. / 3.;
    double sa = 0.5 * cross(syclct_operator_overloading::operator-(px[p2] , px[p1]),  syclct_operator_overloading::operator-(zx[z] , px[p1]));
    double sv = third * sa * (static_cast<const double>(px[p1].x()) + static_cast<const double>(px[p2].x()) + static_cast<const double>(zx[z].x()));
    sarea[s] = sa;
    svol[s] = sv;
    
    if (sv <= 0.) syclct::atomic_fetch_add(&numsbad, 1);
}


static void calcZoneVols(
    const int s,
    const int s0,
    const int z,
    const double* __restrict__ sarea,
    const double* __restrict__ svol,
    double* __restrict__ zarea,
    double* __restrict__ zvol, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::device, 1> mapss4)
{
    // make sure all side volumes have been stored
    item_ct1.barrier();

    double zatot = sarea[s];
    double zvtot = svol[s];
    for (int sn = mapss4[s]; sn != s; sn = mapss4[sn]) {
        zatot += sarea[sn];
        zvtot += svol[sn];
    }
    zarea[z] = zatot;
    zvol[z] = zvtot;
}


static void meshCalcCharLen(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const int* __restrict__ znump,
        const double2** __restrict__ px,
        const double2** __restrict__ zx,
        double* __restrict__ zdl, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double, syclct::shared, 1> ctemp) {

    double area = 0.5 * cross(syclct_operator_overloading::operator-(px[p2] , px[p1]), syclct_operator_overloading::operator-(zx[z] , px[p1]));
    double base = length(syclct_operator_overloading::operator-(px[p2] , px[p1]));
    double fac = (znump[z] == 3 ? 3. : 4.);
    double sdl = fac * area / base;

    ctemp[s0] = sdl;
    item_ct1.barrier();
    double sdlmin = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        sdlmin = cl::sycl::min(sdlmin, ctemp[sn]);
    }
    zdl[z] = sdlmin;
}

static void hydroCalcRho(const int z,
        const double* __restrict__ zm,
        const double* __restrict__ zvol,
        double* __restrict__ zr)
{
    zr[z] = zm[z] / zvol[z];
}


static void pgasCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zp,
        const double2** __restrict__ ssurf,
        double2** __restrict__ sf) {
    sf[s] = syclct_operator_overloading::operator*(-zp[z] , ssurf[s]);
}


static void ttsCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zarea,
        const double* __restrict__ zr,
        const double* __restrict__ zss,
        const double* __restrict__ sarea,
        const double* __restrict__ smf,
        const double2** __restrict__ ssurf,
        double2** __restrict__ sf, syclct::syclct_accessor<double, syclct::constant, 0> talfa, syclct::syclct_accessor<double, syclct::constant, 0> tssmin) {
    double svfacinv = zarea[z] / sarea[s];
    double srho = zr[z] * smf[s] * svfacinv;
    double sstmp = cl::sycl::max(zss[z], tssmin);
    sstmp = (double)talfa * sstmp * sstmp;
    double sdp = sstmp * (srho - zr[z]);
    sf[s] = syclct_operator_overloading::operator*(-sdp , ssurf[s]);
}


// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
static void qcsSetCornerDiv(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::device, 1> mapsp1, syclct::syclct_accessor<double2*, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> zxp, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double, syclct::device, 1> careap, syclct::syclct_accessor<double, syclct::device, 1> cevol, syclct::syclct_accessor<double, syclct::device, 1> cdu, syclct::syclct_accessor<double, syclct::device, 1> cdiv, syclct::syclct_accessor<double2, syclct::device, 1> zuc, syclct::syclct_accessor<double, syclct::device, 1> ccos, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double2, syclct::shared, 1> ctemp2) {

    // [1] Compute a zone-centered velocity
    ctemp2[s0] = pu[p1];
    item_ct1.barrier();

    double2* zutot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zutot  = syclct_operator_overloading::operator+=(zutot , ctemp2[sn]);
        zct += 1.;
    }
    zuc[z] = syclct_operator_overloading::operator/(zutot , zct);

    // [2] Divergence at the corner
    // Associated zone, corner, point
    const int p0 = mapsp1[s3];
    double2* up0 = pu[p1];
    double2* xp0 = pxp[p1];
    double2* up1 = syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(pu[p1] , pu[p2])));
    double2* xp1 = syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(pxp[p1] , pxp[p2])));
    double2* up2 = zuc[z];
    double2* xp2 = zxp[z];
    double2* up3 = syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(pu[p0] , pu[p1])));
    double2* xp3 = syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(pxp[p0] , pxp[p1])));

    // position, velocity diffs along diagonals
    double2* up2m0 = syclct_operator_overloading::operator-(up2 , up0);
    double2* xp2m0 = syclct_operator_overloading::operator-(xp2 , xp0);
    double2* up3m1 = syclct_operator_overloading::operator-(up3 , up1);
    double2* xp3m1 = syclct_operator_overloading::operator-(xp3 , xp1);

    // average corner-centered velocity
    double2* duav = syclct_operator_overloading::operator*(0.25 , (syclct_operator_overloading::operator+(syclct_operator_overloading::operator+(syclct_operator_overloading::operator+(up0 , up1) , up2) , up3)));

    // compute cosine angle
    double2* v1 = syclct_operator_overloading::operator-(xp1 , xp0);
    double2* v2 = syclct_operator_overloading::operator-(xp3 , xp0);
    double de1 = length(v1);
    double de2 = length(v2);
    double minelen = 2.0 * cl::sycl::min(de1, de2);
    ccos[s] = (minelen < 1.e-12 ? 0. : dot(v1, v2) / (de1 * de2));

    // compute 2d cartesian volume of corner
    double cvolume = 0.5 * cross(xp2m0, xp3m1);
    careap[s] = cvolume;

    // compute velocity divergence of corner
    cdiv[s] = (cross(up2m0, xp3m1) - cross(up3m1, xp2m0)) /
            (2.0 * cvolume);

    // compute delta velocity
    double dv1 = length2(syclct_operator_overloading::operator-(up2m0 , up3m1));
    double dv2 = length2(syclct_operator_overloading::operator+(up2m0 , up3m1));
    double du = cl::sycl::sqrt(cl::sycl::max(dv1, dv2));
    cdu[s]   = (cdiv[s] < 0.0 ? du   : 0.);

    // compute evolution factor
    double2* dxx1 = syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator-(xp2m0 , xp3m1)));
    double2* dxx2 = syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(xp2m0 , xp3m1)));
    double dx1 = length(dxx1);
    double dx2 = length(dxx2);

    double test1 = cl::sycl::fabs(dot(dxx1, duav) * dx2);
    double test2 = cl::sycl::fabs(dot(dxx2, duav) * dx1);
    double num = (test1 > test2 ? dx1 : dx2);
    double den = (test1 > test2 ? dx2 : dx1);
    double r = num / den;
    double evol = cl::sycl::sqrt(4.0 * cvolume * r);
    evol = cl::sycl::min(evol, (double)(2.0 * minelen));
    cevol[s] = (cdiv[s] < 0.0 ? evol : 0.);

}


// Routine number [4]  in the full algorithm CS2DQforce(...)
static void qcsSetQCnForce(
        const int s,
        const int s3,
        const int z,
        const int p1,
        const int p2, syclct::syclct_accessor<double, syclct::constant, 0> qgamma, syclct::syclct_accessor<double, syclct::constant, 0> q1, syclct::syclct_accessor<double, syclct::constant, 0> q2, syclct::syclct_accessor<int, syclct::device, 1> mapsp1, syclct::syclct_accessor<double2*, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double, syclct::device, 1> zrp, syclct::syclct_accessor<double, syclct::device, 1> zss, syclct::syclct_accessor<double, syclct::device, 1> cevol, syclct::syclct_accessor<double, syclct::device, 1> cdu, syclct::syclct_accessor<double, syclct::device, 1> cdiv, syclct::syclct_accessor<double2, syclct::device, 1> cqe) {

    const double gammap1 = (double)qgamma + 1.0;

    // [4.1] Compute the rmu (real Kurapatenko viscous scalar)
    // Kurapatenko form of the viscosity
    double ztmp2 = (double)q2 * 0.25 * gammap1 * cdu[s];
    double ztmp1 = (double)q1 * zss[z];
    double zkur = ztmp2 + cl::sycl::sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
    // Compute rmu for each corner
    double rmu = zkur * zrp[z] * cevol[s];
    rmu = (cdiv[s] > 0. ? 0. : rmu);

    // [4.2] Compute the cqe for each corner
    const int p0 = mapsp1[s3];
    const double elen1 = length(syclct_operator_overloading::operator-(pxp[p1] , pxp[p0]));
    const double elen2 = length(syclct_operator_overloading::operator-(pxp[p2] , pxp[p1]));
    // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
    //          cqe(2,1,3)=edge 2, x component (1st)
    cqe[2 * s]     = syclct_operator_overloading::operator/(syclct_operator_overloading::operator*(rmu , (syclct_operator_overloading::operator-(pu[p1] , pu[p0]))) , elen1);
    cqe[2 * s + 1] = syclct_operator_overloading::operator/(syclct_operator_overloading::operator*(rmu , (syclct_operator_overloading::operator-(pu[p2] , pu[p1]))) , elen2);
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
static void qcsSetForce(
        const int s,
        const int s4,
        const int p1,
        const int p2, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double2*, syclct::device, 1> pxp, syclct::syclct_accessor<double, syclct::device, 1> careap, syclct::syclct_accessor<double2, syclct::device, 1> sfq, syclct::syclct_accessor<double2, syclct::device, 1> cqe, syclct::syclct_accessor<double, syclct::device, 1> ccos, syclct::syclct_accessor<double, syclct::device, 1> cw) {

    // [5.1] Preparation of extra variables
    double csin2 = 1. - ccos[s] * ccos[s];
    cw[s]   = ((csin2 < 1.e-4) ? 0. : careap[s] / csin2);
    ccos[s] = ((csin2 < 1.e-4) ? 0. : ccos[s]);
    item_ct1.barrier();

    // [5.2] Set-Up the forces on corners
    const double2* x1 = pxp[p1];
    const double2* x2 = pxp[p2];
    // Edge length for c1, c2 contribution to s
    double elen = length(syclct_operator_overloading::operator-(x1 , x2));
    sfq[s] = syclct_operator_overloading::operator/((syclct_operator_overloading::operator+(syclct_operator_overloading::operator*(cw[s] , (syclct_operator_overloading::operator+(cqe[2*s+1] , syclct_operator_overloading::operator*(ccos[s] , cqe[2*s])))) ,
             syclct_operator_overloading::operator*(cw[s4] , (syclct_operator_overloading::operator+(cqe[2*s4] , syclct_operator_overloading::operator*(ccos[s4] , cqe[2*s4+1]))))))
            , elen);
}


// Routine number [6]  in the full algorithm
static void qcsSetVelDiff(
        const int s,
        const int s0,
        const int p1,
        const int p2,
        const int z, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double, syclct::constant, 0> q1, syclct::syclct_accessor<double, syclct::constant, 0> q2, syclct::syclct_accessor<double2*, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double, syclct::device, 1> zss, syclct::syclct_accessor<double, syclct::device, 1> zdu, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double, syclct::shared, 1> ctemp) {

    double2* dx = syclct_operator_overloading::operator-(pxp[p2] , pxp[p1]);
    double2* du = syclct_operator_overloading::operator-(pu[p2] , pu[p1]);
    double lenx = length(dx);
    double dux = dot(du, dx);
    dux = (lenx > 0. ? cl::sycl::fabs(dux) / lenx : 0.);

    ctemp[s0] = dux;
    item_ct1.barrier();

    double ztmp = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        ztmp = cl::sycl::max(ztmp, ctemp[sn]);
    }
    item_ct1.barrier();

    zdu[z] = (double)q1 * zss[z] + 2. * (double)q2 * ztmp;
}


static void qcsCalcForce(
        const int s,
        const int s0,
        const int s3,
        const int s4,
        const int z,
        const int p1,
        const int p2, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double, syclct::constant, 0> qgamma, syclct::syclct_accessor<double, syclct::constant, 0> q1, syclct::syclct_accessor<double, syclct::constant, 0> q2, syclct::syclct_accessor<int, syclct::device, 1> mapsp1, syclct::syclct_accessor<double2*, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> zxp, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double, syclct::device, 1> zrp, syclct::syclct_accessor<double, syclct::device, 1> zss, syclct::syclct_accessor<double, syclct::device, 1> careap, syclct::syclct_accessor<double, syclct::device, 1> zdu, syclct::syclct_accessor<double2, syclct::device, 1> sfp, syclct::syclct_accessor<double2, syclct::device, 1> sft, syclct::syclct_accessor<double2, syclct::device, 1> sfq, syclct::syclct_accessor<double2, syclct::device, 1> cftot, syclct::syclct_accessor<double, syclct::device, 1> cevol, syclct::syclct_accessor<double, syclct::device, 1> cdu, syclct::syclct_accessor<double, syclct::device, 1> cdiv, syclct::syclct_accessor<double2, syclct::device, 1> zuc, syclct::syclct_accessor<double2, syclct::device, 1> cqe, syclct::syclct_accessor<double, syclct::device, 1> ccos, syclct::syclct_accessor<double, syclct::device, 1> cw, syclct::syclct_accessor<int, syclct::shared, 1> dss3, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double, syclct::shared, 1> ctemp, syclct::syclct_accessor<double2, syclct::shared, 1> ctemp2) {
    // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv(s, s0, s3, z, p1, p2, item_ct1, mapsp1, pxp, zxp, pu, careap, cevol, cdu, cdiv, zuc, ccos, dss4, ctemp2);

    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce(s, s3, z, p1, p2, qgamma, q1, q2, mapsp1, pxp, pu, zrp, zss, cevol, cdu, cdiv, cqe);

    // [5] Compute the Q forces
    qcsSetForce(s, s4, p1, p2, item_ct1, pxp, careap, sfq, cqe, ccos, cw);

    ctemp2[s0] = syclct_operator_overloading::operator+(syclct_operator_overloading::operator+(sfp[s] , sft[s]) , sfq[s]);
    item_ct1.barrier();
    cftot[s] = syclct_operator_overloading::operator-(ctemp2[s0] , ctemp2[s0 + dss3[s0]]);

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff(s, s0, p1, p2, z, item_ct1, q1, q2, pxp, pu, zss, zdu, dss4, ctemp);

}


static void calcCrnrMass(
    const int s,
    const int s3,
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ zarea,
    const double* __restrict__ smf,
    double* __restrict__ cmaswt)
{
    double m = zr[z] * zarea[z] * 0.5 * (smf[s] + smf[s3]);
    cmaswt[s] = m;
}


static void pgasCalcEOS(
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ ze,
    double* __restrict__ zp,
    double& zper,
    double* __restrict__ zss, syclct::syclct_accessor<double, syclct::constant, 0> pgamma, syclct::syclct_accessor<double, syclct::constant, 0> pssmin)
{
    const double gm1 = (double)pgamma - 1.;
    const double ss2 = cl::sycl::max(pssmin * pssmin, 1.e-99);

    double rx = zr[z];
    double ex = cl::sycl::max(ze[z], 0.0);
    double px = gm1 * rx * ex;
    double prex = gm1 * ex;
    double perx = gm1 * rx;
    double csqd = cl::sycl::max(ss2, prex + perx * px / (rx * rx));
    zp[z] = px;
    zper = perx;
    zss[z] = cl::sycl::sqrt(csqd);
}


static void pgasCalcStateAtHalf(
    const int z,
    const double* __restrict__ zr0,
    const double* __restrict__ zvolp,
    const double* __restrict__ zvol0,
    const double* __restrict__ ze,
    const double* __restrict__ zwrate,
    const double* __restrict__ zm,
    const double dt,
    double* __restrict__ zp,
    double* __restrict__ zss, syclct::syclct_accessor<double, syclct::constant, 0> pgamma, syclct::syclct_accessor<double, syclct::constant, 0> pssmin)
{
    double zper;
    pgasCalcEOS(z, zr0, ze, zp, zper, zss, pgamma, pssmin);

    const double dth = 0.5 * dt;
    const double zminv = 1. / zm[z];
    double dv = (zvolp[z] - zvol0[z]) * zminv;
    double bulk = zr0[z] * zss[z] * zss[z];
    double denom = 1. + 0.5 * zper * dv;
    double src = zwrate[z] * dth * zminv;
    zp[z] += (zper * src - zr0[z] * bulk * dv) / denom;
}


static void gpuInvMap(
        const int* mapspkey,
        const int* mapspval,
        int* mappsfirst,
        int* mapssnext, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::constant, 0> nums)
{
    const int i = item_ct1.get_group(0) * CHUNK_SIZE + item_ct1.get_local_id(0);
    if (i >= (int)nums) return;

    int p = mapspkey[i];
    int pp = mapspkey[i+1];
    int pm = mapspkey[i-1];
    int s = mapspval[i];
    int sp = mapspval[i+1];

    if (i == 0 || p != pm)  mappsfirst[p] = s;
    if (i+1 == (int)nums || p != pp)
        mapssnext[s] = -1;
    else
        mapssnext[s] = sp;

}


static void gatherToPoints(
        const int p,
        const double* __restrict__ cvar,
        double* __restrict__ pvar, syclct::syclct_accessor<int, syclct::device, 1> mappsfirst, syclct::syclct_accessor<int, syclct::device, 1> mapssnext)
{
    double x = 0.;
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        x += cvar[s];
    }
    pvar[p] = x;
}


static void gatherToPoints(
        const int p,
        const double2** __restrict__ cvar,
        double2** __restrict__ pvar, syclct::syclct_accessor<int, syclct::device, 1> mappsfirst, syclct::syclct_accessor<int, syclct::device, 1> mapssnext)
{
    double2* x = double2(0., 0.);
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        x  = syclct_operator_overloading::operator+=(x , cvar[s]);
    }
    pvar[p] = x;
}


static void applyFixedBC(
        const int p,
        const double2** __restrict__ px,
        double2** __restrict__ pu,
        double2** __restrict__ pf,
        const double2* vfix,
        const double bcconst) {

    const double eps = 1.e-12;
    double dp = dot(px[p], vfix);

    if (cl::sycl::fabs(dp - bcconst) < eps) {
        pu[p] = project(pu[p], vfix);
        pf[p] = project(pf[p], vfix);
    }

}


static void calcAccel(
        const int p,
        const double2** __restrict__ pf,
        const double* __restrict__ pmass,
        double2** __restrict__ pa) {

    const double fuzz = 1.e-99;
    pa[p] = syclct_operator_overloading::operator/(pf[p] , cl::sycl::max(pmass[p], fuzz));

}


static void advPosFull(
        const int p,
        const double2** __restrict__ px0,
        const double2** __restrict__ pu0,
        const double2** __restrict__ pa,
        const double dt,
        double2** __restrict__ px,
        double2** __restrict__ pu) {

    pu[p] = syclct_operator_overloading::operator+(pu0[p] , syclct_operator_overloading::operator*(pa[p] , dt));
    px[p] = syclct_operator_overloading::operator+(px0[p] , syclct_operator_overloading::operator*(syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(pu[p] , pu0[p]))) , dt));

}


static void hydroCalcWork(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const double2** __restrict__ sf,
        const double2** __restrict__ sf2,
        const double2** __restrict__ pu0,
        const double2** __restrict__ pu,
        const double2** __restrict__ px,
        const double dt,
        double* __restrict__ zw,
        double* __restrict__ zetot, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double, syclct::shared, 1> ctemp) {

    // Compute the work done by finding, for each element/node pair
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    double sd1 = dot( (syclct_operator_overloading::operator+(sf[s] , sf2[s])), (syclct_operator_overloading::operator+(pu0[p1] , pu[p1])));
    double sd2 = dot(syclct_operator_overloading::operator-((syclct_operator_overloading::operator+(sf[s] , sf2[s]))), (syclct_operator_overloading::operator+(pu0[p2] , pu[p2])));
    double dwork = -0.5 * dt * (sd1 * static_cast<const double>(px[p1].x()) + sd2 * static_cast<const double>(px[p2].x()));

    ctemp[s0] = dwork;
    double etot = zetot[z];
    item_ct1.barrier();

    double dwtot = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        dwtot += ctemp[sn];
    }
    zetot[z] = etot + dwtot;
    zw[z] = dwtot;

}


static void hydroCalcWorkRate(
        const int z,
        const double* __restrict__ zvol0,
        const double* __restrict__ zvol,
        const double* __restrict__ zw,
        const double* __restrict__ zp,
        const double dt,
        double* __restrict__ zwrate) {

    double dvol = zvol[z] - zvol0[z];
    zwrate[z] = (zw[z] + zp[z] * dvol) / dt;

}


static void hydroCalcEnergy(
        const int z,
        const double* __restrict__ zetot,
        const double* __restrict__ zm,
        double* __restrict__ ze) {

    const double fuzz = 1.e-99;
    ze[z] = zetot[z] / (zm[z] + fuzz);

}


static void hydroCalcDtCourant(
        const int z,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        double& dtz,
        int& idtz, syclct::syclct_accessor<double, syclct::constant, 0> hcfl) {

    const double fuzz = 1.e-99;
    double cdu = cl::sycl::max(zdu[z], cl::sycl::max(zss[z], fuzz));
    double dtzcour = zdl[z] * (double)hcfl / cdu;
    dtz = dtzcour;
    idtz = z << 1;

}


static void hydroCalcDtVolume(
        const int z,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
        double& dtz,
        int& idtz, syclct::syclct_accessor<double, syclct::constant, 0> hcflv) {

    double zdvov = cl::sycl::fabs((zvol[z] - zvol0[z]) / zvol0[z]);
    double dtzvol = dtlast * (double)hcflv / zdvov;

    if (dtzvol < dtz) {
        dtz = dtzvol;
        idtz = (z << 1) | 1;
    }

}


static double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = syclct::atomic_compare_exchange_strong(address_as_ull, assumed,
                (unsigned long long)(syclct::bit_cast<double, long long>(cl::sycl::min(val, syclct::bit_cast<long long, double>(assumed)))));
    } while (assumed != old);
    return syclct::bit_cast<long long, double>(old);
}


static void hydroFindMinDt(
        const int z,
        const int z0,
        const int zlength,
        const double dtz,
        const int idtz,
        double& dtnext,
        int& idtnext, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double, syclct::shared, 1> ctemp, syclct::syclct_accessor<double2*, syclct::shared, 1> ctemp2) {

    int* ctempi = (int*) (&ctemp2[0]);

    ctemp[z0] = dtz;
    ctempi[z0] = idtz;
    item_ct1.barrier();

    int len = zlength;
    int half = len >> 1;
    while (z0 < half) {
        len = half + (len & 1);
        if (ctemp[z0+len] < ctemp[z0]) {
            ctemp[z0]  = ctemp[z0+len];
            ctempi[z0] = ctempi[z0+len];
        }
        item_ct1.barrier();
        half = len >> 1;
    }
    if (z0 == 0 && ctemp[0] < dtnext) {
        atomicMin(&dtnext, ctemp[0]);
        // This line isn't 100% thread-safe, but since it is only for
        // a debugging aid, I'm not going to worry about it.
        if (dtnext == ctemp[0]) idtnext = ctempi[0];
    }
}


static void hydroCalcDt(
        const int z,
        const int z0,
        const int zlength,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
        double& dtnext,
        int& idtnext, cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double, syclct::constant, 0> dt, syclct::syclct_accessor<double, syclct::constant, 0> hcfl, syclct::syclct_accessor<double, syclct::constant, 0> hcflv, syclct::syclct_accessor<double, syclct::shared, 1> ctemp, syclct::syclct_accessor<double2*, syclct::shared, 1> ctemp2) {

    double dtz;
    int idtz;
    hydroCalcDtCourant(z, zdu, zss, zdl, dtz, idtz, hcfl);
    hydroCalcDtVolume(z, zvol, zvol0, (double)dt, dtz, idtz, hcflv);
    hydroFindMinDt(z, z0, zlength, dtz, idtz, dtnext, idtnext, item_ct1, ctemp, ctemp2);

}


static void gpuMain1(cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::constant, 0> nump, syclct::syclct_accessor<double, syclct::constant, 0> dt, syclct::syclct_accessor<double2*, syclct::device, 1> px, syclct::syclct_accessor<double2, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> px0, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double2, syclct::device, 1> pu0)
{
    const int p = item_ct1.get_group(0) * CHUNK_SIZE + item_ct1.get_local_id(0);
    if (p >= (int)nump) return;

    double dth = 0.5 * (double)dt;

    // save off point variable values from previous cycle
    px0[p] = px[p];
    pu0[p] = pu[p];

    // ===== Predictor step =====
    // 1. advance mesh to center of time step
    advPosHalf(p, (double2**)px0, (double2*)pu0, dth, (double2*)pxp);

}


static void gpuMain2(cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double, syclct::constant, 0> dt, syclct::syclct_accessor<double, syclct::constant, 0> pgamma, syclct::syclct_accessor<double, syclct::constant, 0> pssmin, syclct::syclct_accessor<double, syclct::constant, 0> talfa, syclct::syclct_accessor<double, syclct::constant, 0> tssmin, syclct::syclct_accessor<double, syclct::constant, 0> qgamma, syclct::syclct_accessor<double, syclct::constant, 0> q1, syclct::syclct_accessor<double, syclct::constant, 0> q2, syclct::syclct_accessor<int, syclct::device, 0> numsbad, syclct::syclct_accessor<int, syclct::device, 1> schsfirst, syclct::syclct_accessor<int, syclct::device, 1> schslast, syclct::syclct_accessor<int, syclct::device, 1> mapsp1, syclct::syclct_accessor<int, syclct::device, 1> mapsp2, syclct::syclct_accessor<int, syclct::device, 1> mapsz, syclct::syclct_accessor<int, syclct::device, 1> mapss4, syclct::syclct_accessor<int, syclct::device, 1> znump, syclct::syclct_accessor<double2*, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> zxp, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double2, syclct::device, 1> ssurf, syclct::syclct_accessor<double, syclct::device, 1> zm, syclct::syclct_accessor<double, syclct::device, 1> zr, syclct::syclct_accessor<double, syclct::device, 1> zrp, syclct::syclct_accessor<double, syclct::device, 1> ze, syclct::syclct_accessor<double, syclct::device, 1> zwrate, syclct::syclct_accessor<double, syclct::device, 1> zp, syclct::syclct_accessor<double, syclct::device, 1> zss, syclct::syclct_accessor<double, syclct::device, 1> smf, syclct::syclct_accessor<double, syclct::device, 1> careap, syclct::syclct_accessor<double, syclct::device, 1> sareap, syclct::syclct_accessor<double, syclct::device, 1> svolp, syclct::syclct_accessor<double, syclct::device, 1> zareap, syclct::syclct_accessor<double, syclct::device, 1> zvolp, syclct::syclct_accessor<double, syclct::device, 1> zvol, syclct::syclct_accessor<double, syclct::device, 1> zvol0, syclct::syclct_accessor<double, syclct::device, 1> zdl, syclct::syclct_accessor<double, syclct::device, 1> zdu, syclct::syclct_accessor<double, syclct::device, 1> cmaswt, syclct::syclct_accessor<double2, syclct::device, 1> sfp, syclct::syclct_accessor<double2, syclct::device, 1> sft, syclct::syclct_accessor<double2, syclct::device, 1> sfq, syclct::syclct_accessor<double2, syclct::device, 1> cftot, syclct::syclct_accessor<double, syclct::device, 1> cevol, syclct::syclct_accessor<double, syclct::device, 1> cdu, syclct::syclct_accessor<double, syclct::device, 1> cdiv, syclct::syclct_accessor<double2, syclct::device, 1> zuc, syclct::syclct_accessor<double2, syclct::device, 1> cqe, syclct::syclct_accessor<double, syclct::device, 1> ccos, syclct::syclct_accessor<double, syclct::device, 1> cw, syclct::syclct_accessor<int, syclct::shared, 1> dss3, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double, syclct::shared, 1> ctemp, syclct::syclct_accessor<double2, syclct::shared, 1> ctemp2)
{
    const int s0 = item_ct1.get_local_id(0);
    const int sch = item_ct1.get_group(0);
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    item_ct1.barrier();

    const int s3 = s + dss3[s0];

    // save off zone variable values from previous cycle
    zvol0[z] = zvol[z];

    // 1a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, (double2**)pxp, (double2*)zxp, item_ct1, dss4, ctemp2);
    meshCalcCharLen(s, s0, s3, z, p1, p2, (const int*)znump, (double2**)pxp, (double2*)zxp, (double*)zdl, item_ct1, dss4, ctemp);

    ssurf[s] = rotateCCW(syclct_operator_overloading::operator-(syclct_operator_overloading::operator*(0.5 , (syclct_operator_overloading::operator+(pxp[p1] , pxp[p2]))) , zxp[z]));

    calcSideVols(s, z, p1, p2, (double2**)pxp, (double2*)zxp, (double*)sareap, (double*)svolp, numsbad);
    calcZoneVols(s, s0, z, (double*)sareap, (double*)svolp, (double*)zareap, (double*)zvolp, item_ct1, mapss4);

    // 2. compute corner masses
    hydroCalcRho(z, (const double*)zm, (double*)zvolp, (double*)zrp);
    calcCrnrMass(s, s3, z, (double*)zrp, (double*)zareap, (const double*)smf, (double*)cmaswt);

    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    if (s3 > s) pgasCalcStateAtHalf(z, (double*)zr, (double*)zvolp, (double*)zvol0, (double*)ze, (double*)zwrate,
            (const double*)zm, (double)dt, (double*)zp, (double*)zss, pgamma, pssmin);
    item_ct1.barrier();

    // 4. compute forces
    pgasCalcForce(s, z, (double*)zp, (double2**)ssurf, (double2*)sfp);
    ttsCalcForce(s, z, (double*)zareap, (double*)zrp, (double*)zss, (double*)sareap, (const double*)smf, (double2**)ssurf, (double2*)sft, talfa, tssmin);
    qcsCalcForce(s, s0, s3, s4, z, p1, p2, item_ct1, qgamma, q1, q2, mapsp1, pxp, zxp, pu, zrp, zss, careap, zdu, sfp, sft, sfq, cftot, cevol, cdu, cdiv, zuc, cqe, ccos, cw, dss3, dss4, ctemp, ctemp2);

}


static void gpuMain3(cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::constant, 0> nump, syclct::syclct_accessor<double, syclct::constant, 0> dt, syclct::syclct_accessor<double2*, syclct::constant, 0> vfixx, syclct::syclct_accessor<double2, syclct::constant, 0> vfixy, syclct::syclct_accessor<int, syclct::constant, 0> numbcx, syclct::syclct_accessor<int, syclct::constant, 0> numbcy, syclct::syclct_accessor<double, syclct::constant, 1> bcx, syclct::syclct_accessor<double, syclct::constant, 1> bcy, syclct::syclct_accessor<int, syclct::device, 1> mappsfirst, syclct::syclct_accessor<int, syclct::device, 1> mapssnext, syclct::syclct_accessor<double2, syclct::device, 1> px, syclct::syclct_accessor<double2, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> px0, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double2, syclct::device, 1> pu0, syclct::syclct_accessor<double2, syclct::device, 1> pap, syclct::syclct_accessor<double, syclct::device, 1> cmaswt, syclct::syclct_accessor<double, syclct::device, 1> pmaswt, syclct::syclct_accessor<double2, syclct::device, 1> cftot, syclct::syclct_accessor<double2, syclct::device, 1> pf)
{
    const int p = item_ct1.get_group(0) * CHUNK_SIZE + item_ct1.get_local_id(0);
    if (p >= (int)nump) return;

    // gather corner masses, forces to points
    gatherToPoints(p, (double*)cmaswt, (double*)pmaswt, mappsfirst, mapssnext);
    gatherToPoints(p, (double2**)cftot, (double2*)pf, mappsfirst, mapssnext);

    // 4a. apply boundary conditions
    for (int bc = 0; bc < (int)numbcx; ++bc)
        applyFixedBC(p, (double2**)pxp, (double2*)pu0, (double2*)pf, vfixx, bcx[bc]);
    for (int bc = 0; bc < (int)numbcy; ++bc)
        applyFixedBC(p, (double2**)pxp, (double2*)pu0, (double2*)pf, vfixy, bcy[bc]);

    // 5. compute accelerations
    calcAccel(p, (double2**)pf, (double*)pmaswt, (double2*)pap);

    // ===== Corrector step =====
    // 6. advance mesh to end of time step
    advPosFull(p, (double2**)px0, (double2*)pu0, (double2*)pap, (double)dt, (double2*)px, (double2*)pu);

}


static void gpuMain4(cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<double, syclct::constant, 0> dt, syclct::syclct_accessor<int, syclct::device, 0> numsbad, syclct::syclct_accessor<int, syclct::device, 1> schsfirst, syclct::syclct_accessor<int, syclct::device, 1> schslast, syclct::syclct_accessor<int, syclct::device, 1> mapsp1, syclct::syclct_accessor<int, syclct::device, 1> mapsp2, syclct::syclct_accessor<int, syclct::device, 1> mapsz, syclct::syclct_accessor<int, syclct::device, 1> mapss4, syclct::syclct_accessor<double2*, syclct::device, 1> px, syclct::syclct_accessor<double2, syclct::device, 1> pxp, syclct::syclct_accessor<double2, syclct::device, 1> zx, syclct::syclct_accessor<double2, syclct::device, 1> pu, syclct::syclct_accessor<double2, syclct::device, 1> pu0, syclct::syclct_accessor<double, syclct::device, 1> zetot, syclct::syclct_accessor<double, syclct::device, 1> zw, syclct::syclct_accessor<double, syclct::device, 1> sarea, syclct::syclct_accessor<double, syclct::device, 1> svol, syclct::syclct_accessor<double, syclct::device, 1> zarea, syclct::syclct_accessor<double, syclct::device, 1> zvol, syclct::syclct_accessor<double2, syclct::device, 1> sfp, syclct::syclct_accessor<double2, syclct::device, 1> sfq, syclct::syclct_accessor<int, syclct::shared, 1> dss3, syclct::syclct_accessor<int, syclct::shared, 1> dss4, syclct::syclct_accessor<double, syclct::shared, 1> ctemp, syclct::syclct_accessor<double2, syclct::shared, 1> ctemp2)
{
    const int s0 = item_ct1.get_local_id(0);
    const int sch = item_ct1.get_group(0);
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    item_ct1.barrier();

    const int s3 = s + dss3[s0];

    // 6a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, (double2**)px, (double2*)zx, item_ct1, dss4, ctemp2);
    calcSideVols(s, z, p1, p2, (double2**)px, (double2*)zx, (double*)sarea, (double*)svol, numsbad);
    calcZoneVols(s, s0, z, (double*)sarea, (double*)svol, (double*)zarea, (double*)zvol, item_ct1, mapss4);

    // 7. compute work
    hydroCalcWork(s, s0, s3, z, p1, p2, (double2**)sfp, (double2*)sfq, (double2*)pu0, (double2*)pu, (double2*)pxp, (double)dt,
            (double*)zw, (double*)zetot, item_ct1, dss4, ctemp);

}


static void gpuMain5(cl::sycl::nd_item<3> item_ct1, syclct::syclct_accessor<int, syclct::constant, 0> numz, syclct::syclct_accessor<double, syclct::constant, 0> dt, syclct::syclct_accessor<double, syclct::constant, 0> hcfl, syclct::syclct_accessor<double, syclct::constant, 0> hcflv, syclct::syclct_accessor<double, syclct::device, 0> dtnext, syclct::syclct_accessor<int, syclct::device, 0> idtnext, syclct::syclct_accessor<double, syclct::device, 1> zm, syclct::syclct_accessor<double, syclct::device, 1> zr, syclct::syclct_accessor<double, syclct::device, 1> ze, syclct::syclct_accessor<double, syclct::device, 1> zetot, syclct::syclct_accessor<double, syclct::device, 1> zw, syclct::syclct_accessor<double, syclct::device, 1> zwrate, syclct::syclct_accessor<double, syclct::device, 1> zp, syclct::syclct_accessor<double, syclct::device, 1> zss, syclct::syclct_accessor<double, syclct::device, 1> zvol, syclct::syclct_accessor<double, syclct::device, 1> zvol0, syclct::syclct_accessor<double, syclct::device, 1> zdl, syclct::syclct_accessor<double, syclct::device, 1> zdu, syclct::syclct_accessor<double, syclct::shared, 1> ctemp, syclct::syclct_accessor<double2*, syclct::shared, 1> ctemp2)
{
    const int z = item_ct1.get_group(0) * CHUNK_SIZE + item_ct1.get_local_id(0);
    if (z >= (int)numz) return;

    const int z0 = item_ct1.get_local_id(0);
    const int zlength = cl::sycl::min((unsigned int)(CHUNK_SIZE), (unsigned int)(numz - item_ct1.get_group(0) * CHUNK_SIZE));

    // 7. compute work
    hydroCalcWorkRate(z, (double*)zvol0, (double*)zvol, (double*)zw, (double*)zp, (double)dt, (double*)zwrate);

    // 8. update state variables
    hydroCalcEnergy(z, (double*)zetot, (const double*)zm, (double*)ze);
    hydroCalcRho(z, (const double*)zm, (double*)zvol, (double*)zr);

    // 9.  compute timestep for next cycle
    hydroCalcDt(z, z0, zlength, (double*)zdu, (double*)zss, (double*)zdl, (double*)zvol, (double*)zvol0, (double)dt,
            dtnext, idtnext, item_ct1, dt, hcfl, hcflv, ctemp, ctemp2);

}


void meshCheckBadSides() {

    int numsbadH;
    syclct::sycl_memcpy_from_symbol((void*)(&numsbadH), numsbad.get_ptr(), sizeof(int));
    // if there were negative side volumes, error exit
    if (numsbadH > 0) {
        cerr << "Error: " << numsbadH << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


void computeChunks(
        const int nums,
        const int numz,
        const int* mapsz,
        const int chunksize,
        int& numsch,
        int*& schsfirst,
        int*& schslast,
        int*& schzfirst,
        int*& schzlast) {

    int* stemp1 = Memory::alloc<int>(nums/3+1);
    int* stemp2 = Memory::alloc<int>(nums/3+1);
    int* ztemp1 = Memory::alloc<int>(nums/3+1);
    int* ztemp2 = Memory::alloc<int>(nums/3+1);

    int nsch = 0;
    int s1;
    int s2 = 0;
    while (s2 < nums) {
        s1 = s2;
        s2 = min(s2 + chunksize, nums);
        if (s2 < nums) {
            while (mapsz[s2] == mapsz[s2-1]) --s2;
        }
        stemp1[nsch] = s1;
        stemp2[nsch] = s2;
        ztemp1[nsch] = mapsz[s1];
        ztemp2[nsch] = (s2 == nums ? numz : mapsz[s2]);
        ++nsch;
    }

    numsch = nsch;
    schsfirst = Memory::alloc<int>(numsch);
    schslast  = Memory::alloc<int>(numsch);
    schzfirst = Memory::alloc<int>(numsch);
    schzlast  = Memory::alloc<int>(numsch);
    copy(stemp1, stemp1 + numsch, schsfirst);
    copy(stemp2, stemp2 + numsch, schslast);
    copy(ztemp1, ztemp1 + numsch, schzfirst);
    copy(ztemp2, ztemp2 + numsch, schzlast);

    Memory::free(stemp1);
    Memory::free(stemp2);
    Memory::free(ztemp1);
    Memory::free(ztemp2);

}


void hydroInit(
        const int numpH,
        const int numzH,
        const int numsH,
        const int numcH,
        const int numeH,
        const double pgammaH,
        const double pssminH,
        const double talfaH,
        const double tssminH,
        const double qgammaH,
        const double q1H,
        const double q2H,
        const double hcflH,
        const double hcflvH,
        const int numbcxH,
        const double* bcxH,
        const int numbcyH,
        const double* bcyH,
        const double2** pxH,
        const double2** puH,
        const double* zmH,
        const double* zrH,
        const double* zvolH,
        const double* zeH,
        const double* zetotH,
        const double* zwrateH,
        const double* smfH,
        const int* mapsp1H,
        const int* mapsp2H,
        const int* mapszH,
        const int* mapss4H,
        const int* mapseH,
        const int* znumpH) try {

    printf("Running Hydro on device...\n");

    computeChunks(numsH, numzH, mapszH, CHUNK_SIZE, numschH,
            schsfirstH, schslastH, schzfirstH, schzlastH);
    numpchH = (numpH+CHUNK_SIZE-1) / CHUNK_SIZE;
    numzchH = (numzH+CHUNK_SIZE-1) / CHUNK_SIZE;

    syclct::sycl_memcpy_to_symbol(numsch.get_ptr(), (void*)(&numschH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(nump.get_ptr(), (void*)(&numpH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(numz.get_ptr(), (void*)(&numzH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(nums.get_ptr(), (void*)(&numsH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(numc.get_ptr(), (void*)(&numcH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(pgamma.get_ptr(), (void*)(&pgammaH), sizeof(double));
    syclct::sycl_memcpy_to_symbol(pssmin.get_ptr(), (void*)(&pssminH), sizeof(double));
    syclct::sycl_memcpy_to_symbol(talfa.get_ptr(), (void*)(&talfaH), sizeof(double));
    syclct::sycl_memcpy_to_symbol(tssmin.get_ptr(), (void*)(&tssminH), sizeof(double));
    syclct::sycl_memcpy_to_symbol(qgamma.get_ptr(), (void*)(&qgammaH), sizeof(double));
    syclct::sycl_memcpy_to_symbol(q1.get_ptr(), (void*)(&q1H), sizeof(double));
    syclct::sycl_memcpy_to_symbol(q2.get_ptr(), (void*)(&q2H), sizeof(double));
    syclct::sycl_memcpy_to_symbol(hcfl.get_ptr(), (void*)(&hcflH), sizeof(double));
    syclct::sycl_memcpy_to_symbol(hcflv.get_ptr(), (void*)(&hcflvH), sizeof(double));

    const double2* vfixxH = double2(1., 0.);
    const double2* vfixyH = double2(0., 1.);
    syclct::sycl_memcpy_to_symbol(vfixx.get_ptr(), (void*)(&vfixxH), sizeof(double2*));
    syclct::sycl_memcpy_to_symbol(vfixy.get_ptr(), (void*)(&vfixyH), sizeof(double2*));
    syclct::sycl_memcpy_to_symbol(numbcx.get_ptr(), (void*)(&numbcxH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(numbcy.get_ptr(), (void*)(&numbcyH), sizeof(int));
    syclct::sycl_memcpy_to_symbol(bcx.get_ptr(), (void*)(bcxH), numbcxH*sizeof(double));
    syclct::sycl_memcpy_to_symbol(bcy.get_ptr(), (void*)(bcyH), numbcyH*sizeof(double));

    syclct::sycl_malloc(&schsfirstD, numschH*sizeof(int));
    syclct::sycl_malloc(&schslastD, numschH*sizeof(int));
    syclct::sycl_malloc(&schzfirstD, numschH*sizeof(int));
    syclct::sycl_malloc(&schzlastD, numschH*sizeof(int));
    syclct::sycl_malloc(&mapsp1D, numsH*sizeof(int));
    syclct::sycl_malloc(&mapsp2D, numsH*sizeof(int));
    syclct::sycl_malloc(&mapszD, numsH*sizeof(int));
    syclct::sycl_malloc(&mapss4D, numsH*sizeof(int));
    syclct::sycl_malloc(&znumpD, numzH*sizeof(int));

    syclct::sycl_malloc(&pxD, numpH*sizeof(double2*));
    syclct::sycl_malloc(&pxpD, numpH*sizeof(double2*));
    syclct::sycl_malloc(&px0D, numpH*sizeof(double2*));
    syclct::sycl_malloc(&zxD, numzH*sizeof(double2*));
    syclct::sycl_malloc(&zxpD, numzH*sizeof(double2*));
    syclct::sycl_malloc(&puD, numpH*sizeof(double2*));
    syclct::sycl_malloc(&pu0D, numpH*sizeof(double2*));
    syclct::sycl_malloc(&papD, numpH*sizeof(double2*));
    syclct::sycl_malloc(&ssurfD, numsH*sizeof(double2*));
    syclct::sycl_malloc(&zmD, numzH*sizeof(double));
    syclct::sycl_malloc(&zrD, numzH*sizeof(double));
    syclct::sycl_malloc(&zrpD, numzH*sizeof(double));
    syclct::sycl_malloc(&sareaD, numsH*sizeof(double));
    syclct::sycl_malloc(&svolD, numsH*sizeof(double));
    syclct::sycl_malloc(&zareaD, numzH*sizeof(double));
    syclct::sycl_malloc(&zvolD, numzH*sizeof(double));
    syclct::sycl_malloc(&zvol0D, numzH*sizeof(double));
    syclct::sycl_malloc(&zdlD, numzH*sizeof(double));
    syclct::sycl_malloc(&zduD, numzH*sizeof(double));
    syclct::sycl_malloc(&zeD, numzH*sizeof(double));
    syclct::sycl_malloc(&zetot0D, numzH*sizeof(double));
    syclct::sycl_malloc(&zetotD, numzH*sizeof(double));
    syclct::sycl_malloc(&zwD, numzH*sizeof(double));
    syclct::sycl_malloc(&zwrateD, numzH*sizeof(double));
    syclct::sycl_malloc(&zpD, numzH*sizeof(double));
    syclct::sycl_malloc(&zssD, numzH*sizeof(double));
    syclct::sycl_malloc(&smfD, numsH*sizeof(double));
    syclct::sycl_malloc(&careapD, numcH*sizeof(double));
    syclct::sycl_malloc(&sareapD, numsH*sizeof(double));
    syclct::sycl_malloc(&svolpD, numsH*sizeof(double));
    syclct::sycl_malloc(&zareapD, numzH*sizeof(double));
    syclct::sycl_malloc(&zvolpD, numzH*sizeof(double));
    syclct::sycl_malloc(&cmaswtD, numsH*sizeof(double));
    syclct::sycl_malloc(&pmaswtD, numpH*sizeof(double));
    syclct::sycl_malloc(&sfpD, numsH*sizeof(double2*));
    syclct::sycl_malloc(&sftD, numsH*sizeof(double2*));
    syclct::sycl_malloc(&sfqD, numsH*sizeof(double2*));
    syclct::sycl_malloc(&cftotD, numcH*sizeof(double2*));
    syclct::sycl_malloc(&pfD, numpH*sizeof(double2*));
    syclct::sycl_malloc(&cevolD, numcH*sizeof(double));
    syclct::sycl_malloc(&cduD, numcH*sizeof(double));
    syclct::sycl_malloc(&cdivD, numcH*sizeof(double));
    syclct::sycl_malloc(&zucD, numzH*sizeof(double2*));
    syclct::sycl_malloc(&crmuD, numcH*sizeof(double));
    syclct::sycl_malloc(&cqeD, 2*numcH*sizeof(double2*));
    syclct::sycl_malloc(&ccosD, numcH*sizeof(double));
    syclct::sycl_malloc(&cwD, numcH*sizeof(double));

    syclct::sycl_malloc(&mapspkeyD, numsH*sizeof(int));
    syclct::sycl_malloc(&mapspvalD, numsH*sizeof(int));
    syclct::sycl_malloc(&mappsfirstD, numpH*sizeof(int));
    syclct::sycl_malloc(&mapssnextD, numsH*sizeof(int));

    schsfirst.assign(schsfirstD, numschH*sizeof(int));
    schslast.assign(schslastD, numschH*sizeof(int));
    schzfirst.assign(schzfirstD, numschH*sizeof(int));
    schzlast.assign(schzlastD, numschH*sizeof(int));
    mapsp1.assign(mapsp1D, numsH*sizeof(int));
    mapsp2.assign(mapsp2D, numsH*sizeof(int));
    mapsz.assign(mapszD, numsH*sizeof(int));
    mapss4.assign(mapss4D, numsH*sizeof(int));
    mapspkey.assign(mapspkeyD, numsH*sizeof(int));
    mapspval.assign(mapspvalD, numsH*sizeof(int));
    mappsfirst.assign(mappsfirstD, numpH*sizeof(int));
    mapssnext.assign(mapssnextD, numsH*sizeof(int));
    znump.assign(znumpD, numzH*sizeof(int));

    px.assign(pxD, numpH*sizeof(double2*));
    pxp.assign(pxpD, numpH*sizeof(double2*));
    px0.assign(px0D, numpH*sizeof(double2*));
    zx.assign(zxD, numzH*sizeof(double2*));
    zxp.assign(zxpD, numzH*sizeof(double2*));
    pu.assign(puD, numpH*sizeof(double2*));
    pu0.assign(pu0D, numpH*sizeof(double2*));
    pap.assign(papD, numpH*sizeof(double2*));
    ssurf.assign(ssurfD, numsH*sizeof(double2*));
    zm.assign(zmD, numzH*sizeof(double));
    zr.assign(zrD, numzH*sizeof(double));
    zrp.assign(zrpD, numzH*sizeof(double));
    sarea.assign(sareaD, numsH*sizeof(double));
    svol.assign(svolD, numsH*sizeof(double));
    zarea.assign(zareaD, numzH*sizeof(double));
    zvol.assign(zvolD, numzH*sizeof(double));
    zvol0.assign(zvol0D, numzH*sizeof(double));
    zdl.assign(zdlD, numzH*sizeof(double));
    zdu.assign(zduD, numzH*sizeof(double));
    ze.assign(zeD, numzH*sizeof(double));
    zetot.assign(zetotD, numzH*sizeof(double));
    zw.assign(zwD, numzH*sizeof(double));
    zwrate.assign(zwrateD, numzH*sizeof(double));
    zp.assign(zpD, numzH*sizeof(double));
    zss.assign(zssD, numzH*sizeof(double));
    smf.assign(smfD, numsH*sizeof(double));
    careap.assign(careapD, numcH*sizeof(double));
    sareap.assign(sareapD, numsH*sizeof(double));
    svolp.assign(svolpD, numsH*sizeof(double));
    zareap.assign(zareapD, numzH*sizeof(double));
    zvolp.assign(zvolpD, numzH*sizeof(double));
    cmaswt.assign(cmaswtD, numsH*sizeof(double));
    pmaswt.assign(pmaswtD, numpH*sizeof(double));
    sfp.assign(sfpD, numsH*sizeof(double2*));
    sft.assign(sftD, numsH*sizeof(double2*));
    sfq.assign(sfqD, numsH*sizeof(double2*));
    cftot.assign(cftotD, numcH*sizeof(double2*));
    pf.assign(pfD, numpH*sizeof(double2*));
    cevol.assign(cevolD, numcH*sizeof(double));
    cdu.assign(cduD, numcH*sizeof(double));
    cdiv.assign(cdivD, numcH*sizeof(double));
    zuc.assign(zucD, numzH*sizeof(double2*));
    crmu.assign(crmuD, numcH*sizeof(double));
    cqe.assign(cqeD, 2*numcH*sizeof(double2*));
    ccos.assign(ccosD, numcH*sizeof(double));
    cw.assign(cwD, numcH*sizeof(double));

    syclct::sycl_memcpy((void*)(schsfirstD), (void*)(schsfirstH), numschH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(schslastD), (void*)(schslastH), numschH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(schzfirstD), (void*)(schzfirstH), numschH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(schzlastD), (void*)(schzlastH), numschH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(mapsp1D), (void*)(mapsp1H), numsH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(mapsp2D), (void*)(mapsp2H), numsH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(mapszD), (void*)(mapszH), numsH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(mapss4D), (void*)(mapss4H), numsH*sizeof(int), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(znumpD), (void*)(znumpH), numzH*sizeof(int), syclct::host_to_device);

    syclct::sycl_memcpy((void*)(zmD), (void*)(zmH), numzH*sizeof(double), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(smfD), (void*)(smfH), numsH*sizeof(double), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(pxD), (void*)(pxH), numpH*sizeof(double2*), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(puD), (void*)(puH), numpH*sizeof(double2*), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(zrD), (void*)(zrH), numzH*sizeof(double), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(zvolD), (void*)(zvolH), numzH*sizeof(double), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(zeD), (void*)(zeH), numzH*sizeof(double), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(zetotD), (void*)(zetotH), numzH*sizeof(double), syclct::host_to_device);
    syclct::sycl_memcpy((void*)(zwrateD), (void*)(zwrateH), numzH*sizeof(double), syclct::host_to_device);

    thrust::device_ptr<int> mapsp1T(mapsp1D);
    thrust::device_ptr<int> mapspkeyT(mapspkeyD);
    thrust::device_ptr<int> mapspvalT(mapspvalD);

    thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
    thrust::sequence(mapspvalT, mapspvalT + numsH);
    thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);

    int gridSize = (numsH+CHUNK_SIZE-1) / CHUNK_SIZE;
    int chunkSize = CHUNK_SIZE;
    {
      std::pair<syclct::buffer_t, size_t> mapspkeyD_buf = syclct::get_buffer_and_offset(mapspkeyD);
      size_t mapspkeyD_offset = mapspkeyD_buf.second;
      std::pair<syclct::buffer_t, size_t> mapspvalD_buf = syclct::get_buffer_and_offset(mapspvalD);
      size_t mapspvalD_offset = mapspvalD_buf.second;
      std::pair<syclct::buffer_t, size_t> mappsfirstD_buf = syclct::get_buffer_and_offset(mappsfirstD);
      size_t mappsfirstD_offset = mappsfirstD_buf.second;
      std::pair<syclct::buffer_t, size_t> mapssnextD_buf = syclct::get_buffer_and_offset(mapssnextD);
      size_t mapssnextD_offset = mapssnextD_buf.second;
      syclct::get_default_queue().submit(
        [&](cl::sycl::handler &cgh) {
          auto nums_acc_ct1 = nums.get_access(cgh);
          auto mapspkeyD_acc = mapspkeyD_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto mapspvalD_acc = mapspvalD_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto mappsfirstD_acc = mappsfirstD_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto mapssnextD_acc = mapssnextD_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<syclct_kernel_name<class gpuInvMap_61d594>>(
            cl::sycl::nd_range<3>((cl::sycl::range<3>(gridSize, 1, 1) * cl::sycl::range<3>(chunkSize, 1, 1)), cl::sycl::range<3>(chunkSize, 1, 1)),
            [=](cl::sycl::nd_item<3> item_ct1) {
              int *mapspkeyD = (int*)(&mapspkeyD_acc[0] + mapspkeyD_offset);
              int *mapspvalD = (int*)(&mapspvalD_acc[0] + mapspvalD_offset);
              int *mappsfirstD = (int*)(&mappsfirstD_acc[0] + mappsfirstD_offset);
              int *mapssnextD = (int*)(&mapssnextD_acc[0] + mapssnextD_offset);
              gpuInvMap(mapspkeyD, mapspvalD, mappsfirstD, mapssnextD, item_ct1, syclct::syclct_accessor<int, syclct::constant, 0>(nums_acc_ct1));
            });
        });
    }

    syclct::get_device_manager().current_device().queues_wait_and_throw();

    int zero = 0;
    syclct::sycl_memcpy_to_symbol(numsbad.get_ptr(), (void*)(&zero), sizeof(int));

}
catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
  std::exit(1);
}


void hydroDoCycle(
        const double dtH,
        double& dtnextH,
        int& idtnextH) try {
    int gridSizeS, gridSizeP, gridSizeZ, chunkSize;

    syclct::sycl_memcpy_to_symbol(dt.get_ptr(), (void*)(&dtH), sizeof(double));

    gridSizeS = numschH;
    gridSizeP = numpchH;
    gridSizeZ = numzchH;
    chunkSize = CHUNK_SIZE;

    {
      syclct::get_default_queue().submit(
        [&](cl::sycl::handler &cgh) {
          auto nump_acc_ct1 = nump.get_access(cgh);
          auto dt_acc_ct1 = dt.get_access(cgh);
          auto px_acc_ct1 = px.get_access(cgh);
          auto pxp_acc_ct1 = pxp.get_access(cgh);
          auto px0_acc_ct1 = px0.get_access(cgh);
          auto pu_acc_ct1 = pu.get_access(cgh);
          auto pu0_acc_ct1 = pu0.get_access(cgh);
          cgh.parallel_for<syclct_kernel_name<class gpuMain1_2a67d7>>(
            cl::sycl::nd_range<3>((cl::sycl::range<3>(gridSizeP, 1, 1) * cl::sycl::range<3>(chunkSize, 1, 1)), cl::sycl::range<3>(chunkSize, 1, 1)),
            [=](cl::sycl::nd_item<3> item_ct1) {
              gpuMain1(item_ct1, syclct::syclct_accessor<int, syclct::constant, 0>(nump_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(dt_acc_ct1), syclct::syclct_accessor<double2*, syclct::device, 1>(px_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pxp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(px0_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu0_acc_ct1));
            });
        });
    }

    syclct::get_device_manager().current_device().queues_wait_and_throw();

    {
      syclct::get_default_queue().submit(
        [&](cl::sycl::handler &cgh) {
          auto dt_acc_ct1 = dt.get_access(cgh);
          auto pgamma_acc_ct1 = pgamma.get_access(cgh);
          auto pssmin_acc_ct1 = pssmin.get_access(cgh);
          auto talfa_acc_ct1 = talfa.get_access(cgh);
          auto tssmin_acc_ct1 = tssmin.get_access(cgh);
          auto qgamma_acc_ct1 = qgamma.get_access(cgh);
          auto q1_acc_ct1 = q1.get_access(cgh);
          auto q2_acc_ct1 = q2.get_access(cgh);
          auto numsbad_acc_ct1 = numsbad.get_access(cgh);
          auto schsfirst_acc_ct1 = schsfirst.get_access(cgh);
          auto schslast_acc_ct1 = schslast.get_access(cgh);
          auto mapsp1_acc_ct1 = mapsp1.get_access(cgh);
          auto mapsp2_acc_ct1 = mapsp2.get_access(cgh);
          auto mapsz_acc_ct1 = mapsz.get_access(cgh);
          auto mapss4_acc_ct1 = mapss4.get_access(cgh);
          auto znump_acc_ct1 = znump.get_access(cgh);
          auto pxp_acc_ct1 = pxp.get_access(cgh);
          auto zxp_acc_ct1 = zxp.get_access(cgh);
          auto pu_acc_ct1 = pu.get_access(cgh);
          auto ssurf_acc_ct1 = ssurf.get_access(cgh);
          auto zm_acc_ct1 = zm.get_access(cgh);
          auto zr_acc_ct1 = zr.get_access(cgh);
          auto zrp_acc_ct1 = zrp.get_access(cgh);
          auto ze_acc_ct1 = ze.get_access(cgh);
          auto zwrate_acc_ct1 = zwrate.get_access(cgh);
          auto zp_acc_ct1 = zp.get_access(cgh);
          auto zss_acc_ct1 = zss.get_access(cgh);
          auto smf_acc_ct1 = smf.get_access(cgh);
          auto careap_acc_ct1 = careap.get_access(cgh);
          auto sareap_acc_ct1 = sareap.get_access(cgh);
          auto svolp_acc_ct1 = svolp.get_access(cgh);
          auto zareap_acc_ct1 = zareap.get_access(cgh);
          auto zvolp_acc_ct1 = zvolp.get_access(cgh);
          auto zvol_acc_ct1 = zvol.get_access(cgh);
          auto zvol0_acc_ct1 = zvol0.get_access(cgh);
          auto zdl_acc_ct1 = zdl.get_access(cgh);
          auto zdu_acc_ct1 = zdu.get_access(cgh);
          auto cmaswt_acc_ct1 = cmaswt.get_access(cgh);
          auto sfp_acc_ct1 = sfp.get_access(cgh);
          auto sft_acc_ct1 = sft.get_access(cgh);
          auto sfq_acc_ct1 = sfq.get_access(cgh);
          auto cftot_acc_ct1 = cftot.get_access(cgh);
          auto cevol_acc_ct1 = cevol.get_access(cgh);
          auto cdu_acc_ct1 = cdu.get_access(cgh);
          auto cdiv_acc_ct1 = cdiv.get_access(cgh);
          auto zuc_acc_ct1 = zuc.get_access(cgh);
          auto cqe_acc_ct1 = cqe.get_access(cgh);
          auto ccos_acc_ct1 = ccos.get_access(cgh);
          auto cw_acc_ct1 = cw.get_access(cgh);
          auto dss3_range_ct1 = dss3.get_range();
          auto dss3_acc_ct1 = dss3.get_access(cgh);
          auto dss4_range_ct1 = dss4.get_range();
          auto dss4_acc_ct1 = dss4.get_access(cgh);
          auto ctemp_range_ct1 = ctemp.get_range();
          auto ctemp_acc_ct1 = ctemp.get_access(cgh);
          auto ctemp2_range_ct1 = ctemp2.get_range();
          auto ctemp2_acc_ct1 = ctemp2.get_access(cgh);
          cgh.parallel_for<syclct_kernel_name<class gpuMain2_748d66>>(
            cl::sycl::nd_range<3>((cl::sycl::range<3>(gridSizeS, 1, 1) * cl::sycl::range<3>(chunkSize, 1, 1)), cl::sycl::range<3>(chunkSize, 1, 1)),
            [=](cl::sycl::nd_item<3> item_ct1) {
              gpuMain2(item_ct1, syclct::syclct_accessor<double, syclct::constant, 0>(dt_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(pgamma_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(pssmin_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(talfa_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(tssmin_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(qgamma_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(q1_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(q2_acc_ct1), syclct::syclct_accessor<int, syclct::device, 0>(numsbad_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(schsfirst_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(schslast_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapsp1_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapsp2_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapsz_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapss4_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(znump_acc_ct1), syclct::syclct_accessor<double2*, syclct::device, 1>(pxp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(zxp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(ssurf_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zm_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zr_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zrp_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(ze_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zwrate_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zp_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zss_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(smf_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(careap_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(sareap_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(svolp_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zareap_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zvolp_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zvol_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zvol0_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zdl_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zdu_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(cmaswt_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(sfp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(sft_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(sfq_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(cftot_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(cevol_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(cdu_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(cdiv_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(zuc_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(cqe_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(ccos_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(cw_acc_ct1), syclct::syclct_accessor<int, syclct::shared, 1>(dss3_acc_ct1, dss3_range_ct1), syclct::syclct_accessor<int, syclct::shared, 1>(dss4_acc_ct1, dss4_range_ct1), syclct::syclct_accessor<double, syclct::shared, 1>(ctemp_acc_ct1, ctemp_range_ct1), syclct::syclct_accessor<double2, syclct::shared, 1>(ctemp2_acc_ct1, ctemp2_range_ct1));
            });
        });
    }

    syclct::get_device_manager().current_device().queues_wait_and_throw();
    meshCheckBadSides();

    {
      syclct::get_default_queue().submit(
        [&](cl::sycl::handler &cgh) {
          auto nump_acc_ct1 = nump.get_access(cgh);
          auto dt_acc_ct1 = dt.get_access(cgh);
          auto vfixx_acc_ct1 = vfixx.get_access(cgh);
          auto vfixy_acc_ct1 = vfixy.get_access(cgh);
          auto numbcx_acc_ct1 = numbcx.get_access(cgh);
          auto numbcy_acc_ct1 = numbcy.get_access(cgh);
          auto bcx_acc_ct1 = bcx.get_access(cgh);
          auto bcy_acc_ct1 = bcy.get_access(cgh);
          auto mappsfirst_acc_ct1 = mappsfirst.get_access(cgh);
          auto mapssnext_acc_ct1 = mapssnext.get_access(cgh);
          auto px_acc_ct1 = px.get_access(cgh);
          auto pxp_acc_ct1 = pxp.get_access(cgh);
          auto px0_acc_ct1 = px0.get_access(cgh);
          auto pu_acc_ct1 = pu.get_access(cgh);
          auto pu0_acc_ct1 = pu0.get_access(cgh);
          auto pap_acc_ct1 = pap.get_access(cgh);
          auto cmaswt_acc_ct1 = cmaswt.get_access(cgh);
          auto pmaswt_acc_ct1 = pmaswt.get_access(cgh);
          auto cftot_acc_ct1 = cftot.get_access(cgh);
          auto pf_acc_ct1 = pf.get_access(cgh);
          cgh.parallel_for<syclct_kernel_name<class gpuMain3_801198>>(
            cl::sycl::nd_range<3>((cl::sycl::range<3>(gridSizeP, 1, 1) * cl::sycl::range<3>(chunkSize, 1, 1)), cl::sycl::range<3>(chunkSize, 1, 1)),
            [=](cl::sycl::nd_item<3> item_ct1) {
              gpuMain3(item_ct1, syclct::syclct_accessor<int, syclct::constant, 0>(nump_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(dt_acc_ct1), syclct::syclct_accessor<double2*, syclct::constant, 0>(vfixx_acc_ct1), syclct::syclct_accessor<double2, syclct::constant, 0>(vfixy_acc_ct1), syclct::syclct_accessor<int, syclct::constant, 0>(numbcx_acc_ct1), syclct::syclct_accessor<int, syclct::constant, 0>(numbcy_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 1>(bcx_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 1>(bcy_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mappsfirst_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapssnext_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(px_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pxp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(px0_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu0_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pap_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(cmaswt_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(pmaswt_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(cftot_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pf_acc_ct1));
            });
        });
    }

    syclct::get_device_manager().current_device().queues_wait_and_throw();

    double bigval = 1.e99;
    syclct::sycl_memcpy_to_symbol(dtnext.get_ptr(), (void*)(&bigval), sizeof(double));

    {
      syclct::get_default_queue().submit(
        [&](cl::sycl::handler &cgh) {
          auto dt_acc_ct1 = dt.get_access(cgh);
          auto numsbad_acc_ct1 = numsbad.get_access(cgh);
          auto schsfirst_acc_ct1 = schsfirst.get_access(cgh);
          auto schslast_acc_ct1 = schslast.get_access(cgh);
          auto mapsp1_acc_ct1 = mapsp1.get_access(cgh);
          auto mapsp2_acc_ct1 = mapsp2.get_access(cgh);
          auto mapsz_acc_ct1 = mapsz.get_access(cgh);
          auto mapss4_acc_ct1 = mapss4.get_access(cgh);
          auto px_acc_ct1 = px.get_access(cgh);
          auto pxp_acc_ct1 = pxp.get_access(cgh);
          auto zx_acc_ct1 = zx.get_access(cgh);
          auto pu_acc_ct1 = pu.get_access(cgh);
          auto pu0_acc_ct1 = pu0.get_access(cgh);
          auto zetot_acc_ct1 = zetot.get_access(cgh);
          auto zw_acc_ct1 = zw.get_access(cgh);
          auto sarea_acc_ct1 = sarea.get_access(cgh);
          auto svol_acc_ct1 = svol.get_access(cgh);
          auto zarea_acc_ct1 = zarea.get_access(cgh);
          auto zvol_acc_ct1 = zvol.get_access(cgh);
          auto sfp_acc_ct1 = sfp.get_access(cgh);
          auto sfq_acc_ct1 = sfq.get_access(cgh);
          auto dss3_range_ct1 = dss3.get_range();
          auto dss3_acc_ct1 = dss3.get_access(cgh);
          auto dss4_range_ct1 = dss4.get_range();
          auto dss4_acc_ct1 = dss4.get_access(cgh);
          auto ctemp_range_ct1 = ctemp.get_range();
          auto ctemp_acc_ct1 = ctemp.get_access(cgh);
          auto ctemp2_range_ct1 = ctemp2.get_range();
          auto ctemp2_acc_ct1 = ctemp2.get_access(cgh);
          cgh.parallel_for<syclct_kernel_name<class gpuMain4_da2afc>>(
            cl::sycl::nd_range<3>((cl::sycl::range<3>(gridSizeS, 1, 1) * cl::sycl::range<3>(chunkSize, 1, 1)), cl::sycl::range<3>(chunkSize, 1, 1)),
            [=](cl::sycl::nd_item<3> item_ct1) {
              gpuMain4(item_ct1, syclct::syclct_accessor<double, syclct::constant, 0>(dt_acc_ct1), syclct::syclct_accessor<int, syclct::device, 0>(numsbad_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(schsfirst_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(schslast_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapsp1_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapsp2_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapsz_acc_ct1), syclct::syclct_accessor<int, syclct::device, 1>(mapss4_acc_ct1), syclct::syclct_accessor<double2*, syclct::device, 1>(px_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pxp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(zx_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(pu0_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zetot_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zw_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(sarea_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(svol_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zarea_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zvol_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(sfp_acc_ct1), syclct::syclct_accessor<double2, syclct::device, 1>(sfq_acc_ct1), syclct::syclct_accessor<int, syclct::shared, 1>(dss3_acc_ct1, dss3_range_ct1), syclct::syclct_accessor<int, syclct::shared, 1>(dss4_acc_ct1, dss4_range_ct1), syclct::syclct_accessor<double, syclct::shared, 1>(ctemp_acc_ct1, ctemp_range_ct1), syclct::syclct_accessor<double2, syclct::shared, 1>(ctemp2_acc_ct1, ctemp2_range_ct1));
            });
        });
    }

    syclct::get_device_manager().current_device().queues_wait_and_throw();

    {
      syclct::get_default_queue().submit(
        [&](cl::sycl::handler &cgh) {
          auto numz_acc_ct1 = numz.get_access(cgh);
          auto dt_acc_ct1 = dt.get_access(cgh);
          auto hcfl_acc_ct1 = hcfl.get_access(cgh);
          auto hcflv_acc_ct1 = hcflv.get_access(cgh);
          auto dtnext_acc_ct1 = dtnext.get_access(cgh);
          auto idtnext_acc_ct1 = idtnext.get_access(cgh);
          auto zm_acc_ct1 = zm.get_access(cgh);
          auto zr_acc_ct1 = zr.get_access(cgh);
          auto ze_acc_ct1 = ze.get_access(cgh);
          auto zetot_acc_ct1 = zetot.get_access(cgh);
          auto zw_acc_ct1 = zw.get_access(cgh);
          auto zwrate_acc_ct1 = zwrate.get_access(cgh);
          auto zp_acc_ct1 = zp.get_access(cgh);
          auto zss_acc_ct1 = zss.get_access(cgh);
          auto zvol_acc_ct1 = zvol.get_access(cgh);
          auto zvol0_acc_ct1 = zvol0.get_access(cgh);
          auto zdl_acc_ct1 = zdl.get_access(cgh);
          auto zdu_acc_ct1 = zdu.get_access(cgh);
          auto ctemp_range_ct1 = ctemp.get_range();
          auto ctemp_acc_ct1 = ctemp.get_access(cgh);
          auto ctemp2_range_ct1 = ctemp2.get_range();
          auto ctemp2_acc_ct1 = ctemp2.get_access(cgh);
          cgh.parallel_for<syclct_kernel_name<class gpuMain5_b6b1a9>>(
            cl::sycl::nd_range<3>((cl::sycl::range<3>(gridSizeZ, 1, 1) * cl::sycl::range<3>(chunkSize, 1, 1)), cl::sycl::range<3>(chunkSize, 1, 1)),
            [=](cl::sycl::nd_item<3> item_ct1) {
              gpuMain5(item_ct1, syclct::syclct_accessor<int, syclct::constant, 0>(numz_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(dt_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(hcfl_acc_ct1), syclct::syclct_accessor<double, syclct::constant, 0>(hcflv_acc_ct1), syclct::syclct_accessor<double, syclct::device, 0>(dtnext_acc_ct1), syclct::syclct_accessor<int, syclct::device, 0>(idtnext_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zm_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zr_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(ze_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zetot_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zw_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zwrate_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zp_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zss_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zvol_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zvol0_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zdl_acc_ct1), syclct::syclct_accessor<double, syclct::device, 1>(zdu_acc_ct1), syclct::syclct_accessor<double, syclct::shared, 1>(ctemp_acc_ct1, ctemp_range_ct1), syclct::syclct_accessor<double2*, syclct::shared, 1>(ctemp2_acc_ct1, ctemp2_range_ct1));
            });
        });
    }

    syclct::get_device_manager().current_device().queues_wait_and_throw();
    meshCheckBadSides();

    syclct::sycl_memcpy_from_symbol((void*)(&dtnextH), dtnext.get_ptr(), sizeof(double));
    syclct::sycl_memcpy_from_symbol((void*)(&idtnextH), idtnext.get_ptr(), sizeof(int));

}
catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
  std::exit(1);
}


void hydroGetData(
        const int numpH,
        const int numzH,
        double2** pxH,
        double* zrH,
        double* zeH,
        double* zpH) {

    syclct::sycl_memcpy((void*)(pxH), (void*)(pxD), numpH*sizeof(double2*), syclct::device_to_host);
    syclct::sycl_memcpy((void*)(zrH), (void*)(zrD), numzH*sizeof(double), syclct::device_to_host);
    syclct::sycl_memcpy((void*)(zeH), (void*)(zeD), numzH*sizeof(double), syclct::device_to_host);
    syclct::sycl_memcpy((void*)(zpH), (void*)(zpD), numzH*sizeof(double), syclct::device_to_host);

}


void hydroInitGPU()
{
    int one = 1;

    0;
    syclct::sycl_memcpy_to_symbol(gpuinit.get_ptr(), (void*)(&one), sizeof(int));

}


void hydroFinalGPU()
{
}

