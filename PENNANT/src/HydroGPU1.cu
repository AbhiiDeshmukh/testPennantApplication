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

#include "HydroGPU.hh"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
//#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "Memory.hh"
#include "Vec2.hh"

using namespace std;


const int CHUNK_SIZE = 64;

static __constant__ int gpuinit;

static __constant__ int numsch;
static __constant__ int nump;
static __constant__ int numz;
static __constant__ int nums;
static __constant__ int numc;
static __constant__ double dt;
static __constant__ double pgamma, pssmin;
static __constant__ double talfa, tssmin;
static __constant__ double qgamma, q1, q2;
static __constant__ double hcfl, hcflv;
static __constant__ double2 vfixx, vfixy;
static __constant__ int numbcx, numbcy;
static __constant__ double bcx[2], bcy[2];

static __device__ int numsbad;
static __device__ double dtnext;
static __device__ int idtnext;

static __constant__ const int* schsfirst;
static __constant__ const int* schslast;
static __constant__ const int* schzfirst;
static __constant__ const int* schzlast;
static __constant__ const int* mapsp1;
static __constant__ const int* mapsp2;
static __constant__ const int* mapsz;
static __constant__ const int* mapss4;
static __constant__ const int *mapspkey, *mapspval;
static __constant__ const int *mappsfirst, *mapssnext;
static __constant__ const int* znump;

static __constant__ double2 *px, *pxp, *px0;
static __constant__ double2 *zx, *zxp;
static __constant__ double2 *pu, *pu0;
static __constant__ double2* pap;
static __constant__ double2* ssurf;
static __constant__ const double* zm;
static __constant__ double *zr, *zrp;
static __constant__ double *ze, *zetot;
static __constant__ double *zw, *zwrate;
static __constant__ double *zp, *zss;
static __constant__ const double* smf;
static __constant__ double *careap, *sareap, *svolp, *zareap, *zvolp;
static __constant__ double *sarea, *svol, *zarea, *zvol, *zvol0;
static __constant__ double *zdl, *zdu;
static __constant__ double *cmaswt, *pmaswt;
static __constant__ double2 *sfp, *sft, *sfq, *cftot, *pf;
static __constant__ double* cevol;
static __constant__ double* cdu;
static __constant__ double* cdiv;
static __constant__ double2* zuc;
static __constant__ double* crmu;
static __constant__ double2* cqe;
static __constant__ double* ccos;
static __constant__ double* cw;

static __shared__ int dss3[CHUNK_SIZE];
static __shared__ int dss4[CHUNK_SIZE];
static __shared__ double ctemp[CHUNK_SIZE];
static __shared__ double2 ctemp2[CHUNK_SIZE];

static int numschH, numpchH, numzchH;
static int *schsfirstH, *schslastH, *schzfirstH, *schzlastH;
static int *schsfirstD, *schslastD, *schzfirstD, *schzlastD;
static int *mapsp1D, *mapsp2D, *mapszD, *mapss4D, *znumpD;
static int *mapspkeyD, *mapspvalD;
static int *mappsfirstD, *mapssnextD;
static double2 *pxD, *pxpD, *px0D, *zxD, *zxpD, *puD, *pu0D, *papD,
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


static __device__ void advPosHalf(const int p,const double2* __restrict__ px0,const double2* __restrict__ pu0,const double dt,double2* __restrict__ pxp) 
{

    pxp[p] = px0[p] + pu0[p] * dt;

}


static __device__ void calcZoneCtrs(
        const int s,
        const int s0,
        const int z,
        const int p1,
        const double2* __restrict__ px,
        double2* __restrict__ zx) {

    ctemp2[s0] = px[p1];
    __syncthreads();

    double2 zxtot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zxtot += ctemp2[sn];
        zct += 1.;
    }
    zx[z] = zxtot / zct;

}


static __device__ void calcSideVols(
    const int s,
    const int z,
    const int p1,
    const int p2,
    const double2* __restrict__ px,
    const double2* __restrict__ zx,
    double* __restrict__ sarea,
    double* __restrict__ svol)
{
    const double third = 1. / 3.;
    double sa = 0.5 * cross(px[p2] - px[p1],  zx[z] - px[p1]);
    double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
    sarea[s] = sa;
    svol[s] = sv;
    
    if (sv <= 0.) atomicAdd(&numsbad, 1);
}


static __device__ void calcZoneVols(
    const int s,
    const int s0,
    const int z,
    const double* __restrict__ sarea,
    const double* __restrict__ svol,
    double* __restrict__ zarea,
    double* __restrict__ zvol)
{
    // make sure all side volumes have been stored
    __syncthreads();

    double zatot = sarea[s];
    double zvtot = svol[s];
    for (int sn = mapss4[s]; sn != s; sn = mapss4[sn]) {
        zatot += sarea[sn];
        zvtot += svol[sn];
    }
    zarea[z] = zatot;
    zvol[z] = zvtot;
}


static __device__ void meshCalcCharLen(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const int* __restrict__ znump,
        const double2* __restrict__ px,
        const double2* __restrict__ zx,
        double* __restrict__ zdl) {

    double area = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
    double base = length(px[p2] - px[p1]);
    double fac = (znump[z] == 3 ? 3. : 4.);
    double sdl = fac * area / base;

    ctemp[s0] = sdl;
    __syncthreads();
    double sdlmin = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        sdlmin = min(sdlmin, ctemp[sn]);
    }
    zdl[z] = sdlmin;
}

static __device__ void hydroCalcRho(const int z,
        const double* __restrict__ zm,
        const double* __restrict__ zvol,
        double* __restrict__ zr)
{
    zr[z] = zm[z] / zvol[z];
}


static __device__ void pgasCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zp,
        const double2* __restrict__ ssurf,
        double2* __restrict__ sf) {
    sf[s] = -zp[z] * ssurf[s];
}


static __device__ void ttsCalcForce(
        const int s,
        const int z,
        const double* __restrict__ zarea,
        const double* __restrict__ zr,
        const double* __restrict__ zss,
        const double* __restrict__ sarea,
        const double* __restrict__ smf,
        const double2* __restrict__ ssurf,
        double2* __restrict__ sf) {
    double svfacinv = zarea[z] / sarea[s];
    double srho = zr[z] * smf[s] * svfacinv;
    double sstmp = max(zss[z], tssmin);
    sstmp = talfa * sstmp * sstmp;
    double sdp = sstmp * (srho - zr[z]);
    sf[s] = -sdp * ssurf[s];
}


// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
static __device__ void qcsSetCornerDiv(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2) {

    // [1] Compute a zone-centered velocity
    ctemp2[s0] = pu[p1];
    __syncthreads();

    double2 zutot = ctemp2[s0];
    double zct = 1.;
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        zutot += ctemp2[sn];
        zct += 1.;
    }
    zuc[z] = zutot / zct;

    // [2] Divergence at the corner
    // Associated zone, corner, point
    const int p0 = mapsp1[s3];
    double2 up0 = pu[p1];
    double2 xp0 = pxp[p1];
    double2 up1 = 0.5 * (pu[p1] + pu[p2]);
    double2 xp1 = 0.5 * (pxp[p1] + pxp[p2]);
    double2 up2 = zuc[z];
    double2 xp2 = zxp[z];
    double2 up3 = 0.5 * (pu[p0] + pu[p1]);
    double2 xp3 = 0.5 * (pxp[p0] + pxp[p1]);

    // position, velocity diffs along diagonals
    double2 up2m0 = up2 - up0;
    double2 xp2m0 = xp2 - xp0;
    double2 up3m1 = up3 - up1;
    double2 xp3m1 = xp3 - xp1;

    // average corner-centered velocity
    double2 duav = 0.25 * (up0 + up1 + up2 + up3);

    // compute cosine angle
    double2 v1 = xp1 - xp0;
    double2 v2 = xp3 - xp0;
    double de1 = length(v1);
    double de2 = length(v2);
    double minelen = 2.0 * min(de1, de2);
    ccos[s] = (minelen < 1.e-12 ? 0. : dot(v1, v2) / (de1 * de2));

    // compute 2d cartesian volume of corner
    double cvolume = 0.5 * cross(xp2m0, xp3m1);
    careap[s] = cvolume;

    // compute velocity divergence of corner
    cdiv[s] = (cross(up2m0, xp3m1) - cross(up3m1, xp2m0)) /
            (2.0 * cvolume);

    // compute delta velocity
    double dv1 = length2(up2m0 - up3m1);
    double dv2 = length2(up2m0 + up3m1);
    double du = sqrt(max(dv1, dv2));
    cdu[s]   = (cdiv[s] < 0.0 ? du   : 0.);

    // compute evolution factor
    double2 dxx1 = 0.5 * (xp2m0 - xp3m1);
    double2 dxx2 = 0.5 * (xp2m0 + xp3m1);
    double dx1 = length(dxx1);
    double dx2 = length(dxx2);

    double test1 = abs(dot(dxx1, duav) * dx2);
    double test2 = abs(dot(dxx2, duav) * dx1);
    double num = (test1 > test2 ? dx1 : dx2);
    double den = (test1 > test2 ? dx2 : dx1);
    double r = num / den;
    double evol = sqrt(4.0 * cvolume * r);
    evol = min(evol, 2.0 * minelen);
    cevol[s] = (cdiv[s] < 0.0 ? evol : 0.);

}


// Routine number [4]  in the full algorithm CS2DQforce(...)
static __device__ void qcsSetQCnForce(
        const int s,
        const int s3,
        const int z,
        const int p1,
        const int p2) {

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the rmu (real Kurapatenko viscous scalar)
    // Kurapatenko form of the viscosity
    double ztmp2 = q2 * 0.25 * gammap1 * cdu[s];
    double ztmp1 = q1 * zss[z];
    double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
    // Compute rmu for each corner
    double rmu = zkur * zrp[z] * cevol[s];
    rmu = (cdiv[s] > 0. ? 0. : rmu);

    // [4.2] Compute the cqe for each corner
    const int p0 = mapsp1[s3];
    const double elen1 = length(pxp[p1] - pxp[p0]);
    const double elen2 = length(pxp[p2] - pxp[p1]);
    // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
    //          cqe(2,1,3)=edge 2, x component (1st)
    cqe[2 * s]     = rmu * (pu[p1] - pu[p0]) / elen1;
    cqe[2 * s + 1] = rmu * (pu[p2] - pu[p1]) / elen2;
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
static __device__ void qcsSetForce(
        const int s,
        const int s4,
        const int p1,
        const int p2) {

    // [5.1] Preparation of extra variables
    double csin2 = 1. - ccos[s] * ccos[s];
    cw[s]   = ((csin2 < 1.e-4) ? 0. : careap[s] / csin2);
    ccos[s] = ((csin2 < 1.e-4) ? 0. : ccos[s]);
    __syncthreads();

    // [5.2] Set-Up the forces on corners
    const double2 x1 = pxp[p1];
    const double2 x2 = pxp[p2];
    // Edge length for c1, c2 contribution to s
    double elen = length(x1 - x2);
    sfq[s] = (cw[s] * (cqe[2*s+1] + ccos[s] * cqe[2*s]) +
             cw[s4] * (cqe[2*s4] + ccos[s4] * cqe[2*s4+1]))
            / elen;
}


// Routine number [6]  in the full algorithm
static __device__ void qcsSetVelDiff(
        const int s,
        const int s0,
        const int p1,
        const int p2,
        const int z) {

    double2 dx = pxp[p2] - pxp[p1];
    double2 du = pu[p2] - pu[p1];
    double lenx = length(dx);
    double dux = dot(du, dx);
    dux = (lenx > 0. ? abs(dux) / lenx : 0.);

    ctemp[s0] = dux;
    __syncthreads();

    double ztmp = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        ztmp = max(ztmp, ctemp[sn]);
    }
    __syncthreads();

    zdu[z] = q1 * zss[z] + 2. * q2 * ztmp;
}


static __device__ void qcsCalcForce(
        const int s,
        const int s0,
        const int s3,
        const int s4,
        const int z,
        const int p1,
        const int p2) {
    // [1] Find the right, left, top, bottom  edges to use for the
    //     limiters
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [2] Compute corner divergence and related quantities
    qcsSetCornerDiv(s, s0, s3, z, p1, p2);

    // [3] Find the limiters Psi(c)
    // *** NOT IMPLEMENTED IN PENNANT ***

    // [4] Compute the Q vector (corner based)
    qcsSetQCnForce(s, s3, z, p1, p2);

    // [5] Compute the Q forces
    qcsSetForce(s, s4, p1, p2);

    ctemp2[s0] = sfp[s] + sft[s] + sfq[s];
    __syncthreads();
    cftot[s] = ctemp2[s0] - ctemp2[s0 + dss3[s0]];

    // [6] Set velocity difference to use to compute timestep
    qcsSetVelDiff(s, s0, p1, p2, z);

}


static __device__ void calcCrnrMass(
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


static __device__ void pgasCalcEOS(
    const int z,
    const double* __restrict__ zr,
    const double* __restrict__ ze,
    double* __restrict__ zp,
    double& zper,
    double* __restrict__ zss)
{
    const double gm1 = pgamma - 1.;
    const double ss2 = max(pssmin * pssmin, 1.e-99);

    double rx = zr[z];
    double ex = max(ze[z], 0.0);
    double px = gm1 * rx * ex;
    double prex = gm1 * ex;
    double perx = gm1 * rx;
    double csqd = max(ss2, prex + perx * px / (rx * rx));
    zp[z] = px;
    zper = perx;
    zss[z] = sqrt(csqd);
}


static __device__ void pgasCalcStateAtHalf(
    const int z,
    const double* __restrict__ zr0,
    const double* __restrict__ zvolp,
    const double* __restrict__ zvol0,
    const double* __restrict__ ze,
    const double* __restrict__ zwrate,
    const double* __restrict__ zm,
    const double dt,
    double* __restrict__ zp,
    double* __restrict__ zss)
{
    double zper;
    pgasCalcEOS(z, zr0, ze, zp, zper, zss);

    const double dth = 0.5 * dt;
    const double zminv = 1. / zm[z];
    double dv = (zvolp[z] - zvol0[z]) * zminv;
    double bulk = zr0[z] * zss[z] * zss[z];
    double denom = 1. + 0.5 * zper * dv;
    double src = zwrate[z] * dth * zminv;
    zp[z] += (zper * src - zr0[z] * bulk * dv) / denom;
}


static __global__ void gpuInvMap(
        const int* mapspkey,
        const int* mapspval,
        int* mappsfirst,
        int* mapssnext)
{
    const int i = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (i >= nums) return;

    int p = mapspkey[i];
    int pp = mapspkey[i+1];
    int pm = mapspkey[i-1];
    int s = mapspval[i];
    int sp = mapspval[i+1];

    if (i == 0 || p != pm)  mappsfirst[p] = s;
    if (i+1 == nums || p != pp)
        mapssnext[s] = -1;
    else
        mapssnext[s] = sp;

}


static __device__ void gatherToPoints(
        const int p,
        const double* __restrict__ cvar,
        double* __restrict__ pvar)
{
    double x = 0.;
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        x += cvar[s];
    }
    pvar[p] = x;
}


static __device__ void gatherToPoints(
        const int p,
        const double2* __restrict__ cvar,
        double2* __restrict__ pvar)
{
    double2 x = make_double2(0., 0.);
    for (int s = mappsfirst[p]; s >= 0; s = mapssnext[s]) {
        x += cvar[s];
    }
    pvar[p] = x;
}


static __device__ void applyFixedBC(
        const int p,
        const double2* __restrict__ px,
        double2* __restrict__ pu,
        double2* __restrict__ pf,
        const double2 vfix,
        const double bcconst) {

    const double eps = 1.e-12;
    double dp = dot(px[p], vfix);

    if (fabs(dp - bcconst) < eps) {
        pu[p] = project(pu[p], vfix);
        pf[p] = project(pf[p], vfix);
    }

}


static __device__ void calcAccel(
        const int p,
        const double2* __restrict__ pf,
        const double* __restrict__ pmass,
        double2* __restrict__ pa) {

    const double fuzz = 1.e-99;
    pa[p] = pf[p] / max(pmass[p], fuzz);

}


static __device__ void advPosFull(
        const int p,
        const double2* __restrict__ px0,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pa,
        const double dt,
        double2* __restrict__ px,
        double2* __restrict__ pu) {

    pu[p] = pu0[p] + pa[p] * dt;
    px[p] = px0[p] + 0.5 * (pu[p] + pu0[p]) * dt;

}


static __device__ void hydroCalcWork(
        const int s,
        const int s0,
        const int s3,
        const int z,
        const int p1,
        const int p2,
        const double2* __restrict__ sf,
        const double2* __restrict__ sf2,
        const double2* __restrict__ pu0,
        const double2* __restrict__ pu,
        const double2* __restrict__ px,
        const double dt,
        double* __restrict__ zw,
        double* __restrict__ zetot) {

    // Compute the work done by finding, for each element/node pair
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    double sd1 = dot( (sf[s] + sf2[s]), (pu0[p1] + pu[p1]));
    double sd2 = dot(-(sf[s] + sf2[s]), (pu0[p2] + pu[p2]));
    double dwork = -0.5 * dt * (sd1 * px[p1].x + sd2 * px[p2].x);

    ctemp[s0] = dwork;
    double etot = zetot[z];
    __syncthreads();

    double dwtot = ctemp[s0];
    for (int sn = s0 + dss4[s0]; sn != s0; sn += dss4[sn]) {
        dwtot += ctemp[sn];
    }
    zetot[z] = etot + dwtot;
    zw[z] = dwtot;

}


static __device__ void hydroCalcWorkRate(
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


static __device__ void hydroCalcEnergy(
        const int z,
        const double* __restrict__ zetot,
        const double* __restrict__ zm,
        double* __restrict__ ze) {

    const double fuzz = 1.e-99;
    ze[z] = zetot[z] / (zm[z] + fuzz);

}


static __device__ void hydroCalcDtCourant(
        const int z,
        const double* __restrict__ zdu,
        const double* __restrict__ zss,
        const double* __restrict__ zdl,
        double& dtz,
        int& idtz) {

    const double fuzz = 1.e-99;
    double cdu = max(zdu[z], max(zss[z], fuzz));
    double dtzcour = zdl[z] * hcfl / cdu;
    dtz = dtzcour;
    idtz = z << 1;

}


static __device__ void hydroCalcDtVolume(
        const int z,
        const double* __restrict__ zvol,
        const double* __restrict__ zvol0,
        const double dtlast,
        double& dtz,
        int& idtz) {

    double zdvov = abs((zvol[z] - zvol0[z]) / zvol0[z]);
    double dtzvol = dtlast * hcflv / zdvov;

    if (dtzvol < dtz) {
        dtz = dtzvol;
        idtz = (z << 1) | 1;
    }

}


static __device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(min(val,
                __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


static __device__ void hydroFindMinDt(
        const int z,
        const int z0,
        const int zlength,
        const double dtz,
        const int idtz,
        double& dtnext,
        int& idtnext) {

    int* ctempi = (int*) ctemp2;

    ctemp[z0] = dtz;
    ctempi[z0] = idtz;
    __syncthreads();

    int len = zlength;
    int half = len >> 1;
    while (z0 < half) {
        len = half + (len & 1);
        if (ctemp[z0+len] < ctemp[z0]) {
            ctemp[z0]  = ctemp[z0+len];
            ctempi[z0] = ctempi[z0+len];
        }
        __syncthreads();
        half = len >> 1;
    }
    if (z0 == 0 && ctemp[0] < dtnext) {
        atomicMin(&dtnext, ctemp[0]);
        // This line isn't 100% thread-safe, but since it is only for
        // a debugging aid, I'm not going to worry about it.
        if (dtnext == ctemp[0]) idtnext = ctempi[0];
    }
}


static __device__ void hydroCalcDt(
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
        int& idtnext) {

    double dtz;
    int idtz;
    hydroCalcDtCourant(z, zdu, zss, zdl, dtz, idtz);
    hydroCalcDtVolume(z, zvol, zvol0, dt, dtz, idtz);
    hydroFindMinDt(z, z0, zlength, dtz, idtz, dtnext, idtnext);

}


static __global__ void gpuMain1()
{
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    double dth = 0.5 * dt;

    // save off point variable values from previous cycle
    px0[p] = px[p];
    pu0[p] = pu[p];

    // ===== Predictor step =====
    // 1. advance mesh to center of time step
    advPosHalf(p, px0, pu0, dth, pxp);

}


static __global__ void gpuMain2()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // save off zone variable values from previous cycle
    zvol0[z] = zvol[z];

    // 1a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, pxp, zxp);
    meshCalcCharLen(s, s0, s3, z, p1, p2, znump, pxp, zxp, zdl);

    ssurf[s] = rotateCCW(0.5 * (pxp[p1] + pxp[p2]) - zxp[z]);

    calcSideVols(s, z, p1, p2, pxp, zxp, sareap, svolp);
    calcZoneVols(s, s0, z, sareap, svolp, zareap, zvolp);

    // 2. compute corner masses
    hydroCalcRho(z, zm, zvolp, zrp);
    calcCrnrMass(s, s3, z, zrp, zareap, smf, cmaswt);

    // 3. compute material state (half-advanced)
    // call this routine from only one thread per zone
    if (s3 > s) pgasCalcStateAtHalf(z, zr, zvolp, zvol0, ze, zwrate,
            zm, dt, zp, zss);
    __syncthreads();

    // 4. compute forces
    pgasCalcForce(s, z, zp, ssurf, sfp);
    ttsCalcForce(s, z, zareap, zrp, zss, sareap, smf, ssurf, sft);
    qcsCalcForce(s, s0, s3, s4, z, p1, p2);

}


static __global__ void gpuMain3()
{
    const int p = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (p >= nump) return;

    // gather corner masses, forces to points
    gatherToPoints(p, cmaswt, pmaswt);
    gatherToPoints(p, cftot, pf);

    // 4a. apply boundary conditions
    for (int bc = 0; bc < numbcx; ++bc)
        applyFixedBC(p, pxp, pu0, pf, vfixx, bcx[bc]);
    for (int bc = 0; bc < numbcy; ++bc)
        applyFixedBC(p, pxp, pu0, pf, vfixy, bcy[bc]);

    // 5. compute accelerations
    calcAccel(p, pf, pmaswt, pap);

    // ===== Corrector step =====
    // 6. advance mesh to end of time step
    advPosFull(p, px0, pu0, pap, dt, px, pu);

}


static __global__ void gpuMain4()
{
    const int s0 = threadIdx.x;
    const int sch = blockIdx.x;
    const int s = schsfirst[sch] + s0;
    if (s >= schslast[sch]) return;

    const int p1 = mapsp1[s];
    const int p2 = mapsp2[s];
    const int z  = mapsz[s];

    const int s4 = mapss4[s];
    const int s04 = s4 - schsfirst[sch];

    dss4[s0] = s04 - s0;
    dss3[s04] = s0 - s04;

    __syncthreads();

    const int s3 = s + dss3[s0];

    // 6a. compute new mesh geometry
    calcZoneCtrs(s, s0, z, p1, px, zx);
    calcSideVols(s, z, p1, p2, px, zx, sarea, svol);
    calcZoneVols(s, s0, z, sarea, svol, zarea, zvol);

    // 7. compute work
    hydroCalcWork(s, s0, s3, z, p1, p2, sfp, sfq, pu0, pu, pxp, dt,
            zw, zetot);

}


static __global__ void gpuMain5()
{
    const int z = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (z >= numz) return;

    const int z0 = threadIdx.x;
    const int zlength = min(CHUNK_SIZE, numz - blockIdx.x * CHUNK_SIZE);

    // 7. compute work
    hydroCalcWorkRate(z, zvol0, zvol, zw, zp, dt, zwrate);

    // 8. update state variables
    hydroCalcEnergy(z, zetot, zm, ze);
    hydroCalcRho(z, zm, zvol, zr);

    // 9.  compute timestep for next cycle
    hydroCalcDt(z, z0, zlength, zdu, zss, zdl, zvol, zvol0, dt,
            dtnext, idtnext);

}


void meshCheckBadSides() {

    int numsbadH;
    cudaMemcpyFromSymbol(&numsbadH, numsbad, sizeof(int));
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
        const double2* pxH,
        const double2* puH,
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
        const int* znumpH) {

    printf("Running Hydro on device...\n");

    computeChunks(numsH, numzH, mapszH, CHUNK_SIZE, numschH,
            schsfirstH, schslastH, schzfirstH, schzlastH);
    numpchH = (numpH+CHUNK_SIZE-1) / CHUNK_SIZE;
    numzchH = (numzH+CHUNK_SIZE-1) / CHUNK_SIZE;

    cudaMemcpyToSymbol(numsch, &numschH, sizeof(int));
    cudaMemcpyToSymbol(nump, &numpH, sizeof(int));
    cudaMemcpyToSymbol(numz, &numzH, sizeof(int));
    cudaMemcpyToSymbol(nums, &numsH, sizeof(int));
    cudaMemcpyToSymbol(numc, &numcH, sizeof(int));
    cudaMemcpyToSymbol(pgamma, &pgammaH, sizeof(double));
    cudaMemcpyToSymbol(pssmin, &pssminH, sizeof(double));
    cudaMemcpyToSymbol(talfa, &talfaH, sizeof(double));
    cudaMemcpyToSymbol(tssmin, &tssminH, sizeof(double));
    cudaMemcpyToSymbol(qgamma, &qgammaH, sizeof(double));
    cudaMemcpyToSymbol(q1, &q1H, sizeof(double));
    cudaMemcpyToSymbol(q2, &q2H, sizeof(double));
    cudaMemcpyToSymbol(hcfl, &hcflH, sizeof(double));
    cudaMemcpyToSymbol(hcflv, &hcflvH, sizeof(double));

    const double2 vfixxH = make_double2(1., 0.);
    const double2 vfixyH = make_double2(0., 1.);
    cudaMemcpyToSymbol(vfixx, &vfixxH, sizeof(double2));
    cudaMemcpyToSymbol(vfixy, &vfixyH, sizeof(double2));
    cudaMemcpyToSymbol(numbcx, &numbcxH, sizeof(int));
    cudaMemcpyToSymbol(numbcy, &numbcyH, sizeof(int));
    cudaMemcpyToSymbol(bcx, bcxH, numbcxH*sizeof(double));
    cudaMemcpyToSymbol(bcy, bcyH, numbcyH*sizeof(double));

    cudaMalloc(&schsfirstD, numschH*sizeof(int));
    cudaMalloc(&schslastD, numschH*sizeof(int));
    cudaMalloc(&schzfirstD, numschH*sizeof(int));
    cudaMalloc(&schzlastD, numschH*sizeof(int));
    cudaMalloc(&mapsp1D, numsH*sizeof(int));
    cudaMalloc(&mapsp2D, numsH*sizeof(int));
    cudaMalloc(&mapszD, numsH*sizeof(int));
    cudaMalloc(&mapss4D, numsH*sizeof(int));
    cudaMalloc(&znumpD, numzH*sizeof(int));

    cudaMalloc(&pxD, numpH*sizeof(double2));
    cudaMalloc(&pxpD, numpH*sizeof(double2));
    cudaMalloc(&px0D, numpH*sizeof(double2));
    cudaMalloc(&zxD, numzH*sizeof(double2));
    cudaMalloc(&zxpD, numzH*sizeof(double2));
    cudaMalloc(&puD, numpH*sizeof(double2));
    cudaMalloc(&pu0D, numpH*sizeof(double2));
    cudaMalloc(&papD, numpH*sizeof(double2));
    cudaMalloc(&ssurfD, numsH*sizeof(double2));
    cudaMalloc(&zmD, numzH*sizeof(double));
    cudaMalloc(&zrD, numzH*sizeof(double));
    cudaMalloc(&zrpD, numzH*sizeof(double));
    cudaMalloc(&sareaD, numsH*sizeof(double));
    cudaMalloc(&svolD, numsH*sizeof(double));
    cudaMalloc(&zareaD, numzH*sizeof(double));
    cudaMalloc(&zvolD, numzH*sizeof(double));
    cudaMalloc(&zvol0D, numzH*sizeof(double));
    cudaMalloc(&zdlD, numzH*sizeof(double));
    cudaMalloc(&zduD, numzH*sizeof(double));
    cudaMalloc(&zeD, numzH*sizeof(double));
    cudaMalloc(&zetot0D, numzH*sizeof(double));
    cudaMalloc(&zetotD, numzH*sizeof(double));
    cudaMalloc(&zwD, numzH*sizeof(double));
    cudaMalloc(&zwrateD, numzH*sizeof(double));
    cudaMalloc(&zpD, numzH*sizeof(double));
    cudaMalloc(&zssD, numzH*sizeof(double));
    cudaMalloc(&smfD, numsH*sizeof(double));
    cudaMalloc(&careapD, numcH*sizeof(double));
    cudaMalloc(&sareapD, numsH*sizeof(double));
    cudaMalloc(&svolpD, numsH*sizeof(double));
    cudaMalloc(&zareapD, numzH*sizeof(double));
    cudaMalloc(&zvolpD, numzH*sizeof(double));
    cudaMalloc(&cmaswtD, numsH*sizeof(double));
    cudaMalloc(&pmaswtD, numpH*sizeof(double));
    cudaMalloc(&sfpD, numsH*sizeof(double2));
    cudaMalloc(&sftD, numsH*sizeof(double2));
    cudaMalloc(&sfqD, numsH*sizeof(double2));
    cudaMalloc(&cftotD, numcH*sizeof(double2));
    cudaMalloc(&pfD, numpH*sizeof(double2));
    cudaMalloc(&cevolD, numcH*sizeof(double));
    cudaMalloc(&cduD, numcH*sizeof(double));
    cudaMalloc(&cdivD, numcH*sizeof(double));
    cudaMalloc(&zucD, numzH*sizeof(double2));
    cudaMalloc(&crmuD, numcH*sizeof(double));
    cudaMalloc(&cqeD, 2*numcH*sizeof(double2));
    cudaMalloc(&ccosD, numcH*sizeof(double));
    cudaMalloc(&cwD, numcH*sizeof(double));

    cudaMalloc(&mapspkeyD, numsH*sizeof(int));
    cudaMalloc(&mapspvalD, numsH*sizeof(int));
    cudaMalloc(&mappsfirstD, numpH*sizeof(int));
    cudaMalloc(&mapssnextD, numsH*sizeof(int));

    cudaMemcpyToSymbol(schsfirst, &schsfirstD, sizeof(void*));
    cudaMemcpyToSymbol(schslast, &schslastD, sizeof(void*));
    cudaMemcpyToSymbol(schzfirst, &schzfirstD, sizeof(void*));
    cudaMemcpyToSymbol(schzlast, &schzlastD, sizeof(void*));
    cudaMemcpyToSymbol(mapsp1, &mapsp1D, sizeof(void*));
    cudaMemcpyToSymbol(mapsp2, &mapsp2D, sizeof(void*));
    cudaMemcpyToSymbol(mapsz, &mapszD, sizeof(void*));
    cudaMemcpyToSymbol(mapss4, &mapss4D, sizeof(void*));
    cudaMemcpyToSymbol(mapspkey, &mapspkeyD, sizeof(void*));
    cudaMemcpyToSymbol(mapspval, &mapspvalD, sizeof(void*));
    cudaMemcpyToSymbol(mappsfirst, &mappsfirstD, sizeof(void*));
    cudaMemcpyToSymbol(mapssnext, &mapssnextD, sizeof(void*));
    cudaMemcpyToSymbol(znump, &znumpD, sizeof(void*));

    cudaMemcpyToSymbol(px, &pxD, sizeof(void*));
    cudaMemcpyToSymbol(pxp, &pxpD, sizeof(void*));
    cudaMemcpyToSymbol(px0, &px0D, sizeof(void*));
    cudaMemcpyToSymbol(zx, &zxD, sizeof(void*));
    cudaMemcpyToSymbol(zxp, &zxpD, sizeof(void*));
    cudaMemcpyToSymbol(pu, &puD, sizeof(void*));
    cudaMemcpyToSymbol(pu0, &pu0D, sizeof(void*));
    cudaMemcpyToSymbol(pap, &papD, sizeof(void*));
    cudaMemcpyToSymbol(ssurf, &ssurfD, sizeof(void*));
    cudaMemcpyToSymbol(zm, &zmD, sizeof(void*));
    cudaMemcpyToSymbol(zr, &zrD, sizeof(void*));
    cudaMemcpyToSymbol(zrp, &zrpD, sizeof(void*));
    cudaMemcpyToSymbol(sarea, &sareaD, sizeof(void*));
    cudaMemcpyToSymbol(svol, &svolD, sizeof(void*));
    cudaMemcpyToSymbol(zarea, &zareaD, sizeof(void*));
    cudaMemcpyToSymbol(zvol, &zvolD, sizeof(void*));
    cudaMemcpyToSymbol(zvol0, &zvol0D, sizeof(void*));
    cudaMemcpyToSymbol(zdl, &zdlD, sizeof(void*));
    cudaMemcpyToSymbol(zdu, &zduD, sizeof(void*));
    cudaMemcpyToSymbol(ze, &zeD, sizeof(void*));
    cudaMemcpyToSymbol(zetot, &zetotD, sizeof(void*));
    cudaMemcpyToSymbol(zw, &zwD, sizeof(void*));
    cudaMemcpyToSymbol(zwrate, &zwrateD, sizeof(void*));
    cudaMemcpyToSymbol(zp, &zpD, sizeof(void*));
    cudaMemcpyToSymbol(zss, &zssD, sizeof(void*));
    cudaMemcpyToSymbol(smf, &smfD, sizeof(void*));
    cudaMemcpyToSymbol(careap, &careapD, sizeof(void*));
    cudaMemcpyToSymbol(sareap, &sareapD, sizeof(void*));
    cudaMemcpyToSymbol(svolp, &svolpD, sizeof(void*));
    cudaMemcpyToSymbol(zareap, &zareapD, sizeof(void*));
    cudaMemcpyToSymbol(zvolp, &zvolpD, sizeof(void*));
    cudaMemcpyToSymbol(cmaswt, &cmaswtD, sizeof(void*));
    cudaMemcpyToSymbol(pmaswt, &pmaswtD, sizeof(void*));
    cudaMemcpyToSymbol(sfp, &sfpD, sizeof(void*));
    cudaMemcpyToSymbol(sft, &sftD, sizeof(void*));
    cudaMemcpyToSymbol(sfq, &sfqD, sizeof(void*));
    cudaMemcpyToSymbol(cftot, &cftotD, sizeof(void*));
    cudaMemcpyToSymbol(pf, &pfD, sizeof(void*));
    cudaMemcpyToSymbol(cevol, &cevolD, sizeof(void*));
    cudaMemcpyToSymbol(cdu, &cduD, sizeof(void*));
    cudaMemcpyToSymbol(cdiv, &cdivD, sizeof(void*));
    cudaMemcpyToSymbol(zuc, &zucD, sizeof(void*));
    cudaMemcpyToSymbol(crmu, &crmuD, sizeof(void*));
    cudaMemcpyToSymbol(cqe, &cqeD, sizeof(void*));
    cudaMemcpyToSymbol(ccos, &ccosD, sizeof(void*));
    cudaMemcpyToSymbol(cw, &cwD, sizeof(void*));

    cudaMemcpy(schsfirstD, schsfirstH, numschH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(schslastD, schslastH, numschH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(schzfirstD, schzfirstH, numschH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(schzlastD, schzlastH, numschH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mapsp1D, mapsp1H, numsH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mapsp2D, mapsp2H, numsH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mapszD, mapszH, numsH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mapss4D, mapss4H, numsH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(znumpD, znumpH, numzH*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(zmD, zmH, numzH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(smfD, smfH, numsH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pxD, pxH, numpH*sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(puD, puH, numpH*sizeof(double2), cudaMemcpyHostToDevice);
    cudaMemcpy(zrD, zrH, numzH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(zvolD, zvolH, numzH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(zeD, zeH, numzH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(zetotD, zetotH, numzH*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(zwrateD, zwrateH, numzH*sizeof(double), cudaMemcpyHostToDevice);

    /*thrust::device_ptr<int> mapsp1T(mapsp1D);
    thrust::device_ptr<int> mapspkeyT(mapspkeyD);
    thrust::device_ptr<int> mapspvalT(mapspvalD);

    thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
    thrust::sequence(mapspvalT, mapspvalT + numsH);
    thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);
*/
    int gridSize = (numsH+CHUNK_SIZE-1) / CHUNK_SIZE;
    int chunkSize = CHUNK_SIZE;
    gpuInvMap<<<gridSize, chunkSize>>>(mapspkeyD, mapspvalD,
            mappsfirstD, mapssnextD);
    cudaDeviceSynchronize();

    int zero = 0;
    cudaMemcpyToSymbol(numsbad, &zero, sizeof(int));

}


void hydroDoCycle(
        const double dtH,
        double& dtnextH,
        int& idtnextH) {
    int gridSizeS, gridSizeP, gridSizeZ, chunkSize;

    cudaMemcpyToSymbol(dt, &dtH, sizeof(double));

    gridSizeS = numschH;
    gridSizeP = numpchH;
    gridSizeZ = numzchH;
    chunkSize = CHUNK_SIZE;

    gpuMain1<<<gridSizeP, chunkSize>>>();
    cudaDeviceSynchronize();

    gpuMain2<<<gridSizeS, chunkSize>>>();
    cudaDeviceSynchronize();
    meshCheckBadSides();

    gpuMain3<<<gridSizeP, chunkSize>>>();
    cudaDeviceSynchronize();

    double bigval = 1.e99;
    cudaMemcpyToSymbol(dtnext, &bigval, sizeof(double));

    gpuMain4<<<gridSizeS, chunkSize>>>();
    cudaDeviceSynchronize();

    gpuMain5<<<gridSizeZ, chunkSize>>>();
    cudaDeviceSynchronize();
    meshCheckBadSides();

    cudaMemcpyFromSymbol(&dtnextH, dtnext, sizeof(double));
    cudaMemcpyFromSymbol(&idtnextH, idtnext, sizeof(int));

}


void hydroGetData(
        const int numpH,
        const int numzH,
        double2* pxH,
        double* zrH,
        double* zeH,
        double* zpH) {

    cudaMemcpy(pxH, pxD, numpH*sizeof(double2), cudaMemcpyDeviceToHost);
    cudaMemcpy(zrH, zrD, numzH*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zeH, zeD, numzH*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zpH, zpD, numzH*sizeof(double), cudaMemcpyDeviceToHost);

}


void hydroInitGPU()
{
    int one = 1;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaMemcpyToSymbol(gpuinit, &one, sizeof(int));

}


void hydroFinalGPU()
{
}

