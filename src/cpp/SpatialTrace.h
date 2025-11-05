/*************************************************************************************

    Grid physics lileftry, www.github.com/paboyle/Grid

    Source file: MomentumProject.h

    Copyright (C) 2025

Author: Peter Boyle <pboyle@bnl.gov>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
#pragma once
#include <nvtx3/nvToolsExt.h>

NAMESPACE_BEGIN(Grid);
/*
   MultiMomProject

   Import vectors -> nxyz x (ncomponent x nt)
   Import complex phases -> nmom x nxy

   apply = via (possibly batched) GEMM
*/
template <class LeftField, class RightField, class ResultField>
class SpatialTrace {
public:
  typedef typename RightField::scalar_type scalar;
  // typedef typename RightField::scalar_type right_scalar;
  // typedef typename LeftField::scalar_type left_scalar;
  // typedef typename ResultField::scalar_type result_scalar;
  typedef typename RightField::scalar_object right_scalar_object;
  typedef typename LeftField::scalar_object left_scalar_object;
  typedef typename ResultField::scalar_object result_scalar_object;

  GridBase *grid;
  uint64_t nxyz;
  uint64_t nt;
  uint64_t nright;
  uint64_t nleft;
  uint64_t rightWords;
  uint64_t leftWords;
  uint64_t resultWords;

  deviceVector<scalar> BLAS_L;
  deviceVector<scalar> BLAS_R;
  deviceVector<scalar> BLAS_T;
  deviceVector<Integer> OMAP;
  deviceVector<Integer> IMAP;

  SpacialTrace() {};
  ~SpacialTrace() { Deallocate(); };

  void Deallocate(void) {
    grid = nullptr;
    nxyz = 0;
    lsites = 0;
    nt = 0;
    nright = 0;
    nleft = 0;
    rightWords = 0;
    leftWords = 0;
    resultWords = 0;
    BLAS_L.resize(0);
    BLAS_R.resize(0);
    BLAS_T.resize(0);
    IMAP.resize(0);
    OMAP.resize(0);
  }

  void Allocate(int _nleft, int _nright, GridBase *_grid) {
    grid = _grid;
    Coordinate ldims = grid->LocalDimensions();

    nt = ldims[grid->Nd() - 1];
    lsites = grid->lSites();
    nxyz = grid->lSites() / nt;
    nleft = _nleft;
    nright = _nright;
    leftWords = sizeof(left_scalar_object) / sizeof(scalar);
    rightWords = sizeof(right_scalar_object) / sizeof(scalar);
    resultWords = sizeof(result_scalar_object) / sizeof(scalar);

    GRID_ASSERT(nleft * leftWords * nright * rightWords == resultWords);

    BLAS_L.resize(nxyz * nt * nleft * leftWords);
    BLAS_R.resize(nxyz * nt * nright * rightWords);
    BLAS_T.resize(nt * resultWords);
  }

  void ImportLeft(const std::vector<LeftField> &left,
                  bool do_conjugate = true) {
    //    might as well just make the momenta here
    typedef typename LeftField::vector_object vobj;

    int nd = grid->_ndimension;

    uint64_t sz = BLAS_L.size();

    GRID_ASSERT(left[0].Grid() == grid);
    GRID_ASSERT(nleft >= left.size());
    GRID_ASSERT(sz >= nxyz * nt * left.size(nleft) * leftWords);

    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims = grid->LocalDimensions();
    int64_t lsites = grid->lSites();
    int64_t nwords = leftWords * nleft;
    Coordinate simd = grid->_simd_layout;
    const int Nsimd = vobj::Nsimd();
    auto blasData_l = &BLAS_L[0];

    for (int li = 0; li < left.size(); li++) {

      autoView(left_v, left[li], AcceleratorRead);
      auto left_p = &left_v[0];

      accelerator_for(ls, lsites, 1, {
        //////////////////////////////////////////
        // isite -- map lane within buffer to lane within lattice
        ////////////////////////////////////////////
        Coordinate lcoor(nd, 0);
        Lexicographic::CoorFromIndex(lcoor, ls, ldims);

        Coordinate icoor(nd);
        Coordinate ocoor(nd);
        for (int d = 0; d < nd; d++) {
          icoor[d] = lcoor[d] / rdimensions[d];
          ocoor[d] = lcoor[d] % rdimensions[d];
        }

        uint64_t l_t = lcoor[nd - 1];

        int64_t l_xyz = 0;
        Coordinate xyz_coor = lcoor;
        xyz_coor[nd - 1] = 0;
        Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

        int64_t osite;
        int64_t isite;
        Lexicographic::IndexFromCoor(ocoor, osite, rdimensions);
        Lexicographic::IndexFromCoor(icoor, isite, simd);

        // BLAS_L[nmom][slice_vol]
        // Fortran Column major BLAS layout is L_xyz,(t,w)
        scalar data = extractLane(isite, left_p[osite]);
        scalar *data_words = (scalar *)&data;
        for (int w = 0; w < nwords; w++) {
          // BLAS_R[slice_vol][nt][words]
          // Fortran Column major BLAS layout is V_(t,w)_xyz
          uint64_t idx = w + l_t * nwords + l_xyz * nwords * Nt;
          blasData_l[idx] = data;
          blasData_p[idx] = data_words[w];
        }
      });
    }
  }

  void ImportRight(RightField &vec) {
    typedef typename RightField::vector_object vobj;

    int nd = grid->_ndimension;

    uint64_t sz = BLAS_R.size();

    GRID_ASSERT(sz = nxyz * words * nt);

    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims = grid->LocalDimensions();
    int64_t osites = grid->oSites();
    Coordinate simd = grid->_simd_layout;
    const int Nsimd = vobj::Nsimd();

    auto blasData_p = &BLAS_R[0];
    autoView(Data, vec, AcceleratorRead);
    auto Data_p = &Data[0];

    int64_t nwords = words; // for capture
    int64_t Nt = nt;        // for capture

    accelerator_for(sf, osites, Nsimd, {
#ifdef GRID_SIMT
      {
        int lane = acceleratorSIMTlane(Nsimd); // buffer lane
#else
	  for(int lane=0;lane<Nsimd;lane++) {
#endif
        //////////////////////////////////////////
        // isite -- map lane within buffer to lane within lattice
        ////////////////////////////////////////////
        Coordinate lcoor(nd, 0);
        Coordinate icoor(nd);
        Coordinate ocoor(nd);

        Lexicographic::CoorFromIndex(icoor, lane, simd);
        Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);

        int64_t l_xyz = 0;
        for (int d = 0; d < nd; d++) {
          lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];
        }
        uint64_t l_t = lcoor[nd - 1];

        Coordinate xyz_coor = lcoor;
        xyz_coor[nd - 1] = 0;
        Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

        scalar_object data = extractLane(lane, Data[sf]);
        scalar *data_words = (scalar *)&data;
        for (int w = 0; w < nwords; w++) {
          // BLAS_R[slice_vol][nt][words]
          // Fortran Column major BLAS layout is V_(t,w)_xyz
          uint64_t idx = w + l_t * nwords + l_xyz * nwords * Nt;
          blasData_p[idx] = data_words[w];
        }
      }
    });
  }

  scalar *getBlasRightPointer() {
    assert(BLAS_R.size() > 0);
    return (scalar *)&BLAS_R[0];
  }

  scalar *getBlasLeftPointer() {
    assert(BLAS_L.size() > 0);
    return (scalar *)&BLAS_L[0];
  }

  Integer *getImapPointer() {
    if (IMAP.size() == 0) {
      buildIndexMap(true);
    }
    return (Integer *)&IMAP[0];
  }

  Integer *getOmapPointer() {
    if (OMAP.size() == 0) {
      buildIndexMap(false);
    }
    return (Integer *)&OMAP[0];
  }

  void buildIndexMap(bool inner) {
    /* Builds mapping from BLAS_R indices (in blocks of `words`) to inner (if
     * `inner` == true) or outer (if `inner` == false) Grid indices. Mapping is
     * stored in `IMAP` and `OMAP`, respectively.
     */
    int nd = grid->_ndimension;

    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims = grid->LocalDimensions();
    int64_t lsites = grid->lSites();
    Coordinate simd = grid->_simd_layout;
    const int Nsimd = grid->Nsimd();

    Integer *map_p;
    if (inner) {
      IMAP.resize(lsites);
      map_p = &IMAP[0];
    } else {
      OMAP.resize(lsites);
      map_p = &OMAP[0];
    }

    int64_t Nt = nt;        // for capture
    bool inner_map = inner; // for capture

    accelerator_for(ls, lsites, 1, {
      int lane = ls % Nsimd;
      int sf = ls / Nsimd;
      //////////////////////////////////////////
      // isite -- map lane within buffer to lane within lattice
      ////////////////////////////////////////////
      Coordinate lcoor(nd, 0);
      Coordinate icoor(nd);
      Coordinate ocoor(nd);

      Lexicographic::CoorFromIndex(icoor, lane, simd);
      Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);

      int64_t l_xyz = 0;
      for (int d = 0; d < nd; d++) {
        lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];
      }
      uint64_t l_t = lcoor[nd - 1];

      Coordinate xyz_coor = lcoor;
      xyz_coor[nd - 1] = 0;
      Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

      uint64_t idx = l_t + l_xyz * Nt;
      if (inner_map) {
        map_p[idx] = lane;
      } else {
        map_p[idx] = sf;
      }
    });
  }

  void ExportTrace(std::vector<typename ResultField::scalar_object> &trace) {
    trace.resize(resultWords * nt);
    acceleratorCopyFromDevice(&BLAS_T[0], (scalar *)&trace[0],
                              BLAS_T.size() * sizeof(scalar));
    // Could decide on a layout late?
  }

  // Row major layout "C" order:
  // BLAS_R[slice_vol][nt][words]
  // BLAS_L[nmom][slice_vol]
  // BLAS_T[nmom][nt][words]
  //
  // Fortran Column major BLAS layout is V_(w,t)_xyz
  // Fortran Column major BLAS layout is M_xyz,mom
  // Fortran Column major BLAS layout is P_(w,t),mom
  //
  // Projected
  //
  // P = (V * M)_(w,t),mom
  //

  void Project(std::vector<typename ResultField::scalar_object> &trace_gdata) {
    double t_import = 0;
    double t_export = 0;
    double t_gemm = 0;
    double t_allreduce = 0;
    t_import -= usecond();

    std::vector<typename ResultField::scalar_object> projected_planes;

    deviceVector<scalar *> Rd(1);
    deviceVector<scalar *> Ld(1);
    deviceVector<scalar *> Td(1);

    scalar *Rh = &BLAS_R[0];
    scalar *Lh = &BLAS_L[0];
    scalar *Th = &BLAS_T[0];

    acceleratorPut(Rd[0], Rh);
    acceleratorPut(Ld[0], Lh);
    acceleratorPut(Td[0], Th);
    t_import += usecond();

    GridBLAS BLAS;

    /////////////////////////////////////////
    // T_im = Lmx . Rxi
    /////////////////////////////////////////
    t_gemm -= usecond();
    BLAS.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N, resultWords * nt,
                     nleft * leftWords, lsites, scalar(1.0), Vd, Md,
                     scalar(0.0), // wipe out result
                     Pd);
    BLAS.synchronise();
    t_gemm += usecond();

    t_export -= usecond();
    ExportMomentumProjection(projected_planes); // resizes
    t_export += usecond();

    /////////////////////////////////
    // Reduce across MPI ranks
    /////////////////////////////////
    int nd = grid->Nd();
    int gt = grid->GlobalDimensions()[nd - 1];
    int lt = grid->LocalDimensions()[nd - 1];
    trace_gdata.resize(gt * nmom);
    for (int t = 0; t < gt * nmom;
         t++) { // global Nt array with zeroes for stuff not on this node
      trace_gdata[t] = Zero();
    }
    for (int t = 0; t < lt; t++) {
      for (int m = 0; m < nmom; m++) {
        int st = grid->LocalStarts()[nd - 1];
        trace_gdata[t + st + gt * m] = projected_planes[t + lt * m];
      }
    }
    t_allreduce -= usecond();
    grid->GlobalSumVector((scalar *)&projected_gdata[0], gt * nmom * words);
    t_allreduce += usecond();

    std::cout << GridLogPerformance << " MomentumProject t_import  " << t_import
              << "us" << std::endl;
    std::cout << GridLogPerformance << " MomentumProject t_export  " << t_export
              << "us" << std::endl;
    std::cout << GridLogPerformance << " MomentumProject t_gemm    " << t_gemm
              << "us" << std::endl;
    std::cout << GridLogPerformance << " MomentumProject t_reduce  "
              << t_allreduce << "us" << std::endl;
  }
  void
  Project(LeftField &data,
          std::vector<typename LeftField::scalar_object> &projected_gdata) {
    this->ImportVector(data);
    Project(projected_gdata);
  }
};

NAMESPACE_END(Grid);
