/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: SpatialTrace.h

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
   SpatialTrace

   Import left fields  -> nt x nxyz x (nleft x leftWords)
   Import right fields -> nt x nxyz x (nright x rightWords)

   Compute spatial trace via batched GEMM, one batch per time slice:
   For each time t: Trace[t] = sum_xyz Left[t,xyz]^dag * Right[t,xyz]
*/
template <class LeftField, class RightField, typename result_object>
class SpatialTrace {
public:
  typedef typename RightField::scalar_type scalar;
  typedef typename RightField::scalar_object right_scalar_object;
  typedef typename LeftField::scalar_object left_scalar_object;

  GridBase *grid;
  uint64_t nxyz;
  uint64_t nt;
  uint64_t lsites;
  uint64_t nright;
  uint64_t nleft;
  uint64_t nresults;
  uint64_t rightWords;
  uint64_t leftWords;
  uint64_t resultWords;

  deviceVector<scalar> BLAS_L;
  deviceVector<scalar> BLAS_R;
  deviceVector<scalar> BLAS_T;
  deviceVector<Integer> OMAP;
  deviceVector<Integer> IMAP;

  SpatialTrace() {};
  ~SpatialTrace() { Deallocate(); };

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
    nresults = 0;
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
    resultWords = sizeof(result_object) / sizeof(scalar);
    nresults = nleft * leftWords * nright * rightWords / resultWords;

    GRID_ASSERT(nleft * leftWords * nright * rightWords ==
                nresults * resultWords);
    // Layout: BLAS_L[nt][nxyz][nleft * leftWords]
    BLAS_L.resize(nt * nxyz * nleft * leftWords);
    // Layout: BLAS_R[nt][nxyz][nright * rightWords]
    BLAS_R.resize(nt * nxyz * nright * rightWords);
    // Layout: BLAS_T[nt][resultWords]
    BLAS_T.resize(nt * nresults * resultWords);
  }

  void ImportLeft(const std::vector<LeftField> &left) {
    typedef typename LeftField::vector_object vobj;

    int nd = grid->_ndimension;

    uint64_t sz = BLAS_L.size();

    GRID_ASSERT(left[0].Grid() == grid);
    GRID_ASSERT(nleft >= left.size());
    GRID_ASSERT(sz >= nt * nxyz * left.size() * leftWords);

    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims = grid->LocalDimensions();
    Coordinate simd = grid->_simd_layout;
    const int Nsimd = vobj::Nsimd();
    auto blasData_l = &BLAS_L[0];

    int64_t Nt = nt;               // for capture
    int64_t Nxyz = nxyz;           // for capture
    int64_t Nleft = nleft;         // for capture
    int64_t LeftWords = leftWords; // for capture

    for (int li = 0; li < left.size(); li++) {

      autoView(left_v, left[li], AcceleratorRead);
      auto left_p = &left_v[0];

      accelerator_for(sf, grid->oSites(), Nsimd, {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd); // buffer lane
#else
	  for(int lane=0;lane<Nsimd;lane++) {
#endif
          //////////////////////////////////////////
          // Map lane within buffer to lane within lattice
          ////////////////////////////////////////////
          Coordinate lcoor(nd, 0);
          Coordinate icoor(nd);
          Coordinate ocoor(nd);

          Lexicographic::CoorFromIndex(icoor, lane, simd);
          Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);

          for (int d = 0; d < nd; d++) {
            lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];
          }
          uint64_t l_t = lcoor[nd - 1];

          Coordinate xyz_coor = lcoor;
          xyz_coor[nd - 1] = 0;
          int64_t l_xyz = 0;
          Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

          left_scalar_object data = extractLane(lane, left_p[sf]);
          scalar *data_words = (scalar *)&data;

          // Layout: BLAS_L[t][xyz][li * leftWords + w]
          // Index: w + li*leftWords + xyz*nleft*leftWords +
          // t*nxyz*nleft*leftWords
          for (int w = 0; w < LeftWords; w++) {
            uint64_t idx = w + li * LeftWords + l_xyz * Nleft * LeftWords +
                           l_t * Nxyz * Nleft * LeftWords;
            blasData_l[idx] = data_words[w];
          }
        }
      });
    }
  }

  void ImportRight(const std::vector<RightField> &right) {
    typedef typename RightField::vector_object vobj;

    int nd = grid->_ndimension;

    uint64_t sz = BLAS_R.size();

    GRID_ASSERT(right[0].Grid() == grid);
    GRID_ASSERT(nright >= right.size());
    GRID_ASSERT(sz >= nt * nxyz * right.size() * rightWords);

    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims = grid->LocalDimensions();
    Coordinate simd = grid->_simd_layout;
    const int Nsimd = vobj::Nsimd();

    auto blasData_r = &BLAS_R[0];

    int64_t Nt = nt;                 // for capture
    int64_t Nxyz = nxyz;             // for capture
    int64_t Nright = nright;         // for capture
    int64_t RightWords = rightWords; // for capture

    for (int ri = 0; ri < right.size(); ri++) {

      autoView(right_v, right[ri], AcceleratorRead);
      auto right_p = &right_v[0];

      accelerator_for(sf, grid->oSites(), Nsimd, {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd); // buffer lane
#else
	  for(int lane=0;lane<Nsimd;lane++) {
#endif
          //////////////////////////////////////////
          // Map lane within buffer to lane within lattice
          ////////////////////////////////////////////
          Coordinate lcoor(nd, 0);
          Coordinate icoor(nd);
          Coordinate ocoor(nd);

          Lexicographic::CoorFromIndex(icoor, lane, simd);
          Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);

          for (int d = 0; d < nd; d++) {
            lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];
          }
          uint64_t l_t = lcoor[nd - 1];

          Coordinate xyz_coor = lcoor;
          xyz_coor[nd - 1] = 0;
          int64_t l_xyz = 0;
          Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

          right_scalar_object data = extractLane(lane, right_p[sf]);
          scalar *data_words = (scalar *)&data;

          // Layout: BLAS_R[t][xyz][ri * rightWords + w]
          // Index: w + ri*rightWords + xyz*nright*rightWords +
          // t*nxyz*nright*rightWords
          for (int w = 0; w < RightWords; w++) {
            uint64_t idx = w + ri * RightWords + l_xyz * Nright * RightWords +
                           l_t * Nxyz * Nright * RightWords;
            blasData_r[idx] = data_words[w];
          }
        }
      });
    }
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
    /* Builds mapping from BLAS indices (in blocks of `words`) to inner (if
     * `inner` == true) or outer (if `inner` == false) Grid indices. Mapping is
     * stored in `IMAP` and `OMAP`, respectively.
     */
    int nd = grid->_ndimension;

    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims = grid->LocalDimensions();
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

      uint64_t idx = l_t * nxyz + l_xyz;
      if (inner_map) {
        map_p[idx] = lane;
      } else {
        map_p[idx] = sf;
      }
    });
  }

  void ExportTrace(std::vector<result_object> &trace) {
    trace.resize(nt * nresults);

    // Copy directly to trace (will be in wrong order initially)
    acceleratorCopyFromDevice(&BLAS_T[0], (scalar *)&trace[0],
                              BLAS_T.size() * sizeof(scalar));

    std::vector<scalar> temp_result(nresults * resultWords);
    // Now transpose each result in-place
    for (int t = 0; t < nt; t++) {
      scalar *data = (scalar *)&trace[t * nresults];

      int M = nleft * leftWords;
      int N = nright * rightWords;

      // Transpose M×N matrix in-place
      // For small matrices, use a temporary
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          // Column-major input at j*M + i becomes row-major at i*N + j
          temp_result[i * N + j] = data[j * M + i];
        }
      }
      // Copy back
      for (int k = 0; k < nresults * resultWords; k++) {
        data[k] = temp_result[k];
      }
    }
  }

  void Trace(std::vector<result_object> &trace_gdata) {
    double t_import = 0;
    double t_export = 0;
    double t_gemm = 0;
    double t_allreduce = 0;
    nvtxRangePushA("Import");
    t_import -= usecond();

    std::vector<result_object> trace_planes;

    // Setup batched GEMM pointers - one batch per time slice
    deviceVector<scalar *> Ld(nt);
    deviceVector<scalar *> Rd(nt);
    deviceVector<scalar *> Td(nt);

    scalar *Lh = &BLAS_L[0];
    scalar *Rh = &BLAS_R[0];
    scalar *Th = &BLAS_T[0];

    // Each batch points to a different time slice
    for (int t = 0; t < nt; t++) {
      acceleratorPut(Ld[t], Lh + t * nxyz * nleft * leftWords);
      acceleratorPut(Rd[t], Rh + t * nxyz * nright * rightWords);
      acceleratorPut(Td[t], Th + t * nresults * resultWords);
    }
    t_import += usecond();
    nvtxRangePop();

    GridBLAS BLAS;

    /////////////////////////////////////////
    // For each time t: T[t] = L[t] * R[t]
    // Sum over spatial xyz dimension
    /////////////////////////////////////////
    nvtxRangePushA("GEMM");
    t_gemm -= usecond();
    BLAS.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N,
                     nleft * leftWords,   // M (rows of L)
                     nright * rightWords, // N (cols of R)
                     nxyz,                // K (sum over spatial)
                     scalar(1.0), Ld, Rd,
                     scalar(0.0), // don't accumulate result
                     Td);
    BLAS.synchronise();
    t_gemm += usecond();
    nvtxRangePop();

    nvtxRangePushA("Export Trace");
    t_export -= usecond();
    ExportTrace(trace_planes);
    t_export += usecond();
    nvtxRangePop();

    /////////////////////////////////
    // Reduce across MPI ranks
    /////////////////////////////////
    int nd = grid->Nd();
    int gt = grid->GlobalDimensions()[nd - 1];
    int lt = grid->LocalDimensions()[nd - 1];
    trace_gdata.resize(gt * nresults);

    // Initialize with zeros
    for (int t = 0; t < gt * nresults; t++) {
      trace_gdata[t] = Zero();
    }

    // Fill in local time slices
    for (int t = 0; t < lt; t++) {
      int st = grid->LocalStarts()[nd - 1];
      for (int r = 0; r < nresults; r++) {
        trace_gdata[(t + st) * nresults + r] = trace_planes[t * nresults + r];
      }
    }

    nvtxRangePushA("Global Sum");
    t_allreduce -= usecond();
    grid->GlobalSumVector((scalar *)&trace_gdata[0],
                          gt * nresults * resultWords);
    t_allreduce += usecond();
    nvtxRangePop();

    std::cout << GridLogPerformance << " SpatialTrace t_import  " << t_import
              << "us" << std::endl;
    std::cout << GridLogPerformance << " SpatialTrace t_export  " << t_export
              << "us" << std::endl;
    std::cout << GridLogPerformance << " SpatialTrace t_gemm    " << t_gemm
              << "us" << std::endl;
    std::cout << GridLogPerformance << " SpatialTrace t_reduce  " << t_allreduce
              << "us" << std::endl;
  }

  void Trace(const std::vector<LeftField> &left,
             const std::vector<RightField> &right,
             std::vector<result_object> &trace_gdata) {
    this->ImportLeft(left);
    this->ImportRight(right);
    Trace(trace_gdata);
  }
};

NAMESPACE_END(Grid);
