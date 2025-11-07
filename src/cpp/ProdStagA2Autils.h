#pragma once

#include <A2AView.h>
#include <Grid/Grid_Eigen_Tensor.h>
#include <StagGamma.h>
#include <nvtx3/nvToolsExt.h>

#ifndef MF_SUM_ARRAY_MAX
#define MF_SUM_ARRAY_MAX 16
#endif

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

#define A2A_TYPEDEFS                                                           \
  typedef typename FImpl::SiteSpinor vobj;                                     \
  typedef typename FImpl::ComplexField ComplexField;                           \
  typedef typename FImpl::FermionField FermionField;                           \
  typedef typename ComplexField::vector_object cobj;                           \
  typedef typename FImpl::ImplParams FImplParams;                              \
  typedef CartesianStencil<vColourMatrix, vColourMatrix, FImplParams>          \
      GaugeStencil;                                                            \
  typedef CartesianStencilView<vColourMatrix, vColourMatrix, FImplParams>      \
      GaugeStencilView;                                                        \
  typedef typename vobj::scalar_type scalar_type;                              \
  typedef typename vobj::vector_type vector_type;                              \
  typedef iSinglet<scalar_type> Scalar_s;                                      \
  typedef iSinglet<vector_type> Scalar_v;                                      \
  typedef decltype(coalescedRead(Scalar_v())) calcScalar;                      \
  typedef decltype(coalescedRead(vobj())) calcSpinor;                          \
  typedef decltype(coalescedRead(vColourMatrix())) calcColourMatrix;           \
  typedef typename FImpl::StencilImpl FermStencil;                             \
  typedef typename FImpl::StencilView FermStencilView;                         \
  typedef LatticeView<vColourMatrix> GaugeView;                                \
  typedef LatticeView<cobj> ComplexView;                                       \
  typedef LatticeView<vobj> FermView;                                          \
  typedef typename A2ATaskBase<FImpl>::ContractType ContractType;              \
  typedef std::function<void(scalar_type *, cobj *)> SimdFunc;                 \
  typedef std::function<void(cobj *, int, int)> VectorFunc;

#define COMMON_VARS                                                            \
  int sizeL = this->_left_view->size();                                        \
  int sizeR = this->_right_view->size();                                       \
                                                                               \
  int orthogDir = this->_orthog_dir;                                           \
  const int simdSize = this->_grid->Nsimd();                                   \
  const int reducedOrthogDimSize = this->_grid->_rdimensions[orthogDir];       \
                                                                               \
  const int nBlocks = this->_grid->_slice_nblock[orthogDir];                   \
                                                                               \
  const int localSpatialVolume = this->_grid->_ostride[orthogDir];             \
                                                                               \
  FermView *viewL_p = this->_left_view->getView();                             \
  FermView *viewR_p = this->_right_view->getView();                            \
  int localOrthogDimSize = this->_grid->_ldimensions[orthogDir];               \
                                                                               \
  int Nt = this->_grid->GlobalDimensions()[orthogDir];                         \
                                                                               \
  int pd = this->_grid->_processors[orthogDir];                                \
  int pc = this->_grid->_processor_coor[orthogDir];                            \
                                                                               \
  int orthogSimdSize = this->_grid->_simd_layout[orthogDir];                   \
  Coordinate *icoor_p = this->_i_coor_container_device;                        \
  Integer *ocoor_p = this->_o_coor_map_device;                                 \
  bool cbEven = this->_cb_left == Even;                                        \
  bool oddShifts = this->_odd_shifts;

template <typename FImpl> class A2ATaskBase {
public:
  GRID_SERIALIZABLE_ENUM(ContractType, undef, Full, 0, RightHalf, 1, LeftHalf,
                         2, BothHalf, 3);

  A2A_TYPEDEFS;

protected:
  std::shared_ptr<A2AFieldView<vobj>> _left_view, _right_view;

  std::vector<Coordinate> _i_coor_container;
  Coordinate *_i_coor_container_device;

  std::vector<Integer> _o_coor_map;
  Integer *_o_coor_map_device;

  GridBase *_grid, *_full_grid;

  ContractType _contract_type;

  bool _odd_shifts;
  int _orthog_dir;
  const int _cb_left;

public:
  A2ATaskBase(GridBase *grid, int orthogDir, int cb = Even)
      : _grid(grid), _full_grid(grid), _cb_left(cb), _orthog_dir(orthogDir),
        _odd_shifts(false), _contract_type(ContractType::undef) {

    _i_coor_container.resize(grid->Nsimd(), Coordinate(grid->_ndimension));
    for (int p = 0; p < grid->Nsimd(); p++) {
      grid->iCoorFromIindex(_i_coor_container[p], p);
    }

    size_t size = _i_coor_container.size() * sizeof(Coordinate);
    _i_coor_container_device = (Coordinate *)acceleratorAllocDevice(size);
    acceleratorCopyToDevice(_i_coor_container.data(), _i_coor_container_device,
                            size);
  }

  virtual ~A2ATaskBase() {
    acceleratorFreeDevice(_i_coor_container_device);
    if (_o_coor_map.size() > 0)
      acceleratorFreeDevice(_o_coor_map_device);
  }

  virtual double getFlops() = 0;

  void generateCoorMap() {
    /* Populates `_o_coor_map` property with full grid indices */

    assert(_grid->CheckerBoarded(_orthog_dir) != 1);

    if (_o_coor_map.size() == 0) {
      _o_coor_map.resize(_grid->oSites(), 0);
      _o_coor_map_device = (Integer *)acceleratorAllocDevice(
          _o_coor_map.size() * sizeof(Integer));

      int nBlocks = _grid->_slice_nblock[_orthog_dir];
      int vecsPerSlicePerBlock = _grid->_slice_block[_orthog_dir];
      int blockStride = _grid->_slice_stride[_orthog_dir];
      int rtStride = _grid->_ostride[_orthog_dir];
      int cb = _cb_left;

      thread_for(ss, _full_grid->oSites(), {
        int cbos;
        Coordinate coor;

        _full_grid->oCoorFromOindex(coor, ss);
        cbos = _grid->CheckerBoard(coor);

        if (cbos == cb) {
          int ssh = _grid->oIndex(coor);
          _o_coor_map[ssh] = ss;
        }
      });
      acceleratorCopyToDevice(_o_coor_map.data(), _o_coor_map_device,
                              _o_coor_map.size() * sizeof(Integer));
    }
  }

  virtual void setLeft(const FermionField *left, int size) {
    /*
     * Updates object to compute meson field with new `left` vectors
     * Updated properties:
     * `_contract_type` - Updated to reflect `left` checkerboard state
     * `_left_view`     - Allocates views from `left` pointer parameter
     * `_grid`          - Sets to checkerboarded grid if `_left_view`
     *                    or `_right_view` are checkerboarded, otherwise set to
     *                    full grid
     */

    bool checkerL = left[0].Grid()->_isCheckerBoarded;

    // Toggle LeftHalf bit
    if (_contract_type == ContractType::undef) {
      _contract_type = checkerL ? ContractType::LeftHalf : ContractType::Full;
    } else {
      if (checkerL)
        _contract_type = _contract_type | ContractType::LeftHalf;
      else
        _contract_type = _contract_type & ContractType::RightHalf;
    }

    switch (_contract_type) {
    case ContractType::LeftHalf:
    case ContractType::BothHalf:
    case ContractType::Full:
      _grid = left[0].Grid();
    default:
      break;
    }

    if (checkerL)
      generateCoorMap();

    _left_view = std::make_shared<A2AFieldView<vobj>>();
    _left_view->openViews(left, size);
  }

  // Updates object to compute meson field with new `right` vectors.
  // See corresponding `setLeft` method.
  virtual void setRight(const FermionField *right, int size) {

    bool checkerR = right[0].Grid()->_isCheckerBoarded;

    // Toggle RightHalf bit
    if (_contract_type == ContractType::undef) {
      _contract_type = checkerR ? ContractType::RightHalf : ContractType::Full;
    } else {
      if (checkerR)
        _contract_type = _contract_type | ContractType::RightHalf;
      else
        _contract_type = _contract_type & ContractType::LeftHalf;
    }

    switch (_contract_type) {
    case ContractType::RightHalf:
    case ContractType::BothHalf:
    case ContractType::Full:
      _grid = right[0].Grid();
    default:
      break;
    }

    if (checkerR)
      generateCoorMap();

    _right_view = std::make_shared<A2AFieldView<vobj>>();
    _right_view->openViews(right, size);
  }

  // Updates object to compute meson field with `other._left_view` vectors.
  // Should only be used for mixed cb/full calculation.
  virtual void setLeft(A2ATaskBase<FImpl> &other) {
    assert(!(other.getType() & ContractType::LeftHalf));
    _left_view = other.getLeftView();

    _contract_type = other.getType();
  }

  virtual void setRight(A2ATaskBase<FImpl> &other) {
    assert(!(other.getType() & ContractType::RightHalf));
    _right_view = other.getRightView();

    _contract_type = other.getType();
  }

  std::shared_ptr<A2AFieldView<vobj>> getLeftView() { return _left_view; }
  std::shared_ptr<A2AFieldView<vobj>> getRightView() { return _right_view; }
  ContractType getType() { return _contract_type; }

  virtual int getNgamma() = 0;
  virtual void vectorSumHalf(cobj *a, int b, int c) = 0;
  virtual void vectorSumFull(cobj *a, int b, int c) = 0;
  virtual void vectorSumMixed(cobj *a, int b, int c) = 0;

  virtual void execute(scalar_type *result_p) {

    COMMON_VARS;

    VectorFunc vectorSum;
    SimdFunc simdSum;

    int multFact;

    switch (this->_contract_type) {
    case ContractType::BothHalf:
      vectorSum = [this](cobj *a, int b, int c) {
        this->vectorSumHalf(a, b, c);
      };
      simdSum = [this](scalar_type *a, cobj *b) { this->simdSumHalf(a, b); };
      multFact = 4;
      break;
    case ContractType::LeftHalf:
    case ContractType::RightHalf:
      vectorSum = [this](cobj *a, int b, int c) {
        this->vectorSumMixed(a, b, c);
      };
      simdSum = [this](scalar_type *a, cobj *b) { this->simdSumMixed(a, b); };
      multFact = 2;
      break;
    case ContractType::Full:
      vectorSum = [this](cobj *a, int b, int c) {
        this->vectorSumFull(a, b, c);
      };
      simdSum = [this](scalar_type *a, cobj *b) { this->simdSumFull(a, b); };
      multFact = 1;
      break;
    }

    int nGamma = this->getNgamma();
    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int MFrvol = gammaStride * nGamma;

    cobj *shm_p = (cobj *)acceleratorAllocDevice(MFrvol * sizeof(cobj));

    // Loop over gammas in batches of MF_SUM_ARRAY_MAX
    for (int mu = 0; mu < nGamma; mu += MF_SUM_ARRAY_MAX) {

      int nGammaBlock = std::min(nGamma - mu, MF_SUM_ARRAY_MAX);

      nvtxRangePushA("vectorSum");
      vectorSum(shm_p + mu * gammaStride, mu, nGammaBlock);
      nvtxRangePop();
    }

    nvtxRangePushA("simdSum");
    for (int mu = 0; mu < nGamma; mu++) {
      simdSum(result_p + mu * multFact * sizeR * sizeL * Nt,
              shm_p + mu * gammaStride);
    }
    nvtxRangePop();

    acceleratorFreeDevice(shm_p);
  }

  // Sums over SIMD vectorized results stored in `shm` parameter.
  // Case: Neither `_left_view` nor `_right_view` are checkerboarded.
  void simdSumFull(scalar_type *result, cobj *shm) {

    COMMON_VARS;

    auto shm_p = shm;
    auto result_p = result;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, 1, {
      ExtractBuffer<Scalar_s> extracted(simdSize);
      scalar_type temp;

      int shmem_idx = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        extract(shm_p[shmem_idx + rt], extracted);

        for (int simdOffset = 0; simdOffset < orthogSimdSize; simdOffset++) {

          temp = scalar_type(0.0);
          for (int idx = 0; idx < simdSize; idx++) {
            if (icoor_p[idx][orthogDir] == simdOffset) {
              temp += TensorRemove(extracted[idx]);
            }
            acceleratorSynchronise();
          }

          // Calculate local time from reduced time
          int lt = rt + simdOffset * reducedOrthogDimSize;
          int gt = lt + pc * localOrthogDimSize;

          int ij_dx = r_index + sizeR * (l_index + sizeL * gt);

          result_p[ij_dx] = temp;
        }
      }
    });
  }

  // Sums over SIMD vectorized results stored in `shm` parameter.
  // Case: Either `_left_view` or `_right_view` is checkerboarded, not both.
  void simdSumMixed(scalar_type *result, cobj *shm) {

    COMMON_VARS;

    auto shm_p = shm;
    auto result_p = result;

    bool fullLeft = this->_contract_type == ContractType::RightHalf;

    int multRight = fullLeft ? 2 : 1;
    int multLeft = fullLeft ? 1 : 2;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, 1, {
      ExtractBuffer<Scalar_s> extracted(simdSize);
      scalar_type temp;

      int shmem_idx = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        extract(shm_p[shmem_idx + rt], extracted);

        for (int simdOffset = 0; simdOffset < orthogSimdSize; simdOffset++) {

          temp = scalar_type(0.0);
          for (int idx = 0; idx < simdSize; idx++) {
            if (icoor_p[idx][orthogDir] == simdOffset) {
              temp += TensorRemove(extracted[idx]);
            }
            acceleratorSynchronise();
          }

          // Calculate local time from reduced time
          int lt = rt + simdOffset * reducedOrthogDimSize;
          int gt = lt + pc * localOrthogDimSize;

          int ij_dx =
              multRight * (r_index + sizeR * multLeft * (l_index + sizeL * gt));

          result_p[ij_dx] += temp;

          if (fullLeft) {
            if (cbEven && oddShifts) {
              result_p[ij_dx + 1] -= temp;
            } else if (cbEven) {
              result_p[ij_dx + 1] += temp;
            } else if (oddShifts) {
              result_p[ij_dx + 1] += temp;
            } else {
              result_p[ij_dx + 1] -= temp;
            }
            acceleratorSynchronise();
          } else {
            if (cbEven && oddShifts) {
              result_p[ij_dx + sizeR] += temp;
            } else if (cbEven) {
              result_p[ij_dx + sizeR] += temp;
            } else if (oddShifts) {
              result_p[ij_dx + sizeR] -= temp;
            } else {
              result_p[ij_dx + sizeR] -= temp;
            }
            acceleratorSynchronise();
          }
          acceleratorSynchronise();
        }
      }
    });
  }

  // Sums over SIMD vectorized results stored in `shm` parameter.
  // Case: Both `_left_view` and `_right_view` are checkerboarded.
  void simdSumHalf(scalar_type *result, cobj *shm) {

    COMMON_VARS;

    auto shm_p = shm;
    auto result_p = result;

    int sizeROut = 2 * sizeR;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, 1, {
      ExtractBuffer<Scalar_s> extracted(simdSize);
      scalar_type temp;

      int shmem_idx = reducedOrthogDimSize * (l_index + sizeL * r_index);

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        extract(shm_p[shmem_idx + rt], extracted);

        for (int simdOffset = 0; simdOffset < orthogSimdSize; simdOffset++) {

          temp = scalar_type(0.0);
          for (int idx = 0; idx < simdSize; idx++) {
            if (icoor_p[idx][orthogDir] == simdOffset) {
              temp += TensorRemove(extracted[idx]);
            }
            acceleratorSynchronise();
          }

          // Calculate local time and global time from reduced time
          int lt = rt + simdOffset * reducedOrthogDimSize;
          int gt = lt + pc * localOrthogDimSize;

          int ij_dx = 2 * (r_index + sizeR * 2 * (l_index + sizeL * gt));

          result_p[ij_dx] += temp;

          if (cbEven && oddShifts) {
            result_p[ij_dx + 1] -= temp;
            result_p[ij_dx + sizeROut + 1] -= temp;
            result_p[ij_dx + sizeROut] += temp;
          } else if (cbEven) {
            result_p[ij_dx + 1] += temp;
            result_p[ij_dx + sizeROut + 1] += temp;
            result_p[ij_dx + sizeROut] += temp;

          } else if (oddShifts) {
            result_p[ij_dx + 1] += temp;
            result_p[ij_dx + sizeROut + 1] -= temp;
            result_p[ij_dx + sizeROut] -= temp;
          } else {
            result_p[ij_dx + 1] -= temp;
            result_p[ij_dx + sizeROut + 1] += temp;
            result_p[ij_dx + sizeROut] -= temp;
          }
          acceleratorSynchronise();
        }
      }
    });
  }
};

template <typename FImpl> class A2ATaskLocal : public A2ATaskBase<FImpl> {
public:
  A2A_TYPEDEFS;

protected:
  std::vector<StagGamma::SpinTastePair> _gammas;
  std::vector<ComplexField> _phase;
  std::shared_ptr<A2AFieldView<cobj>> _phase_view;

public:
  A2ATaskLocal(GridBase *grid, int orthogDir, A2ATaskLocal<FImpl> &other,
               const std::vector<StagGamma::SpinTastePair> &gammas = {},
               int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {
    _phase_view = other.getPhaseView();
  }

  A2ATaskLocal(GridBase *grid, int orthogDir,
               const std::vector<StagGamma::SpinTastePair> &gammas,
               int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {

    int nGamma = _gammas.size();

    _phase.resize(nGamma, this->_full_grid);

    ComplexField temp(this->_full_grid);
    temp = 1.0;
    StagGamma spinTaste;

    _phase_view = std::make_shared<A2AFieldView<cobj>>();
    _phase_view->reserve(nGamma);

    for (int mu = 0; mu < nGamma; mu++) {

      spinTaste.setSpinTaste(_gammas[mu]);

      spinTaste.applyPhase(_phase[mu], temp); // store spin-taste phase
    }
    _phase_view->openViews(_phase.data(), nGamma);
  }

  A2ATaskLocal(GridBase *grid, int orthogDir,
               const std::vector<ComplexField> &gammas, int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb) {

    int nGamma = gammas.size();

    _phase_view = std::make_shared<A2AFieldView<cobj>>();
    _phase_view->reserve(nGamma);
    _phase_view->openViews(gammas.data(), nGamma);
  }

  virtual ~A2ATaskLocal() {
    if (_phase_view)
      _phase_view->closeViews();
  }

  std::shared_ptr<A2AFieldView<cobj>> getPhaseView() { return _phase_view; }

  virtual int getNgamma() { return _phase_view->size(); }

  virtual double getFlops() {
    // One complex multiply takes 6 floating point ops (4 mult, 2 add)
    // --> complex inner product is 3 complex mult, 2 complex add = 3*6 + 2*2 =
    // 22 double precision floating ops

    // For each vector and at each lattice site:
    //  - one inner product
    //  - For each gamma and momentum
    //    - multiply by gamma phase
    //    - sum
    return (22.0 + (6.0 + 2.0) * (this->getNgamma()));
  }

  virtual void vectorSumHalf(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    ComplexView *viewG_p = this->_phase_view->getView() + mu_offset;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {

          ss = rt * localSpatialVolume + so;

          temp_site = innerProduct(coalescedRead(viewL_p[l_index][ss]),
                                   coalescedRead(viewR_p[r_index][ss]));

          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[mu][ocoor_p[ss]]);
            sum[mu] += gamma_phase * temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumFull(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    ComplexView *viewG_p = this->_phase_view->getView() + mu_offset;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {

          ss = rt * localSpatialVolume + so;

          temp_site = innerProduct(coalescedRead(viewL_p[l_index][ss]),
                                   coalescedRead(viewR_p[r_index][ss]));

          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[mu][ss]);
            sum[mu] += gamma_phase * temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }

  virtual void vectorSumMixed(cobj *shm_p, int mu_offset, int N) {

    COMMON_VARS;

    ComplexView *viewG_p = this->_phase_view->getView() + mu_offset;

    assert(orthogDir ==
           Tdir); // This kernel assumes lattice is coalesced over time slices
    assert(nBlocks == 1);

    int gammaStride = sizeR * sizeL * reducedOrthogDimSize;
    int nGamma = N;

    bool checkerL = this->_contract_type == ContractType::LeftHalf;

    accelerator_for2d(l_index, sizeL, r_index, sizeR, simdSize, {
      int ss, shmem_base = reducedOrthogDimSize * (l_index + sizeL * r_index);
      calcScalar gamma_phase, temp_site, sum[MF_SUM_ARRAY_MAX];

      for (int rt = 0; rt < reducedOrthogDimSize; rt++) {

        for (int mu = 0; mu < nGamma; mu++) {
          sum[mu] = Zero();
        }

        for (int so = 0; so < localSpatialVolume; so++) {

          ss = rt * localSpatialVolume + so;

          if (checkerL) {
            temp_site =
                innerProduct(coalescedRead(viewL_p[l_index][ss]),
                             coalescedRead(viewR_p[r_index][ocoor_p[ss]]));
          } else {
            temp_site =
                innerProduct(coalescedRead(viewL_p[l_index][ocoor_p[ss]]),
                             coalescedRead(viewR_p[r_index][ss]));
          }
          acceleratorSynchronise();

          for (int mu = 0; mu < nGamma; mu++) {
            gamma_phase = coalescedRead(viewG_p[mu][ocoor_p[ss]]);
            sum[mu] += gamma_phase * temp_site;
          }
        }

        for (int mu = 0; mu < nGamma; mu++) {
          int shmem_idx = rt + shmem_base + mu * gammaStride;
          coalescedWrite(shm_p[shmem_idx], sum[mu]);
        }
      }
    });
  }
};

template <typename FImpl> class A2AWorkerBase {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

public:
  GridBase *_grid, *_cb_grid;

  double _flops, _t_kernel, _t_gsum;

  A2ATaskBase<FImpl> *_task_e, *_task_o;

  const FermionField *_l_addr, *_r_addr;

  scalar_type *_cache_device;
  size_t _cache_bytes = 0;

  bool _odd_shifts{false};

public:
  A2AWorkerBase() = delete;
  A2AWorkerBase(GridBase *grid)
      : _grid(grid), _l_addr(nullptr), _r_addr(nullptr) {}

  virtual ~A2AWorkerBase() {
    if (_cache_bytes != 0) {
      acceleratorFreeDevice(_cache_device);
    }
    delete _task_e;
    delete _task_o;
  }

public:
  template <typename TensorType> // output: rank 5 tensor, e.g.
                                 // Eigen::Tensor<ComplexD, 5>
  void StagMesonField(TensorType &mat, const FermionField *lhs_wi_E,
                      const FermionField *lhs_wi_O,
                      const FermionField *rhs_vj_E,
                      const FermionField *rhs_vj_O) {
    if (_cache_bytes < mat.size() * sizeof(scalar_type)) {
      if (_cache_bytes != 0) {
        acceleratorFreeDevice(_cache_device);
      }
      _cache_bytes = mat.size() * sizeof(scalar_type);
      std::cout << GridLogPerformance << "cache bytes: " << _cache_bytes
                << std::endl;
      _cache_device = (scalar_type *)acceleratorAllocDevice(_cache_bytes);
    }

    int sizeL = mat.dimension(3);
    int sizeR = mat.dimension(4);
    bool checkerL = lhs_wi_E[0].Grid()->_isCheckerBoarded;
    bool checkerR = rhs_vj_E[0].Grid()->_isCheckerBoarded;

    if (checkerL)
      sizeL /= 2;
    if (checkerR)
      sizeR /= 2;

    // scalar_type *matDevice = _cache_device;
    scalar_type *matDevice = mat.data();

    nvtxRangePushA("setLeft");
    if (_l_addr != lhs_wi_E) {
      _l_addr = lhs_wi_E;

      _task_e->setLeft(lhs_wi_E, sizeL);
      if (checkerL)
        _task_o->setLeft(lhs_wi_O, sizeL);
      else if (checkerR)
        _task_o->setLeft(*_task_e);
    }
    nvtxRangePop();

    nvtxRangePushA("setRight");
    if (_r_addr != rhs_vj_E) {
      _r_addr = rhs_vj_E;

      if (checkerR) {
        if (_odd_shifts) {
          _task_e->setRight(rhs_vj_O, sizeR);
          _task_o->setRight(rhs_vj_E, sizeR);
        } else {
          _task_e->setRight(rhs_vj_E, sizeR);
          _task_o->setRight(rhs_vj_O, sizeR);
        }
      } else {
        _task_e->setRight(rhs_vj_E, sizeR);
        if (checkerL) {
          _task_o->setRight(*_task_e);
        }
      }
    }
    nvtxRangePop();

    nvtxRangePushA("Sum: Total");
    nvtxRangePushA("Sum: Kernel");
    _t_kernel = -usecond();
    if (!(checkerL || checkerR)) {
      _task_e->execute(matDevice);
    } else {
      _task_e->execute(matDevice);
      _task_o->execute(matDevice);
    }
    nvtxRangePop();

    setFlops(_task_e->getFlops());
    _t_kernel += usecond();

    int resultStride = mat.dimension(2) * mat.dimension(3) * mat.dimension(4);
    int nGamma = mat.dimension(0) * mat.dimension(1);

    // auto matHost = mat.data();
    // size_t matBytes = mat.size()*sizeof(scalar_type);
    // acceleratorCopyFromDevice(matDevice,matHost,matBytes);

    nvtxRangePushA("Sum: Global");
    _t_gsum = -usecond();
    int comm_batch = std::pow(2, 20);
    int offset = 0;
    int buff = comm_batch;

    this->_grid->GlobalSumVector(matDevice, nGamma * resultStride);
    _t_gsum += usecond();
    nvtxRangePop();
    nvtxRangePop();
  }
  void setFlops(double flops) { _flops = flops; }
  double getFlops() { return _flops; }
};

template <typename FImpl> class A2AWorkerLocal : public A2AWorkerBase<FImpl> {
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_type scalar_type;

public:
  A2AWorkerLocal() = delete;
  A2AWorkerLocal(GridBase *grid, const std::vector<ComplexField> &mom,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 int orthogDir)
      : A2AWorkerBase<FImpl>(grid) {
    this->_odd_shifts = false;
    if (mom.size()) {
      assert(0);
    } else {
      this->_task_e = new A2ATaskLocal<FImpl>(grid, orthogDir, gammas, Even);
      this->_task_o = new A2ATaskLocal<FImpl>(
          grid, orthogDir, dynamic_cast<A2ATaskLocal<FImpl> &>(*this->_task_e),
          gammas, Odd);
    }
  }
};
#undef A2A_TYPEDEFS
#undef COMMON_VARS

NAMESPACE_END(Grid);
