#ifndef FMGRID_IO_H
#define FMGRID_IO_H

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/LocalCoherenceLanczos.h>
#include <StagGamma.h>

NAMESPACE_BEGIN(Grid);

#ifdef HAVE_HDF5
typedef Hdf5Reader ResultReader;
typedef Hdf5Writer ResultWriter;
#else
typedef XmlReader ResultReader;
typedef XmlWriter ResultWriter;
#endif

// ============================================================================
// MAction Module Parameter Classes
// ============================================================================
class ImprovedStaggeredMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(ImprovedStaggeredMILCPar, std::string,
                                  gaugefat, std::string, gaugelong, double,
                                  mass, double, c1, double, c2, double, tad,
                                  std::string, boundary, std::string, twist);

  std::string parString(void) const {
    XmlWriter writer("", "");
    write(writer, "ImprovedStaggeredMILCPar", *this);
    return writer.string();
  }
};

class GaugePropMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GaugePropMILCPar, std::string, source,
                                  SpinTasteParams, spinTaste, std::string,
                                  solver);
};

// ============================================================================
// MSolver Module Parameter Classes
// ============================================================================
class ImplicitlyRestartedLanczosMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(ImplicitlyRestartedLanczosMILCPar,
                                  LanczosParams, lanczosParams, bool,
                                  evenEigen);

  std::string parString(void) const {
    XmlWriter writer("", "");
    write(writer, "ImplicitlyRestartedLanczosMILCPar", *this);
    return writer.string();
  }
};

class LowModeProjMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(LowModeProjMILCPar, ImprovedStaggeredMILCPar,
                                  action, bool, projector, unsigned int,
                                  eigStart, int, nEigs, std::string, lowModes);
};

class MixedPrecisionCGMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MixedPrecisionCGMILCPar,
                                  ImprovedStaggeredMILCPar, action,
                                  unsigned int, maxInnerIteration, unsigned int,
                                  maxOuterIteration, double, residual);
};

// ============================================================================
// MSink Module Parameter Classes
// ============================================================================
class PointPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(PointPar, std::string, mom);
};

// ============================================================================
// MSource Module Parameter Classes
// ============================================================================
class RandomWallMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(RandomWallMILCPar, unsigned int, tStep,
                                  unsigned int, t0, unsigned int, nSrc,
                                  std::string, seed);
  RandomWallMILCPar() : seed("default_seed") {}
};

// ============================================================================
// MContraction Module Parameter Classes
// ============================================================================
class ContractionPar : Serializable {
public:
  GRID_SERIALIZABLE_ENUM(SolverType, undef, lma, 0, mpcg, 1);
  GRID_SERIALIZABLE_CLASS_MEMBERS(ContractionPar, SolverType, solver,
                                  SpinTasteParams, quark, SpinTasteParams,
                                  antiquark, SpinTasteParams, sink, std::string,
                                  output);
};

class MesonResult : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonResult, std::string, sourceGamma,
                                  std::string, sinkGamma, std::vector<Complex>,
                                  corr, std::vector<std::vector<Complex>>,
                                  srcCorrs, std::vector<Integer>, timeShifts,
                                  Real, scaling);
};

class MesonFieldMILCPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFieldMILCPar, int, block, std::string,
                                  lowModes, std::string, left, std::string,
                                  action, std::string, right, std::string,
                                  output, SpinTasteParams, spinTaste,
                                  std::vector<std::string>, mom);
};

// ============================================================================
// High-level Parameter Classes
// ============================================================================
class GaugePar : Serializable {
public:
  GRID_SERIALIZABLE_ENUM(GaugeType, undef, free, 0, file, 1, hot, 2);

  GRID_SERIALIZABLE_CLASS_MEMBERS(GaugePar, std::string, link, std::string,
                                  fatlink, std::string, longlink, GaugeType,
                                  type);
};

class EpackPar : Serializable {
public:
  GRID_SERIALIZABLE_ENUM(EpackType, undef, load, 0, solve, 1);
  GRID_SERIALIZABLE_CLASS_MEMBERS(EpackPar, ImprovedStaggeredMILCPar, action,
                                  ImplicitlyRestartedLanczosMILCPar, irl,
                                  std::string, evalSave, EpackType, type,
                                  unsigned int, size, std::string, file, bool,
                                  multiFile);
};

class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, GaugePar, gauge, EpackPar, epack,
                                  LowModeProjMILCPar, lma,
                                  MixedPrecisionCGMILCPar, mpcg,
                                  GaugePropMILCPar, gaugeProp, ContractionPar,
                                  corr, MesonFieldMILCPar, a2a,
                                  std::vector<RandomWallMILCPar>, sources,
                                  std::string, series, unsigned int,
                                  trajectory);
};

int mkdir(const std::string dirName);
std::string dirname(const std::string &s);
void makeFileDir(const std::string filename, GridBase *g);
std::string resultFilename(const std::string stem, const GlobalPar &inputParams,
                           const std::string ext);
template <typename T>
void saveResult(GridBase *grid, const std::string stem, const std::string name,
                const T &result, const GlobalPar &inputParams,
                const std::string ext = "h5") {
  if (grid->IsBoss() and !stem.empty()) {
    makeFileDir(stem, grid);
    {
      ResultWriter writer(resultFilename(stem, inputParams, ext));
      write(writer, name, result);
    }
  }
}
template <typename T>
void saveResult(GridBase *grid, const std::string stem, const std::string name,
                const T &result, const GlobalPar &inputParams,
                const int tsource, const std::string ext = "h5") {
  std::string stem_with_tsource = stem + "_t" + std::to_string(tsource);
  saveResult(grid, stem_with_tsource, name, result, inputParams, ext);
}

NAMESPACE_END(Grid);

#endif
