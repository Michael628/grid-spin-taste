/*
 * MesonField.hpp, part of Hadrons (https://github.com/aportelli/Hadrons)
 *
 * Copyright (C) 2015 - 2020
 *
 * Author: Antonin Portelli <antonin.portelli@me.com>
 * Author: Peter Boyle <paboyle@ph.ed.ac.uk>
 * Author: ferben <ferben@debian.felix.com>
 * Author: paboyle <paboyle@ph.ed.ac.uk>
 * Author: Michael Lynch <michaellynch628@gmail.com>
 *
 * Hadrons is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * Hadrons is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Hadrons.  If not, see <http://www.gnu.org/licenses/>.
 *
 * See the full license in the file "LICENSE" in the top level distribution
 * directory.
 */

/*  END LEGAL */
#ifndef A2AMesonField_h_
#define A2AMesonField_h_

#include <A2AMatrix.h>
#include <Eigenpack.h>
#include <IO.h>
#include <StagGamma.h>
#include <a2a/A2AWorker.h>

NAMESPACE_BEGIN(Grid)

class MesonFieldMetadata : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFieldMetadata, std::vector<RealF>,
                                  momentum, StagGamma::StagAlgebra, gamma_spin,
                                  StagGamma::StagAlgebra, gamma_taste);

  MesonFieldMetadata()
      : momentum{}, gamma_spin(StagGamma::StagAlgebra::undef),
        gamma_taste(StagGamma::StagAlgebra::undef) {}
};

template <typename FImpl, typename Pack> class TMesonField {
public:
  FERM_TYPE_ALIASES(FImpl, );
  typedef A2AMatrixBlockComputation<ComplexD, FermionField, MesonFieldMetadata,
                                    HADRONS_A2AM_IO_TYPE>
      Computation;
  typedef MesonFieldKernel<Complex, FImpl> Kernel;

public:
  // constructor
  TMesonField(GridBase *grid, MesonFieldPar &par);
  // destructor
  virtual ~TMesonField(void) {};

  virtual void setup(void);
  // execution
  virtual void execute(void);

private:
  GridBase *_grid;
  bool _hasPhase{false};
  std::vector<StagGamma::SpinTastePair> _gammas, _gammaComms, _gammaLocal;
  std::vector<std::vector<Real>> _mom;
  Pack *_low_modes = nullptr;
  MesonFieldPar &_par;
};

template <typename FImpl, typename Pack>
TMesonField<FImpl, Pack>::TMesonField(GridBase *grid, MesonFieldPar &par)
    : _grid(grid), _par(par) {}

template <typename FImpl, typename Pack>
void TMesonField<FImpl, Pack>::setup() {
  _gammas = StagGamma::ParseSpinTasteString(_par.spinTaste.gammas,
                                            _par.spinTaste.applyG5);

  _gammaComms.clear();
  _gammaLocal.clear();

  StagGamma spinTaste;
  for (auto &g : _gammas) {
    spinTaste.setSpinTaste(g);

    if (spinTaste._spin ^ spinTaste._taste) {
      _gammaComms.push_back(g);
    } else {
      _gammaLocal.push_back(g);
    }
  }

  _mom.clear();

  for (auto &pstr : _par.mom) {
    auto p = strToVec<Real>(pstr);

    // if (p.size() != env().getNd() - 1) {
    //   HADRONS_ERROR(Size, "Momentum has " + std::to_string(p.size()) +
    //                           " components instead of " +
    //                           std::to_string(env().getNd() - 1));
    // }
    _mom.push_back(p);
  }
  int nmom = _mom.size();
  bool allzero = true;
  if (_par.mom.size() == 1) {
    for (auto p : _mom[0]) {
      if (p != 0)
        allzero = false;
    }
  }
  if (allzero)
    nmom = 0;

  // TODO: Implement non-zero momentum
  // envCache(std::vector<ComplexField>, _momphName, 1, nmom, _grid);
  std::shared_ptr<std::vector<ComplexField>> ph =
      make_shared<std::vector<ComplexField>>(_grid);

  envTmpLat(ComplexField, "coor");

  std::shared_ptr<Computation> computationLocal =
      std::make_shared<Computation>(_grid, env().getNd() - 1, _mom.size(),
                                    _gammaLocal.size(), this->_par.block, this);

  envTmp(Computation, "computationComms", 1, envGetGrid(FermionField),
         env().getNd() - 1, _mom.size(), _gammaComms.size(), this->_par.block,
         this);
}

// execution ///////////////////////////////////////////////////////////////////
template <typename FImpl, typename Pack>
void TMesonField<FImpl, Pack>::execute(void) {
  bool hasLowModes = this->_low_modes != nullptr;
  bool isCheckerBoarded = (!par.action.empty());

  // std::vector<FermionField> *left, *right;

  // TODO: Set left and right fields with method call
}
// int N_i = left->size();
// int N_j = right->size();

if (hasLowModes) {
  auto &lowModes = envGet(Pack, this->_par.lowModes);
  if (N_j != 0 && N_i == 0) {
    N_i += (isCheckerBoarded ? 2 : 1) * lowModes.evec.size();
  } else if (N_i != 0 && N_j == 0) {
    N_j += (isCheckerBoarded ? 2 : 1) * lowModes.evec.size();
  } else {
    N_i += (isCheckerBoarded ? 2 : 1) * lowModes.evec.size();
    N_j += (isCheckerBoarded ? 2 : 1) * lowModes.evec.size();
  }
}
int block = this->_par.block;

/*if (N_i < block || N_j < block)
{
    HADRONS_ERROR(Range, "blockSize must not exceed size of input vector.");
}*/

LOG(Message) << "Computing all-to-all meson fields" << std::endl;
if (hasLowModes)
  LOG(Message) << "Low Modes: '" << this->_par.lowModes << "'" << std::endl;

if (!(this->_par.left.empty() && this->_par.right.empty())) {
  if (!this->_par.left.empty())
    LOG(Message) << "Left: '" << this->_par.left << "'" << std::endl;
  if (!this->_par.right.empty())
    LOG(Message) << "Right: '" << this->_par.right << "'" << std::endl;
}

LOG(Message) << "Momenta:" << std::endl;

for (auto &p : _mom) {
  LOG(Message) << "  " << p << std::endl;
}

LOG(Message) << "Spin bilinears:" << std::endl;

for (auto &g : _gammas) {
  LOG(Message) << "  " << StagGamma::GetName(g) << std::endl;
}

LOG(Message) << "Meson field size: " << Nt << "*" << N_i << "*" << N_j
             << " (filesize "
             << sizeString(Nt * N_i * N_j * sizeof(HADRONS_A2AM_IO_TYPE))
             << "/momentum/bilinear)" << std::endl;

std::vector<ComplexField> ph;
// TODO: Implement non-zero momentum
// auto &ph = envGet(std::vector<ComplexField>, _momphName);
// startTimer("Momentum phases");
// for (unsigned int j = 0; j < ph.size(); ++j) {
//   Complex i(0.0, 1.0);
//   std::vector<Real> p;
//
//   envGetTmp(ComplexField, coor);
//   ph[j] = Zero();
//   for (unsigned int mu = 0; mu < _mom[j].size(); mu++) {
//     LatticeCoordinate(coor, mu);
//     ph[j] = ph[j] + (_mom[j][mu] / env().getDim(mu)) * coor;
//   }
//   ph[j] = exp((Real)(2 * M_PI) * i * ph[j]);
// }
// stopTimer("Momentum phases");

auto gammaIOnameFn = [this](const unsigned int m, const unsigned int g) {
  std::stringstream ss;

  ss << StagGamma::GetName(_gammas[g]) << "_";

  for (unsigned int mu = 0; mu < _mom[m].size(); ++mu) {
    ss << _mom[m][mu] << ((mu == _mom[m].size() - 1) ? "" : "_");
  }

  return ss.str();
};

auto gammaFilenameFn = [this, &gammaIOnameFn](const unsigned int m,
                                              const unsigned int g) {
  return this->_par.output + "." + std::to_string(vm().getTrajectory()) + "/" +
         gammaIOnameFn(m, g) + ".h5";
};

auto gammaMetadataFn = [this](const unsigned int m, const unsigned int g) {
  MesonFieldMetadata md;

  for (auto pmu : _mom[m]) {
    md.momentum.push_back(pmu);
  }

  md.gamma_spin = _gammas[g].first;
  md.gamma_taste = _gammas[g].second;

  return md;
};

envGetTmp(Computation, computationLocal);
envGetTmp(Computation, computationComms);

Kernel kernel(envGetGrid(FermionField));

GaugeField *U = nullptr;
if (!this->par.spinTaste.gauge.empty()) {
  U = env().template getObject<GaugeField>(this->par.spinTaste.gauge);
}

int orthogDir = env().getNd() - 1;

if (hasLowModes) {
  auto &lowModes = envGet(Pack, this->_par.lowModes);

  if (isCheckerBoarded) {

    auto &action = envGet(FMat, this->_par.action);
    std::function<void(int)> swapEvecCheckerFn = [this, &action,
                                                  &lowModes](int index) {
      ComplexD eval_D = ComplexD(0.0, lowModes.eval[index].imag());
      int cb = lowModes.evec[index].Checkerboard();
      int cbNeg = (cb == Even) ? Odd : Even;

      FermionField temp(lowModes.evec[index].Grid());
      temp.Checkerboard() = cbNeg;
      action.Meooe(lowModes.evec[index], temp);
      lowModes.evec[index].Checkerboard() = cbNeg;
      lowModes.evec[index] = (1.0 / eval_D) * temp;
    };

    if (_gammaLocal.size() > 0) {
      _gammas = _gammaLocal;
      kernel.setWorker(envGetGrid(FermionField), ph, _gammas, orthogDir);
      computationLocal.execute(*left, *right, kernel, gammaIOnameFn,
                               gammaFilenameFn, gammaMetadataFn, &lowModes.evec,
                               lowModes.eval, &swapEvecCheckerFn);
    }
    if (_gammaComms.size() > 0) {
      _gammas = _gammaComms;
      kernel.setWorker(envGetGrid(FermionField), ph, _gammas, orthogDir, U);
      computationComms.execute(*left, *right, kernel, gammaIOnameFn,
                               gammaFilenameFn, gammaMetadataFn, &lowModes.evec,
                               lowModes.eval, &swapEvecCheckerFn);
    }
  } else {
    if (_gammaLocal.size() > 0) {
      _gammas = _gammaLocal;
      kernel.setWorker(envGetGrid(FermionField), ph, _gammas, orthogDir);
      computationLocal.execute(*left, *right, kernel, gammaIOnameFn,
                               gammaFilenameFn, gammaMetadataFn, &lowModes.evec,
                               lowModes.eval);
    }
    if (_gammaComms.size() > 0) {
      _gammas = _gammaComms;
      kernel.setWorker(envGetGrid(FermionField), ph, _gammas, orthogDir, U);
      computationComms.execute(*left, *right, kernel, gammaIOnameFn,
                               gammaFilenameFn, gammaMetadataFn, &lowModes.evec,
                               lowModes.eval);
    }
  }
} else {
  if (_gammaLocal.size() > 0) {
    _gammas = _gammaLocal;
    kernel.setWorker(envGetGrid(FermionField), ph, _gammas, orthogDir);
    computationLocal.execute(*left, *right, kernel, gammaIOnameFn,
                             gammaFilenameFn, gammaMetadataFn);
  }
  if (_gammaComms.size() > 0) {
    _gammas = _gammaComms;
    kernel.setWorker(envGetGrid(FermionField), ph, _gammas, orthogDir, U);
    computationComms.execute(*left, *right, kernel, gammaIOnameFn,
                             gammaFilenameFn, gammaMetadataFn);
  }
}
}

END_MODULE_NAMESPACE

END_HADRONS_NAMESPACE

#endif // Hadrons_MContraction_MesonField_h_
