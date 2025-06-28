#pragma once

#include <Grid/Grid.h>
#include <array>
#include <iostream>

NAMESPACE_BEGIN(Grid)

class StagGamma {
public:
  // XYZT convention
  // GRID_SERIALIZABLE_ENUM(StagAlgebra, undef, G1, 0, GT, 1, GZ, 2, GZT, 3, GY,
  // 4,
  //                        GYT, 5, GYZ, 6, G5X, 7, GX, 8, GXT, 9, GZX, 10, G5Y,
  //                        11, GXY, 12, G5Z, 13, G5T, 14, G5, 15);

  // TXYZ convention
  // clang-format off
  GRID_SERIALIZABLE_ENUM(StagAlgebra, undef, 
                         G1,  0b0000,
                         GZ,  0b0001,
                         GY,  0b0010,
                         GYZ, 0b0011,
                         GX,  0b0100,
                         GZX, 0b0101,
                         GXY, 0b0110,
                         G5T, 0b0111,
                         GT,  0b1000,
                         GZT, 0b1001,
                         GYT, 0b1010,
                         G5X, 0b1011,
                         GXT, 0b1100,
                         G5Y, 0b1101,
                         G5Z, 0b1110,
                         G5,  0b1111);
  // clang-format on

  typedef std::pair<StagAlgebra, StagAlgebra> SpinTastePair;

public:
  StagGamma() : _spin(0), _taste(0) {}

  StagGamma(StagAlgebra spin, StagAlgebra taste) {
    _spin = spin;
    _taste = taste;
    calculatePhase();
  }

  StagGamma(SpinTastePair st) { StagGamma(st.first, st.second); }

  void setGaugeField(LatticeGaugeField &U) { _U = &U; }

  inline bool isLocal() const { return !(_spin ^ _taste); }

  inline void setSpin(StagAlgebra spin) {
    _spin = spin;
    calculatePhase();
  }

  inline void setTaste(StagAlgebra taste) {
    _taste = taste;
    calculatePhase();
  }

  inline void setSpinTaste(StagAlgebra spin, StagAlgebra taste) {
    _spin = spin;
    _taste = taste;
    calculatePhase();
  }

  inline void setSpinTaste(SpinTastePair st) {
    setSpinTaste(st.first, st.second);
  }

  static std::vector<StagGamma::SpinTastePair>
  ParseSpinTaste(std::string stString, bool applyG5 = false) {
    auto spinTastes = strToVec<StagGamma::SpinTastePair>(stString);

    if (applyG5) {
      StagGamma gamma;
      StagGamma g5g5(StagGamma::StagAlgebra::G5, StagGamma::StagAlgebra::G5);

      for (auto &st : spinTastes) {
        gamma.setSpinTaste(st);
        gamma = gamma * g5g5;
        st.first = gamma._spin;
        st.second = gamma._taste;
      }
    }

    return spinTastes;
  }
  static std::string GetName(StagAlgebra spin, StagAlgebra taste) {

    std::string name = StagGamma::name[spin];
    name = (name + "_") + StagGamma::name[taste];

    return name;
  }

  static std::string GetName(SpinTastePair st) {
    return StagGamma::GetName(st.first, st.second);
  }

  std::string getName() const { return StagGamma::GetName(_spin, _taste); }

  template <typename obj>
  void applyGamma(Lattice<obj> &lhs, const Lattice<obj> &rhs) const;

  template <typename obj>
  void applyPhase(Lattice<obj> &lhs, const Lattice<obj> &rhs) const;

  template <typename obj>
  void oneLink(Lattice<obj> &lhs, const Lattice<obj> &rhs, int shift_dir) const;

  template <typename obj>
  inline void operator()(Lattice<obj> &lhs, const Lattice<obj> &rhs) const {
    applyGamma(lhs, rhs);
  }

private:
  // Calculate the < and > operations as defined in Follana (2007) eqns A5 and
  // A7.
  static inline StagAlgebra LessThan(StagAlgebra alg);
  static inline StagAlgebra GreaterThan(StagAlgebra alg);

  // Implements eqn. E3 of Follana (2007)
  inline void calculatePhase();

  // Implements (-1)^(x[mu] * ( _taste^< + _spin^> ) ( see eqn. E3 of Follana
  // (2007) )
  inline void calculateOscillation();

  // Implements (-1)^(_spin * (_spin + _taste)^<) ( see eqn. E3 of Follana
  // (2007) )
  inline void calculateNegation();

  inline void toggleNegation() { _negated = !_negated; }

public:
  static constexpr unsigned int nGamma = 16;
  // TXYZ convention

  static inline const std::array<const char *, nGamma> name = {
      {"G1", "GZ", "GY", "GYZ", "GX", "GZX", "GXY", "G5T", "GT", "GZT", "GYT",
       "G5X", "GXT", "G5Y", "G5Z", "G5"}};

  static inline const std::array<const StagAlgebra, 4> gmu = {{
      // Index ordering that matches Grid convention, 0=X, 1=Y, 2=Z, 3=T
      StagGamma::StagAlgebra::GX,
      StagGamma::StagAlgebra::GY,
      StagGamma::StagAlgebra::GZ,
      StagGamma::StagAlgebra::GT,
  }};

  friend inline StagGamma operator*(const StagGamma &g1, const StagGamma &g2);

public:
  StagAlgebra _spin, _taste;
  LatticeGaugeField *_U = nullptr;

private:
  StagAlgebra _oscillateDirs = 0b0000;
  bool _negated = false;
  RealD _scaling;
};

inline StagGamma::StagAlgebra StagGamma::LessThan(StagAlgebra alg) {
  uint8_t ret = 0;
  uint8_t gammaMask = alg;

  for (int i = 0; i < Nd - 1; i++) {
    gammaMask = gammaMask >> 1;
    // each bit will toggle for each 1 that passes over it
    ret = ret ^ gammaMask;
  }
  return ret;
}

inline StagGamma::StagAlgebra StagGamma::GreaterThan(StagAlgebra alg) {
  uint8_t ret = 0;
  uint8_t gammaMask = alg;

  for (int i = 0; i < Nd - 1; i++) {
    gammaMask = gammaMask << 1;
    // each bit will toggle for each 1 that passes over it
    ret = ret ^ gammaMask;
  }
  return (0b1111 & ret); // Zero out `ret` after fourth bit
}

template <class obj>
void StagGamma::applyGamma(Lattice<obj> &lhs, const Lattice<obj> &rhs) const {

  if (!this->isLocal()) {
    assert(_U != nullptr);
  }

  uint8_t shift = _spin ^ _taste;

  switch (shift) {
  case StagAlgebra::G1:
    applyPhase(lhs, rhs);
    break;
  case StagAlgebra::GT:
    oneLink(lhs, rhs, 0);
    applyPhase(lhs, lhs);
    break;
  case StagAlgebra::GZ:
    oneLink(lhs, rhs, 1);
    applyPhase(lhs, lhs);
    break;
  case StagAlgebra::GY:
    oneLink(lhs, rhs, 2);
    applyPhase(lhs, lhs);
    break;
  case StagAlgebra::GX:
    oneLink(lhs, rhs, 3);
    applyPhase(lhs, lhs);
    break;
  default:
    // TODO: Handle all spin-taste shifts
    assert(0);
  }
}

template <class obj>
void StagGamma::oneLink(Lattice<obj> &lhs, const Lattice<obj> &rhs,
                        int shiftdir) const {

  Lattice<obj> temp(rhs.Grid());
  LatticeColourMatrix Umu(rhs.Grid());

  if (rhs.Grid()->_isCheckerBoarded) {
    LatticeColourMatrix Umu_full(_U->Grid());
    Umu_full = PeekIndex<LorentzIndex>(*_U, shiftdir);
    pickCheckerboard(rhs.Checkerboard(), Umu, Umu_full);
    temp = adj(Umu) * rhs;
    pickCheckerboard(lhs.Checkerboard(), Umu, Umu_full);
  } else {
    Umu = PeekIndex<LorentzIndex>(*_U, shiftdir);
    temp = adj(Umu) * rhs;
  }
  lhs = Cshift(temp, shiftdir, -1);
  temp = Cshift(rhs, shiftdir, 1);
  lhs += Umu * temp;
}

inline void StagGamma::calculateNegation() {
  StagAlgebra result = _spin & LessThan(_spin ^ _taste);

  for (auto &dir : StagGamma::gmu) {
    if (dir & result) {
      toggleNegation();
    }
  }
}

inline void StagGamma::calculateOscillation() {
  _oscillateDirs = LessThan(_taste) ^ GreaterThan(_spin);
}

inline void StagGamma::calculatePhase() {

  // scale down according to number of terms in symmetric shift
  switch (_spin ^ _taste) {
  case StagAlgebra::G1:
    _scaling = 1.0;
    break;
  case StagAlgebra::GT:
  case StagAlgebra::GZ:
  case StagAlgebra::GY:
  case StagAlgebra::GX:
    _scaling = 0.5;
    break;
  default:
    // TODO: Handle all spin-taste shifts
    assert(0);
  }

  _negated = false;
  calculateNegation();

  // TODO: Include sign flip for consistent orientation of gammas ( see eqn. A4
  // of Follana (2007) )
}

template <class obj>
void StagGamma::applyPhase(Lattice<obj> &lhs, const Lattice<obj> &rhs) const {

  GridBase *grid = lhs.Grid();

  Lattice<obj> temp(grid);
  Lattice<iScalar<vInteger>> coor(grid), negate(grid);
  iScalar<vInteger> one = 1;

  if (_negated) {
    negate = one;
  } else {
    negate = Zero();
  }

  for (int dir = 0; dir < gmu.size(); dir++) {
    if (gmu[dir] & _oscillateDirs) { // gmu[dir] maps Grid XYZT convention to
                                     // our current binary convention
      LatticeCoordinate(coor, dir);
      negate += coor;
    }
  }

  temp = where(mod(negate, 2) == 0, _scaling * rhs, -_scaling * rhs);

  lhs = std::move(temp);
}

inline StagGamma operator*(const StagGamma &g1, const StagGamma &g2) {

  StagGamma ret(g1._spin ^ g2._spin, g1._taste ^ g2._taste);

  if (!ret.isLocal()) {
    if (g1._U != nullptr) {
      if (g2._U != nullptr) {
        assert(g2._U == g1._U);
      }
      ret.setGaugeField(*(g1._U));
    } else if (g2._U != nullptr) {
      ret.setGaugeField(*(g2._U));
    } else {
      assert(0);
    }
  }
  if (g1._negated != g2._negated) {
    ret.toggleNegation();
  }

  // Following eqn. A4 of Follana (2007)
  uint8_t negate = ((g1._spin & StagGamma::LessThan(g2._spin)) ^
                    (g1._taste & StagGamma::LessThan(g2._taste)));

  for (auto &dir : StagGamma::gmu) {
    if (dir & negate) {
      ret.toggleNegation();
    }
  }

  return ret;
}
inline void operator*=(StagGamma &g1, const StagGamma &g2) { g1 = g1 * g2; }

template <class obj>
inline Lattice<obj> operator*(const StagGamma &g1, const Lattice<obj> &lat) {

  Lattice<obj> temp(lat.Grid());
  g1.applyGamma(temp, lat);
  return temp;
}

template <class obj>
inline Lattice<obj> operator*(const Lattice<obj> &lat, const StagGamma &g1) {
  // BUG: This doen't work. gammas are not commutative with all lattice fields
  return g1 * lat;
}

NAMESPACE_END(Grid)
