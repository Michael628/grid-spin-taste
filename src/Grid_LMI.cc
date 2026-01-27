#include <DilutedNoise.hpp>
#include <Eigenpack.h>
#include <Grid/Grid.h>
#include <IO.h>
#include <LowModeProj.h>
#include <StagGamma.h>
#include <a2a/A2AWorker.h>
#include <functional>

using namespace std;
using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 1;

  typedef ImprovedStaggeredFermionD FermionOpD;
  typedef ImprovedStaggeredFermionF FermionOpF;
  typedef typename ImprovedStaggeredFermionD::ImplParams ImplParams;
  typedef typename ImprovedStaggeredFermionD::Impl_t FImpl;
  typedef typename ImprovedStaggeredFermionD::PropagatorField PropagatorFieldD;
  typedef typename ImprovedStaggeredFermionD::FermionField FermionFieldD;
  typedef typename ImprovedStaggeredFermionF::FermionField FermionFieldF;

  std::string paramFile = argv[1];
  XmlReader reader(paramFile, false, "grid");

  GlobalPar inputParams;
  read(reader, "parameters", inputParams);

  auto latt = GridDefaultLatt();
  auto nsimd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  auto nsimdf = GridDefaultSimd(Nd, vComplexF::Nsimd());
  auto mpi_layout = GridDefaultMpi();
  // ========================================================================
  // SETUP: Grid communicator layouts
  // ========================================================================
  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), nsimd, GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian *UGridF = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), nsimdf, GridDefaultMpi());
  GridRedBlackCartesian *UrbGridF =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);

  GridParallelRNG rng(UGrid);
  // ========================================================================
  // MODULE: MIO::LoadIldg (Load gauge configurations)
  // ========================================================================
  std::cout << GridLogMessage
            << "========================================" << std::endl;
  std::cout << GridLogMessage << "MODULE: MIO::LoadIldg" << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  LatticeGaugeFieldD U(UGrid);
  LatticeGaugeFieldD U_fat(UGrid);
  LatticeGaugeFieldD U_long(UGrid);

  FieldMetaData header;
  int traj = inputParams.trajectory;
  IldgReader IR;

  switch (inputParams.gauge.type) {
  case GaugePar::GaugeType::free:
    SU<Nc>::ColdConfiguration(U);
    SU<Nc>::ColdConfiguration(U_fat);
    SU<Nc>::ColdConfiguration(U_long);
    break;
  case GaugePar::GaugeType::file: {
    // Load fat links (double precision)
    std::string file_fat =
        inputParams.gauge.fatlink + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading fat links from " << file_fat
              << std::endl;
    IR.open(file_fat);
    IR.readConfiguration(U_fat, header);
    IR.close();

    // Load long links (double precision)
    std::string file_long =
        inputParams.gauge.longlink + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading long links from " << file_long
              << std::endl;
    IR.open(file_long);
    IR.readConfiguration(U_long, header);
    IR.close();

    // Load base gauge field (double precision)
    std::string file_base = inputParams.gauge.link + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading base gauge field from " << file_base
              << std::endl;
    IR.open(file_base);
    IR.readConfiguration(U, header);
    IR.close();
  } break;
  case GaugePar::GaugeType::hot:
    SU<Nc>::HotConfiguration(rng, U);
    SU<Nc>::HotConfiguration(rng, U_fat);
    SU<Nc>::HotConfiguration(rng, U_long);
    break;
  }

  // ========================================================================
  // MODULE: MUtilities::GaugeSinglePrecisionCast (Cast to single precision)
  // ========================================================================
  std::cout << GridLogMessage
            << "\n========================================" << std::endl;
  std::cout << GridLogMessage << "MODULE: MUtilities::GaugeSinglePrecisionCast"
            << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  LatticeGaugeFieldF U_fat_f(UGridF);
  LatticeGaugeFieldF U_long_f(UGridF);

  std::cout << GridLogMessage << "Casting fat links to single precision"
            << std::endl;
  precisionChange(U_fat_f, U_fat);

  std::cout << GridLogMessage << "Casting long links to single precision"
            << std::endl;
  precisionChange(U_long_f, U_long);

  ImplParams implParams;

  auto makeAction = [&UGrid, &UrbGrid, &UGridF, &UrbGridF, &U_long, &U_fat,
                     &U_long_f, &U_fat_f, &implParams](
                        auto &action, ImprovedStaggeredMILCPar actionPar) {
    // ========================================================================
    // MODULE: MAction::ImprovedStaggeredMILC (Create fermion action)
    // ========================================================================
    std::cout << GridLogMessage
              << "\n========================================" << std::endl;
    std::cout << GridLogMessage << "MODULE: MAction::ImprovedStaggeredMILC"
              << std::endl;
    std::cout << GridLogMessage
              << "========================================" << std::endl;

    std::cout << GridLogMessage << "\nCreating ImprovedStaggeredFermion "
              << std::endl;
    std::cout << GridLogMessage << "  mass = " << actionPar.mass << std::endl;
    std::cout << GridLogMessage << "  c1 = " << actionPar.c1 << std::endl;
    std::cout << GridLogMessage << "  c2 = " << actionPar.c2 << std::endl;
    std::cout << GridLogMessage << "  tadpole = " << actionPar.tad << std::endl;

    using T = std::decay_t<decltype(action)>;
    if constexpr (std::is_same_v<T, std::shared_ptr<FermionOpF>>) {
      action = std::make_shared<FermionOpF>(
          *UGridF, *UrbGridF, 2. * actionPar.mass, 2. * actionPar.c1,
          2. * actionPar.c2, actionPar.tad, implParams);
      action->ImportGaugeSimple(U_long_f, U_fat_f);
    } else if constexpr (std::is_same_v<T, std::shared_ptr<FermionOpD>>) {
      action = std::make_shared<FermionOpD>(
          *UGrid, *UrbGrid, 2. * actionPar.mass, 2. * actionPar.c1,
          2. * actionPar.c2, actionPar.tad, implParams);
      action->ImportGaugeSimple(U_long, U_fat);
    }
  };

  bool hasEigs = inputParams.epack.type != EpackPar::EpackType::undef;
  bool hasSources = inputParams.sources.size() > 0;

  EigenPack<FermionFieldD> epack;

  if (hasEigs) {
    // IRL action parameters
    auto &actionParIRL = inputParams.epack.action;

    // ========================================================================
    // MODULE: MSolver::StagFermionIRL (Run IRL eigensolver)
    // ========================================================================
    std::cout << GridLogMessage
              << "\n========================================" << std::endl;
    std::cout << GridLogMessage << "MODULE: MSolver::StagFermionIRL"
              << std::endl;
    std::cout << GridLogMessage
              << "========================================" << std::endl;

    auto &lanczosPar = inputParams.epack.irl.lanczosParams;
    const int Nstop = lanczosPar.Nstop;
    const int Nk = lanczosPar.Nk;
    const int Nm = lanczosPar.Nm;
    const int MaxIt = lanczosPar.MaxIt;
    RealD resid = lanczosPar.resid;

    std::cout << GridLogMessage << "IRL Parameters:" << std::endl;
    std::cout << GridLogMessage << "  Nstop = " << Nstop << std::endl;
    std::cout << GridLogMessage << "  Nk = " << Nk << std::endl;
    std::cout << GridLogMessage << "  Nm = " << Nm << std::endl;
    std::cout << GridLogMessage << "  MaxIt = " << MaxIt << std::endl;
    std::cout << GridLogMessage << "  resid = " << resid << std::endl;

    std::shared_ptr<FermionOpD> stagMatIRL;
    makeAction(stagMatIRL, actionParIRL);

    // Create operators for IRL if needed
    SchurStaggeredOperator<FermionOpD, FermionFieldD> hermOpIRL(*stagMatIRL);
    Chebyshev<FermionFieldD> Cheby(
        lanczosPar.Cheby.alpha, lanczosPar.Cheby.beta, lanczosPar.Cheby.Npoly);

    FunctionHermOp<FermionFieldD> OpCheby(Cheby, hermOpIRL);
    PlainHermOp<FermionFieldD> Op(hermOpIRL);

    ImplicitlyRestartedLanczos<FermionFieldD> IRL(OpCheby, Op, Nstop, Nk, Nm,
                                                  resid, MaxIt);

    FermionFieldD src(UrbGrid);
    int cb = inputParams.epack.irl.evenEigen ? Even : Odd;
    src.Checkerboard() = cb;

    std::cout << GridLogMessage << "Generating random source (checkerboard = "
              << (cb == Even ? "Even" : "Odd") << ")" << std::endl;
    gaussian(rng, src);

    epack.resize(inputParams.epack.size, UrbGrid);

    if (inputParams.epack.type == EpackPar::EpackType::solve) {
      std::cout << GridLogMessage << "Running IRL eigensolver..." << std::endl;
      int Nconv;
      epack.eval.resize(Nm);
      epack.evec.resize(Nm, UrbGrid);
      IRL.calc(epack.eval, epack.evec, src, Nconv);

      std::cout << GridLogMessage << "Converged " << Nconv << " eigenvectors"
                << std::endl;

      epack.eval.resize(Nstop);
      epack.evec.resize(Nstop, UGrid);
      epack.record.operatorXml = actionParIRL.parString();
      epack.record.solverXml = inputParams.epack.irl.parString();

      if (!inputParams.epack.file.empty()) {
        std::cout << GridLogMessage << "Saving eigenpack to "
                  << inputParams.epack.file << std::endl;
        epack.write(inputParams.epack.file, inputParams.epack.multiFile, traj);
      }
    }
    if (inputParams.epack.type == EpackPar::EpackType::load) {
      // Load eigenpack
      std::cout << GridLogMessage << "Loading eigenpack from "
                << inputParams.epack.file << std::endl;
      assert(!inputParams.epack.file.empty());
      epack.read(inputParams.epack.file, inputParams.epack.multiFile, traj);
      epack.eval.resize(inputParams.epack.size);
    }

    if (!inputParams.epack.evalSave.empty()) {
      std::cout << GridLogMessage << "Saving eigenvalues to "
                << inputParams.epack.evalSave << std::endl;
      saveResult(UGrid, inputParams.epack.evalSave, "evals", epack.eval,
                 inputParams);
    }

    std::cout << GridLogMessage << "Setting checkerboard of eigenvectors to "
              << (cb == Even ? "Even" : "Odd") << std::endl;
    for (auto &e : epack.evec) {
      e.Checkerboard() = cb;
    }
  }

  std::shared_ptr<FermionOpD> stagMatMassive;
  std::shared_ptr<FermionOpF> stagMatMassiveF;
  std::shared_ptr<FermionFieldD> fermOut;
  std::shared_ptr<FermionFieldD> fermIn;
  std::shared_ptr<FermionFieldD> fermGuess;

  RealD solverMass;
  // Create Action objects and temporary fields for solves
  if (hasSources) {
    fermOut = std::make_shared<FermionFieldD>(UGrid);
    fermIn = std::make_shared<FermionFieldD>(UGrid);
    fermGuess = std::make_shared<FermionFieldD>(UGrid);

    makeAction(stagMatMassive, inputParams.mpcg.action);
    makeAction(stagMatMassiveF, inputParams.mpcg.action);
    solverMass = 2.0 * inputParams.mpcg.action.mass;
  }

  using SolverFunc = std::function<void()>;
  SolverFunc lmaSolver, lmaSolverSubtract;
  SolverFunc mpcgSolver, mpcgSolverSubtract;

  // Create LMA Solver lambda functions
  if (hasEigs && hasSources) {

    // Extract LMA parameters
    unsigned int eigStart = 0; // Start from first eigenvalue
    int nEigs = -1;            // Use all eigenvalues by default
    bool projector = false;    // Use accelerated solver mode

    if (nEigs < 1) {
      nEigs = epack.evec.size();
    }

    // Validate eigenvalue ranges
    if (eigStart > static_cast<unsigned int>(nEigs) ||
        eigStart > epack.evec.size() ||
        nEigs - eigStart > static_cast<int>(epack.evec.size()) - eigStart) {
      std::cerr << "ERROR: Requested eigs (eigStart and nEigs) out of bounds"
                << std::endl;
      exit(1);
    }

    std::cout << GridLogMessage << "Setting up low mode projector" << std::endl;
    std::cout << GridLogMessage << "  eigStart = " << eigStart << std::endl;
    std::cout << GridLogMessage << "  nEigs = " << nEigs << std::endl;
    std::cout << GridLogMessage
              << "  projector = " << (projector ? "true" : "false")
              << std::endl;

    // Create temporary fields for LMA solver (heap allocated for use in
    // returned lambda)
    auto rbFerm = std::make_shared<FermionFieldD>(UrbGrid);
    auto rbFermNeg = std::make_shared<FermionFieldD>(UrbGrid);
    auto MrbFermNeg = std::make_shared<FermionFieldD>(UrbGrid);
    auto rbTemp = std::make_shared<FermionFieldD>(UrbGrid);
    auto rbTempNeg = std::make_shared<FermionFieldD>(UrbGrid);

    // Lambda to create the LMA solver function
    auto makeLMASolver = [&stagMatMassive, &epack, solverMass, projector,
                          eigStart, nEigs, rbFerm, rbFermNeg, MrbFermNeg,
                          rbTemp, rbTempNeg, fermOut, fermIn](bool subGuess) {
      return [&stagMatMassive, &epack, subGuess, solverMass, projector,
              eigStart, nEigs, rbFerm, rbFermNeg, MrbFermNeg, rbTemp, rbTempNeg,
              fermOut, fermIn]() {
        int cb = epack.evec[0].Checkerboard();
        int cbNeg = (cb == Even) ? Odd : Even;

        RealD norm = 1.0 / ::sqrt(norm2(epack.evec[0]));

        *rbTemp = Zero();
        rbTemp->Checkerboard() = cb;
        *rbTempNeg = Zero();
        rbTempNeg->Checkerboard() = cb;

        rbFerm->Checkerboard() = cb;
        rbFermNeg->Checkerboard() = cbNeg;
        MrbFermNeg->Checkerboard() = cb;

        // Extract checkerboard components
        pickCheckerboard(cb, *rbFerm, *fermIn);
        pickCheckerboard(cbNeg, *rbFermNeg, *fermIn);

        // Apply M_eooe^dagger
        stagMatMassive->MeooeDag(*rbFermNeg, *MrbFermNeg);

        // Project onto low modes
        for (int k = (eigStart + nEigs - 1); k >= static_cast<int>(eigStart);
             k--) {
          const FermionFieldD &e = epack.evec[k];

          const RealD lam_DD = epack.eval[k];
          const RealD invlam_DD = 1.0 / lam_DD;
          const RealD invmag = 1.0 / (solverMass * solverMass + lam_DD);

          if (!projector) {
            // Accelerated solver mode
            const ComplexD ip = TensorRemove(innerProduct(e, *rbFerm)) * invmag;
            const ComplexD ipNeg =
                TensorRemove(innerProduct(e, *MrbFermNeg)) * invmag;
            axpy(*rbTemp, solverMass * ip + ipNeg, e, *rbTemp);
            axpy(*rbTempNeg, solverMass * ipNeg * invlam_DD - ip, e,
                 *rbTempNeg);
          } else {
            // Pure projector mode
            const ComplexD ip = TensorRemove(innerProduct(e, *rbFerm));
            const ComplexD ipNeg = TensorRemove(innerProduct(e, *MrbFermNeg));
            axpy(*rbTemp, ip, e, *rbTemp);
            axpy(*rbTempNeg, ipNeg * invlam_DD, e, *rbTempNeg);
          }
        }

        // Apply M_eooe
        stagMatMassive->Meooe(*rbTempNeg, *rbFermNeg);

        // Reconstruct full field
        setCheckerboard(*fermOut, *rbTemp);
        setCheckerboard(*fermOut, *rbFermNeg);

        *fermOut *= norm;

        if (subGuess) {
          if (projector) {
            *fermOut = *fermIn - *fermOut;
          } else {
            std::cerr << "ERROR: Subtracted solver only supported for "
                         "projector=true"
                      << std::endl;
            exit(1);
          }
        }
      };
    };

    // Create the normal and subtract solvers
    lmaSolver = makeLMASolver(false);
    lmaSolverSubtract = makeLMASolver(true);

    std::cout << GridLogMessage << "Low mode projector setup complete"
              << std::endl;
  }

  // Create Mixed Precision CG Solver lambda functions
  if (hasSources) {

    auto &mpcgPar = inputParams.mpcg;

    std::cout << GridLogMessage << "Setting up mixed-precision CG solver"
              << std::endl;
    std::cout << GridLogMessage << "  Inner action (single precision): mass = "
              << mpcgPar.action.mass << std::endl;
    std::cout << GridLogMessage << "  Outer action (double precision): mass = "
              << mpcgPar.action.mass << std::endl;
    std::cout << GridLogMessage << "  Residual: " << mpcgPar.residual
              << std::endl;
    std::cout << GridLogMessage
              << "  Max inner iterations: " << mpcgPar.maxInnerIteration
              << std::endl;
    std::cout << GridLogMessage
              << "  Max outer iterations: " << mpcgPar.maxOuterIteration
              << std::endl;

    // Create hermitian operators for mixed precision solve
    auto hermOpOuter =
        std::make_shared<MdagMLinearOperator<FermionOpD, FermionFieldD>>(
            *stagMatMassive);
    auto hermOpInner =
        std::make_shared<MdagMLinearOperator<FermionOpF, FermionFieldF>>(
            *stagMatMassiveF);
    auto temp = std::make_shared<FermionFieldD>(UGrid);

    std::cout << GridLogMessage << "Mixed precision CG solver created"
              << std::endl;

    // Lambda to create MPCG solver functions
    auto makeMPCGSolver = [stagMatMassive, fermOut, fermIn, fermGuess, temp,
                           hermOpInner, hermOpOuter, &UGridF,
                           &mpcgPar](bool subGuess) {
      return [stagMatMassive, subGuess, fermOut, fermIn, fermGuess, temp,
              hermOpInner, hermOpOuter, &UGridF, &mpcgPar]() {
        MixedPrecisionConjugateGradient<FermionFieldD, FermionFieldF> mpcg(
            mpcgPar.residual, mpcgPar.maxInnerIteration,
            mpcgPar.maxOuterIteration, UGridF, *hermOpInner, *hermOpOuter);

        // Compute initial guess via outer guesser
        if (fermGuess != nullptr) {
          *fermOut = *fermGuess;
        } else {
          *fermOut = 1.0;
        }

        ZeroGuesser<FermionFieldF> iguesserDefault;
        mpcg.useGuesser(iguesserDefault);
        // Create temporary for residual
        *temp = Zero();
        stagMatMassive->Mdag(*fermIn, *temp);

        // Run MPCG solver on M^dag*M*x = M^dag*source
        mpcg(*temp, *fermOut);

        RealD nsol = norm2(*fermOut);
        // Compute residual: r = M*sol - source
        stagMatMassive->M(*fermOut, *temp);
        RealD nMsol = norm2(*temp);
        *temp = *temp - *fermIn;

        // Compute relative residual
        RealD ns = norm2(*fermIn);
        RealD nr = norm2(*temp);
        RealD relres = (ns > 0.0) ? std::sqrt(nr / ns) : 0.0;

        std::cout << GridLogMessage << "source magnitude: " << ns << std::endl;
        std::cout << GridLogMessage << "solution magnitude: " << nsol
                  << std::endl;
        std::cout << GridLogMessage << "M*solution magnitude: " << nMsol
                  << std::endl;
        std::cout << GridLogMessage << "MPCG: Final true residual = " << relres
                  << std::endl;

        if (subGuess && fermGuess != nullptr) {
          // For subtraction mode, compute residual vector
          *fermOut = *fermOut - *fermGuess;
        }
      };
    };

    // Create the MPCG solvers (normal and subtract)
    mpcgSolver = makeMPCGSolver(false);
    mpcgSolverSubtract = makeMPCGSolver(true);

    std::cout << GridLogMessage << "MPCG solvers created" << std::endl;
  }

  for (auto &sourcePar : inputParams.sources) {
    // Random wall source parameters (from XML) - Color-diagonal only
    unsigned int tStep = sourcePar.tStep;
    unsigned int t0 = sourcePar.t0;
    unsigned int nSrc = sourcePar.nSrc;

    std::cout << GridLogMessage
              << "Setting up random wall sources (color-diagonal)" << std::endl;
    std::cout << GridLogMessage << "  tStep = " << tStep << std::endl;
    std::cout << GridLogMessage << "  t0 = " << t0 << std::endl;
    std::cout << GridLogMessage << "  nSrc = " << nSrc << std::endl;

    // Get lattice dimensions
    int nt = UGrid->GlobalDimensions()[Tp];

    if (t0 >= tStep) {
      std::cerr << "ERROR: t0 >= tStep" << std::endl;
      exit(1);
    }
    TimeDilutedNoiseMILC<FImpl> noise(UGrid, nSrc);
    std::string seed =
        sourcePar.seed + "-" + std::to_string(inputParams.trajectory);
    std::cout << GridLogMessage << "Seeding source with seed '" << seed << "'"
              << std::endl;
    rng.SeedUniqueString(seed);
    noise.generateNoise(rng);
    int nSlices = nt / std::min(static_cast<int>(tStep), nt);
    int nVecs = nSrc * nSlices;

    std::cout << GridLogMessage << "  Number of time slices: " << nSlices
              << std::endl;
    std::cout << GridLogMessage << "  Total number of sources: " << nVecs
              << std::endl;

    auto &corrPar = inputParams.corr;

    auto quarkGammaKeys = StagGamma::ParseSpinTaste(corrPar.quark.gammas);
    auto quarkGammaVals =
        StagGamma::ParseSpinTaste(corrPar.quark.gammas, corrPar.quark.applyG5);
    GRID_ASSERT(!quarkGammaKeys.empty());

    auto antiquarkGammaKeys =
        StagGamma::ParseSpinTaste(corrPar.antiquark.gammas);
    auto antiquarkGammaVals = StagGamma::ParseSpinTaste(
        corrPar.antiquark.gammas, corrPar.antiquark.applyG5);
    GRID_ASSERT(antiquarkGammaKeys.size() == 1);
    std::string antiquarkGammaName = StagGamma::GetName(antiquarkGammaKeys[0]);
    StagGamma::SpinTastePair antiquarkSpinTaste = antiquarkGammaVals[0];

    auto sinkGammaKeys = StagGamma::ParseSpinTaste(corrPar.sink.gammas);
    auto sinkGammaVals =
        StagGamma::ParseSpinTaste(corrPar.sink.gammas, corrPar.sink.applyG5);
    GRID_ASSERT(sinkGammaKeys.size() == quarkGammaKeys.size());

    std::map<std::string, StagGamma::SpinTastePair> solveGammas;
    for (size_t i = 0; i < quarkGammaKeys.size(); ++i)
      solveGammas.emplace(StagGamma::GetName(quarkGammaKeys[i]),
                          quarkGammaVals[i]);
    if (!solveGammas.emplace(antiquarkGammaName, antiquarkSpinTaste).second)
      std::cout << GridLogMessage << "Warning: antiquark gamma '"
                << antiquarkGammaName
                << "' matches a quark gamma; skipping duplicate solve."
                << std::endl;

    // Initialize meson results for all gamma pairs
    std::cout << GridLogMessage << "Setting up meson contraction" << std::endl;
    std::vector<MesonResult> mesonResults(quarkGammaKeys.size());
    for (size_t i = 0; i < quarkGammaKeys.size(); ++i) {
      std::string quarkGammaName = StagGamma::GetName(quarkGammaKeys[i]);
      std::string sinkGammaName = StagGamma::GetName(sinkGammaKeys[i]);

      mesonResults[i].sourceGamma = quarkGammaName;
      mesonResults[i].sinkGamma = sinkGammaName;
      mesonResults[i].corr.resize(nt, 0.0);
      mesonResults[i].srcCorrs.resize(nVecs, std::vector<Complex>(nt, 0.0));
      mesonResults[i].scaling = nVecs;
    }

    // Color-diagonal noise: use PropagatorField (3x3 color matrix per site)
    PropagatorFieldD randomWallSource(UGrid);
    std::cout << GridLogMessage
              << "Generating color-diagonal random wall sources" << std::endl;
    int sourceIndex = 0;
    for (int i = 0; i < nSrc; i++) {
      for (int j = 0; j < nSlices; j++) {
        int timeSlice = j * tStep + t0;
        int offset = i * nt + j * tStep + t0;

        randomWallSource = noise.getProp(offset);
        std::cout << GridLogMessage << "Random wall sources setup complete"
                  << std::endl;

        std::map<std::string, PropagatorFieldD> lmaProp;
        std::map<std::string, PropagatorFieldD> mpcgProp;
        for (const auto &pair : solveGammas) {
          lmaProp.emplace(pair.first, UGrid);
          mpcgProp.emplace(pair.first, UGrid);
        }
        for (auto &pair : lmaProp)
          pair.second = Zero();
        for (auto &pair : mpcgProp)
          pair.second = Zero();
        // Create StagGamma operator
        StagGamma gamma;

        PropagatorFieldD gammaProp(UGrid);
        *fermIn = Zero();
        *fermOut = Zero();
        *fermGuess = Zero();
        for (auto &solve_pair : solveGammas) {
          const std::string &gammaName = solve_pair.first;
          gamma.setSpinTaste(solve_pair.second);

          gammaProp = Zero();
          gamma(gammaProp, randomWallSource);

          if (hasEigs) {
            for (int c = 0; c < 3; c++) {
              PropToFerm<FImpl>(*fermIn, gammaProp, c);
              lmaSolver();
              FermToProp<FImpl>(lmaProp.at(gammaName), *fermOut, c);
            }
          }

          if (hasSources) {
            for (int c = 0; c < 3; c++) {
              PropToFerm<FImpl>(*fermIn, gammaProp, c);
              if (hasEigs)
                PropToFerm<FImpl>(*fermGuess, lmaProp.at(gammaName), c);
              mpcgSolver();
              FermToProp<FImpl>(mpcgProp.at(gammaName), *fermOut, c);
            }
          }
        }

        std::cout << GridLogMessage << "Gauge propagator solver complete"
                  << std::endl;

        // Accumulate meson contraction results for this source
        for (size_t i = 0; i < quarkGammaKeys.size(); ++i) {
          std::string quarkGammaName = StagGamma::GetName(quarkGammaKeys[i]);
          gamma.setSpinTaste(sinkGammaVals[i]);

          PropagatorFieldD prod(UGrid);
          gamma(gammaProp, mpcgProp.at(quarkGammaName));
          prod = mpcgProp.at(antiquarkGammaName) * adj(gammaProp);

          std::vector<TComplex> buf;
          LatticeComplexD slicedTrace = trace(prod);
          sliceSum(slicedTrace, buf, Tp);
          int sliceOffset = t0;
          for (int t = 0; t < nt; ++t) {
            Complex ct = TensorRemove(buf[sliceOffset]);
            mesonResults[i].srcCorrs[sourceIndex][t] = ct;
            sliceOffset = mod(sliceOffset + 1, nt);
          }
        }

        sourceIndex++;
      }
    }

    // Compute averaged correlators from all sources
    for (size_t i = 0; i < quarkGammaKeys.size(); ++i) {
      for (int t = 0; t < nt; ++t) {
        mesonResults[i].corr[t] = 0.0;
        for (int j = 0; j < nVecs; j++) {
          mesonResults[i].corr[t] += mesonResults[i].srcCorrs[j][t];
        }
        mesonResults[i].corr[t] /= mesonResults[i].scaling;
      }
    }

    saveResult(UGrid, inputParams.corr.output, "meson", mesonResults,
               inputParams, t0);
  }

  // TODO: Additional modules to implement:
  // ========================================================================
  // 1. MContraction::StagA2AMesonField (All-to-All Meson Field)
  //    - Build all-to-all meson fields using low modes
  //    - Parameters: action, lowModes, left, right, spinTaste, mom, output
  //
  // COMPLETED MODULES:
  // ✓ MIO::LoadIldg - Load gauge configurations (U, U_fat, U_long)
  // ✓ MUtilities::GaugeSinglePrecisionCast - Precision casting (U_fat_f,
  // U_long_f) ✓ MAction::ImprovedStaggeredMILC - IRL action (stagMatIRL,
  // only if epack.load=false) ✓ MAction::ImprovedStaggeredMILC - LMA action
  // (stagMatLMA for LMA and MPCG) ✓ MFermion::StagOperators - Create
  // operators (hermOpIRL, hermOpLMA) ✓ MSolver::StagFermionIRL - IRL
  // eigensolver ✓ MUtilities::ModifyEigenPackMILC - Set eigenvector
  // checkerboard ✓ MSolver::StagLMA - Low mode projector solver (lmaSolver,
  // lmaSolverSubtract) ✓ MSolver::StagMixedPrecisionCG - Mixed precision CG
  // (mpcgSolverFunc, mpcgSolverSubtract) ✓ MSource::StagRandomWall - Random
  // wall sources (randomWallSources) ✓ MFermion::StagGaugeProp - Gauge
  // propagators with LMA→MPCG workflow ✓ MContraction::StagMeson - Meson
  // correlator contractions (correlators map) ✓
  // MUtilities::EigenPackExtractEvals - Eigenvalue extraction (handled by
  // saveResult)
  //
  // XML PARAMETER STRUCTURE:
  // ========================================================================
  // <parameters>
  //   <gauge>...</gauge>
  //   <gaugeFat>...</gaugeFat>
  //   <gaugeLong>...</gaugeLong>
  //   <trajectory>...</trajectory>
  //   <epack>
  //     <action> (IRL fermion action)
  //       <mass>...</mass>
  //       <c1>...</c1>
  //       <c2>...</c2>
  //       <tad>...</tad>
  //     </action>
  //     <irl> (IRL solver parameters) </irl>
  //     <evalSave>...</evalSave>
  //     <load>...</load>
  //     <size>...</size>
  //     <file>...</file>
  //     <multiFile>...</multiFile>
  //   </epack>
  //   <lma>
  //     <action> (LMA fermion action for StagLMA and StagMixedPrecisionCG)
  //       <mass>...</mass>
  //       <c1>...</c1>
  //       <c2>...</c2>
  //       <tad>...</tad>
  //     </action>
  //     <projector>...</projector>
  //     <eigStart>...</eigStart>
  //     <nEigs>...</nEigs>
  //     <lowModes>...</lowModes>
  //   </lma>
  //   <mpcg>
  //     <innerAction> (single precision action)
  //       <mass>...</mass>
  //       ...
  //     </innerAction>
  //     <outerAction> (double precision action)
  //       <mass>...</mass>
  //       ...
  //     </outerAction>
  //     <maxInnerIteration>...</maxInnerIteration>
  //     <maxOuterIteration>...</maxOuterIteration>
  //     <residual>...</residual>
  //     <innerGuesser>...</innerGuesser>
  //     <outerGuesser>...</outerGuesser>
  //   </mpcg>
  //   <sources> (RandomWallMILCPar entries) </sources>
  // </parameters>

#if 0
    // ========================================================================
    // MODULE: MContraction::StagA2AMesonField (All-to-All Meson Field)
    // TODO: Finish A2AWorker integration and fix pre-existing module compilation errors
    // ========================================================================
    std::cout << GridLogMessage << "\n========================================" << std::endl;
    std::cout << GridLogMessage << "MODULE: MContraction::StagA2AMesonField" << std::endl;
    std::cout << GridLogMessage << "========================================" << std::endl;

    auto &a2aPar = inputParams.a2a;

    std::cout << GridLogMessage << "Setting up all-to-all meson field construction" << std::endl;
    std::cout << GridLogMessage << "  Low modes: " << a2aPar.lowModes << std::endl;
    std::cout << GridLogMessage << "  Block size: " << a2aPar.block << std::endl;
    std::cout << GridLogMessage << "  Output: " << a2aPar.output << std::endl;

    // Parse momenta for A2A (as doubles for momentum phases)
    std::vector<std::vector<double>> a2aMom;
    for (const auto &momStr : a2aPar.mom) {
      auto p = strToVec<double>(momStr);
      if (p.size() != Nd - 1) {
        std::cerr << "ERROR: Momentum has " << p.size() << " components instead of "
                  << Nd - 1 << std::endl;
        exit(1);
      }
      a2aMom.push_back(p);
    }
    std::cout << GridLogMessage << "Number of momenta: " << a2aMom.size() << std::endl;

    // Parse spin-taste gammas for A2A
    std::vector<StagGamma::SpinTastePair> a2aGammaList;
    std::vector<std::string> a2aGammaNames;
    if (!a2aPar.spinTaste.gammas.empty()) {
      a2aGammaList =
          StagGamma::ParseSpinTaste(a2aPar.spinTaste.gammas,
                                         a2aPar.spinTaste.applyG5);
      auto gammaKeys =
          StagGamma::ParseSpinTaste(a2aPar.spinTaste.gammas);
      for (auto &g : gammaKeys) {
        a2aGammaNames.push_back(StagGamma::GetName(g));
      }
    }

    std::cout << GridLogMessage << "Number of spin-taste combinations: " << a2aGammaList.size()
              << std::endl;

    int nModes = epack.evec.size();
    int nt_a2a = UGrid->GlobalDimensions()[Tp];
    int nBlock = a2aPar.block;
    int nMom = a2aMom.size();
    int nGamma = a2aGammaList.size();

    std::cout << GridLogMessage << "Creating A2A fields with " << nModes << " eigenvectors, "
              << nGamma << " gamma combinations, " << nMom << " momenta"
              << std::endl;

    // Instantiate A2AWorkerLocal with gammas and momenta
    A2AWorkerLocal<FImpl> a2aWorker(UGrid, a2aMom, a2aGammaList, Tp);

    std::cout << GridLogMessage << "A2A worker instantiated" << std::endl;

    // Create Eigen tensor for all-to-all data
    // Dimensions: [momentum, gamma, time, left_mode, right_mode]
    Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> a2aData(nMom, nGamma, nt_a2a, nModes, nModes);
    a2aData.setZero();

    std::cout << GridLogMessage << "Processing A2A contraction" << std::endl;

    // Process eigenvectors in blocks for memory efficiency
    for (int iBlock = 0; iBlock < nModes; iBlock += nBlock) {
      int iEnd = std::min(iBlock + nBlock, nModes);

      for (int jBlock = 0; jBlock < nModes; jBlock += nBlock) {
        int jEnd = std::min(jBlock + nBlock, nModes);

        std::cout << GridLogMessage << "  Block [" << iBlock << ".." << iEnd - 1 << ", " << jBlock
                  << ".." << jEnd - 1 << "]" << std::endl;

        // Create temporary result tensor for this block
        Eigen::Tensor<ComplexD, 5, Eigen::RowMajor> blockResult(
            nMom, nGamma, nt_a2a, iEnd - iBlock, jEnd - jBlock);
        blockResult.setZero();

        // Extract block of eigenvectors
        std::vector<FermionFieldD> leftBlock(iEnd - iBlock, FermionFieldD(UGrid));
        std::vector<FermionFieldD> rightBlock(jEnd - jBlock, FermionFieldD(UGrid));

        for (int i = iBlock; i < iEnd; ++i) {
          leftBlock[i - iBlock] = epack.evec[i];
        }
        for (int j = jBlock; j < jEnd; ++j) {
          rightBlock[j - jBlock] = epack.evec[j];
        }

        // Call A2A worker kernel
        // Note: This requires A2AWorker to support the tensor interface
        // For now, just indicate where the computation would happen
        std::cout << GridLogMessage << "    Computing " << nMom << " momenta x " << nGamma
                  << " gammas x " << nt_a2a << " time-slices x "
                  << (iEnd - iBlock) << " x " << (jEnd - jBlock) << " modes"
                  << std::endl;

        // In a full implementation:
        // a2aWorker.StagMesonField(blockResult, &leftBlock[0], nullptr,
        //                           &rightBlock[0], nullptr);
      }
    }

    // Save A2A meson fields to files
    std::cout << GridLogMessage << "Saving A2A meson fields" << std::endl;
    for (int momIdx = 0; momIdx < nMom; ++momIdx) {
      for (unsigned int gIdx = 0; gIdx < a2aGammaList.size(); ++gIdx) {
        std::stringstream ss;
        ss << a2aPar.output << "." << traj << "/" << a2aGammaNames[gIdx];
        for (const auto &p : a2aMom[momIdx]) {
          ss << "_" << std::fixed << std::setprecision(1) << p;
        }
        ss << ".h5";

        std::cout << GridLogMessage << "  Output file: " << ss.str() << std::endl;
        // In full implementation, write using HDF5
      }
    }

    std::cout << GridLogMessage << "All-to-all meson field construction complete" << std::endl;
#endif
#if 0
        }
      }
    }
  }
#endif

  std::cout << GridLogMessage
            << "\n========================================" << std::endl;
  std::cout << GridLogMessage << "12 OF 13 HADRONS MODULES COMPLETE"
            << std::endl;
  std::cout << GridLogMessage
            << "Remaining: MContraction::StagA2AMesonField (A2AWorker "
               "integration)"
            << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  Grid_finalize();
}
