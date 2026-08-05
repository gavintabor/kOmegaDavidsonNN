/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2023 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kOmegaDavidsonNN.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"
#include "wallFvPatch.H"
#include "wallDistAddressing.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void kOmegaDavidsonNN<BasicTurbulenceModel>::correctNut()
{
    this->nut_ = k_/omega_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void kOmegaDavidsonNN<BasicTurbulenceModel>::computeNNCoefficients()
{
    const volScalarField& y = wallDist::New(this->mesh_).y();

    const volScalarField ss(sqrt(2.0) * mag(symm(fvc::grad(this->U_))));
    const volScalarField tauTot((this->nu() + this->nut_) * ss);

    // -------------------------------------------------------------------------
    // Friction velocity for the NN input features (Davidson Eq. 17: x0, x1).
    //
    // Computed from the wall-adjacent cells (viscous-sublayer formula,
    // matching Davidson's reference implementation -- see
    // literature/pythons-rans-code-RANS-open/.../modify_case.py:
    //   ustar = (u2d[:,0]*viscos/yp)**0.5
    //   ustar = np.repeat(ustar[:,None], repeats=nj, axis=1)   # held constant
    //           # along the wall-normal direction, not recomputed per cell.
    //
    // Using a per-cell Cmu^0.25*sqrt(k) estimate instead (an earlier
    // approach here) couples the NN's own inputs to the k field it is
    // actively boosting, which drifts the normalisation away from what the
    // NN was trained on and was found to distort nu_t away from the
    // standard k-omega model outside the immediate near-wall region.
    //
    // Computed once per wall-normal column (constant in y, varying along
    // the wall) via wallDistAddressing's nearest-wall-face broadcast --
    // matching Davidson's per-column method exactly, rather than the single
    // domain-wide average this used to collapse to. That average was exact
    // for a homogeneous channel flow (channelFlow5200) and a reasonable
    // approximation for a slowly-growing boundary layer (flatPlate), but
    // was badly wrong for a wall passing through separation, recirculation,
    // reattachment and re-acceleration (periodicHill), pushing (x0, x1)
    // outside the trained range almost everywhere on that wall.
    // -------------------------------------------------------------------------
    volScalarField uTauField
    (
        IOobject
        (
            "kOmegaDavidsonNN:uTau",
            this->mesh_.time().timeName(),
            this->mesh_.thisDb(),
            IOobjectOption::NO_REGISTER
        ),
        this->mesh_,
        dimensionedScalar
        (
            dimVelocity,
            pow(Cmu_.value(), 0.25) * Foam::sqrt(gAverage(k_.primitiveField()))
        )
    );

    {
        const tmp<volScalarField> tnu = this->nu();
        const volScalarField& nu = tnu();

        volScalarField::Boundary& uTauBf = uTauField.boundaryFieldRef();

        forAll(this->mesh_.boundary(), patchi)
        {
            if (isA<wallFvPatch>(this->mesh_.boundary()[patchi]))
            {
                const fvPatch& p = this->mesh_.boundary()[patchi];
                const labelUList& faceCells = p.faceCells();
                const scalarField yWall(y.boundaryField()[patchi].patchInternalField());
                const vectorField UWall
                (
                    this->U_.boundaryField()[patchi].patchInternalField()
                );
                const scalarField nuWall(nu.boundaryField()[patchi].patchInternalField());

                scalarField uStarWall(faceCells.size());
                forAll(faceCells, facei)
                {
                    const scalar yw = max(yWall[facei], SMALL);
                    const scalar uw = mag(UWall[facei]);
                    uStarWall[facei] = Foam::sqrt(nuWall[facei]*uw/yw);
                }
                uTauBf[patchi] = uStarWall;
            }
        }

        // Broadcast each wall face's u* to every cell whose nearest wall
        // face is that one -- constant along the wall-normal column,
        // varying along the wall. Domains with no wall patches simply keep
        // every cell at the fallback default set above.
        const auto& wDist = wallDistAddressing::New(this->mesh_);
        wDist.map(uTauField, mapDistribute::transform());
    }

    // -------------------------------------------------------------------------
    // Clipping bounds (training data ranges)
    // -------------------------------------------------------------------------
    static const scalar voyMin = 1.883e-04;
    static const scalar voyMax = 3.679e-01;
    static const scalar uvMin  = 9.149e-02;
    static const scalar uvMax  = 9.950e-01;

    static const scalar sigmakMin  = 2.088e-03;
    static const scalar sigmakMax  = 1.847e+00;
    static const scalar CkMin      = 4.182e-04;
    static const scalar CkMax      = 1.000e+00;
    static const scalar Comega2Min = 1.971e-04;
    static const scalar Comega2Max = 7.500e-02;

    // -------------------------------------------------------------------------
    // prand_k network weights  →  sigmakNN   (2→10→10→1, ReLU, linear out)
    // -------------------------------------------------------------------------
    static const scalar sk_W1[10][2] =
    {
        { 0.84573334,  0.71675086}, { 0.30598933, -0.15374757},
        {-0.01062202, -0.4276313 }, {-0.3824671 , -0.9245036 },
        { 1.2481849 , -0.12163923}, { 0.07547101, -0.4652685 },
        { 0.75766486,  0.4380672 }, {-0.27057737, -0.31967464},
        {-0.44039032,  0.8418275 }, { 0.24243979, -0.67738307}
    };
    static const scalar sk_b1[10] =
    {
        -1.4722832 , -0.52008015, -0.53476745,  0.47544998,  0.11527929,
        -0.4076968 , -0.32854357,  0.09974384,  0.16558872, -0.10354772
    };
    static const scalar sk_W2[10][10] =
    {
        {-0.02149744,  0.23287492, -0.13618506, -0.192645  , -1.1495858 ,
          0.13889815, -0.55999213, -0.17327847,  0.5297393 , -0.1548153 },
        {-0.0286337 , -0.12169755,  0.14927661,  0.0751792 , -0.24786106,
          0.09324402, -0.43199304, -0.13053326, -0.09313165, -0.11620283},
        { 0.13326661,  0.02152391, -0.12624392,  0.10522312,  0.03179962,
          0.06272271, -0.16517013, -0.2785166 , -0.30229142,  0.250737  },
        { 0.51622456, -0.22447969,  0.19519451,  0.16535406, -0.11379611,
         -0.00536794,  0.3753603 ,  0.04931431, -0.12518048,  0.1416591 },
        {-0.27521715,  0.1836319 , -0.2344452 , -0.09755673,  0.02493668,
         -0.2673787 , -0.02570092, -0.04516621,  0.05903452, -0.26542327},
        { 0.27670395, -0.00626766,  0.29013163,  0.20745046,  0.18861842,
         -0.1905118 ,  0.21369375, -0.29337433, -0.21886474, -0.20482357},
        {-1.3629714 , -0.00798734,  0.20846705, -0.79892176,  0.28744307,
          0.02012251, -0.2881416 ,  0.30818254,  0.28893405,  0.1221489 },
        { 0.05828657, -0.21776566,  0.17297788, -0.2498747 , -0.24160889,
         -0.17431074,  0.18842086, -0.10950086,  0.14537056, -0.04841214},
        {-0.26397598,  0.24057534, -0.10995997, -0.18727595, -0.47234324,
         -0.31139007, -0.22392434,  0.08821702,  0.05741542, -0.19032577},
        { 0.32245833, -0.24188577, -0.15992098, -0.2472531 ,  0.16982083,
         -0.17523241,  0.00941061,  0.15298535,  0.21800682, -0.30430284}
    };
    static const scalar sk_b2[10] =
    {
         0.19627425,  0.6763832 , -0.21749318,  0.4943252 , -0.12401722,
         0.38866934,  0.20923899, -0.16567044,  0.20975457,  0.03469171
    };
    static const scalar sk_W3[10] =
    {
        -1.3251549 ,  0.7409264 ,  0.07024844,  0.63121665,  0.2323689 ,
         0.41286683, -1.3745543 ,  0.00994973, -0.43276906,  0.1472507
    };
    static const scalar sk_b3 = 0.91513103;

    // -------------------------------------------------------------------------
    // c_k network weights  →  CkNN
    // -------------------------------------------------------------------------
    static const scalar ck_W1[10][2] =
    {
        {-0.6706352 , -0.11565896}, { 0.08191139, -0.29265523},
        { 0.7194075 , -0.45442116}, { 0.10355792,  0.15975173},
        { 0.3345124 , -0.45836246}, {-0.14580584,  0.6666446 },
        {-0.19294487,  1.0030158 }, {-0.04875571,  0.43166134},
        {-0.18230611, -0.67036533}, {-0.13399394, -0.5175329 }
    };
    static const scalar ck_b1[10] =
    {
        -0.02854716, -0.06567212,  0.01560317, -0.69735104,  0.54883635,
         0.5221333 , -0.35360262, -0.43210363,  0.01414929, -0.6831468
    };
    static const scalar ck_W2[10][10] =
    {
        { 0.125272  ,  0.25092533,  0.276597  ,  0.18672176,  0.238216  ,
          0.08103832, -0.6904357 , -0.04154851,  0.22215566, -0.15291502},
        { 0.03483474, -0.31559518, -0.2894059 ,  0.2688756 , -0.20037095,
         -0.19373661, -0.04507084,  0.07235176,  0.1969779 ,  0.03715689},
        {-0.16980796, -0.00178221, -0.1804553 ,  0.3007165 , -0.5379675 ,
          0.34389973,  0.60373574, -0.17898515,  0.22570576,  0.2504251 },
        { 0.18793225,  0.06275942, -0.25753555,  0.31257105, -0.25562736,
         -0.16699792, -0.27974987,  0.31149948, -0.14857933, -0.25764546},
        {-0.19426894,  0.14390619, -0.06763518,  0.10293505, -0.08274508,
         -0.15527982,  0.3034187 ,  0.15530504, -0.25604692,  0.16808674},
        {-0.30942252,  0.12802982, -0.07983677,  0.01279171,  0.2601738 ,
          0.09602885,  0.22321604, -0.17011765,  0.06142471, -0.15742862},
        { 0.20460443, -0.05032362, -0.01530452,  0.14976811, -0.07467391,
          0.07019374, -0.22705245,  0.1749277 ,  0.18005736,  0.30938244},
        {-0.22235526, -0.31228143, -0.21157862, -0.21810286,  0.03713537,
         -0.2125085 ,  0.06637079, -0.29066867,  0.01055705, -0.1792025 },
        {-0.22521487, -0.31225184, -0.23799416, -0.03030545, -0.17002623,
         -0.17663479,  0.07101712,  0.2468436 ,  0.02512064, -0.08649548},
        {-0.01022392, -0.01680956,  0.19381377,  0.05422423,  0.06047041,
         -0.1441465 , -0.11931799,  0.16921456,  0.08405398,  0.23965436}
    };
    static const scalar ck_b2[10] =
    {
         0.15603745,  0.2284753 ,  0.06215555, -0.12819026,  0.21622367,
         0.00643995,  0.2842259 , -0.23632964, -0.11477045, -0.27203998
    };
    static const scalar ck_W3[10] =
    {
         0.68675494,  0.10051088, -0.7917539 ,  0.2861621 , -0.19273394,
        -0.12251709,  0.20660718, -0.20291503, -0.253113  , -0.2096754
    };
    static const scalar ck_b3 = 0.6932656;

    // -------------------------------------------------------------------------
    // c_omega_2 network weights  →  Comega2NN
    // -------------------------------------------------------------------------
    static const scalar co_W1[10][2] =
    {
        { 0.64225286,  0.6421793 }, {-0.2624265 , -0.541474  },
        {-0.26130334,  0.48005596}, { 0.07950017, -0.16250071},
        { 0.68885887,  0.5195316 }, { 0.6395563 , -0.3474187 },
        {-0.570702  , -0.26000914}, {-0.3375184 , -0.38430062},
        { 0.5593689 , -0.46433762}, { 0.3920139 , -0.6010107 }
    };
    static const scalar co_b1[10] =
    {
         0.13212705, -0.1233962 ,  0.45035556, -0.49396905, -0.2792228 ,
         0.49126515,  0.1398982 ,  0.31832808,  0.22840098, -0.48710874
    };
    static const scalar co_W2[10][10] =
    {
        { 4.34355922e-02, -1.73254088e-01, -1.15541860e-01,  1.02654614e-01,
         -2.41500750e-01, -7.52514824e-02,  7.65006104e-03,  7.73532838e-02,
          2.93647975e-01, -1.63473755e-01},
        { 5.43159200e-03,  2.41725370e-01,  2.65700459e-01, -3.01275216e-02,
          1.44886062e-01,  1.77900225e-01,  8.41812864e-02,  1.30336508e-01,
          1.18385606e-01, -2.55000800e-01},
        {-2.83855736e-01, -6.41989335e-02, -1.00109100e-01, -2.33556256e-01,
          2.40719080e-01, -4.45787385e-02, -3.00688982e-01, -9.53193232e-02,
          1.64467007e-01, -3.08365017e-01},
        { 1.33214116e-01, -1.77128658e-01, -3.21721464e-01, -1.33528158e-01,
         -1.81705326e-01,  9.25299674e-02, -8.50628763e-02, -2.16491818e-01,
          2.85836488e-01,  4.52365167e-02},
        { 1.23714708e-01, -1.45128593e-01,  1.83202460e-01, -1.64559513e-01,
         -1.44198015e-01, -9.90626514e-02,  2.89266855e-01,  3.44305150e-02,
         -1.55026555e-01, -2.50073615e-02},
        { 1.04679324e-01, -4.36686873e-02, -1.44178659e-01,  1.65821880e-01,
          2.53947258e-01,  2.19149530e-01,  1.20860368e-01,  6.40927255e-02,
          2.36142203e-02,  7.78831616e-02},
        {-1.76065728e-01, -1.95544913e-01,  1.20514013e-01, -2.71857083e-01,
          4.78894673e-02, -1.05473241e-02, -3.14956635e-01, -1.41534135e-01,
          2.74416715e-01, -3.05148542e-01},
        { 1.78533003e-01,  2.82409340e-01,  1.32515714e-01,  1.52094260e-01,
         -5.06751388e-02,  2.22569361e-01, -2.99837589e-01,  5.98013029e-02,
          1.92218766e-01, -1.77610323e-01},
        { 1.03784963e-01,  1.89790726e-01, -1.16089739e-01, -4.07170281e-02,
          2.34020650e-01, -2.58162975e-01, -2.00638533e-01, -1.52147532e-01,
         -1.10467978e-01,  1.20202474e-01},
        {-3.45364541e-01, -2.59202063e-01,  1.92471713e-01, -2.20736653e-01,
          2.86736995e-01, -3.10422108e-02, -1.95518523e-01, -1.84867502e-04,
         -1.78102300e-01, -1.85488552e-01}
    };
    static const scalar co_b2[10] =
    {
         0.31513286, -0.08149902, -0.26853368,  0.05588951, -0.14562656,
        -0.16596714,  0.01635724, -0.03219016, -0.23034962,  0.1153323
    };
    static const scalar co_W3[10] =
    {
         0.35092634,  0.00715044,  0.22688176, -0.23939902, -0.1556035 ,
         0.01669093, -0.16002087,  0.10286507, -0.22823867, -0.15349457
    };
    static const scalar co_b3 = -0.0470497;

    // -------------------------------------------------------------------------
    // Cell loop
    // -------------------------------------------------------------------------

    // EWMA coefficients (Davidson Eq. 18): <f>^n = a*<f>^(n-1) + (1-a)*f^n
    // The fields already hold <f>^(n-1) from the previous iteration.
    const scalar a = Foam::exp(-1.0/max(ewmaM_.value(), scalar(1)));
    const scalar b = 1.0 - a;

    const scalarField& yIn    = y.internalField();
    const scalarField& tauIn  = tauTot.internalField();
    const scalarField& nutIn  = this->nut_.internalField();
    const scalarField& uTauIn = uTauField.internalField();

    forAll(this->mesh_.cells(), cellI)
    {
        const scalar voy = nutIn[cellI]
            / max(yIn[cellI]*uTauIn[cellI], SMALL);
        const scalar uv  = tauIn[cellI]
            / max(sqr(uTauIn[cellI]), SMALL);

        // Clip inputs to training range then normalise to [0, 1]
        const scalar x0 =
            (min(max(voy, voyMin), voyMax) - voyMin) / (voyMax - voyMin);
        const scalar x1 =
            (min(max(uv,  uvMin),  uvMax)  - uvMin)  / (uvMax  - uvMin);

        scalar h1[10], h2[10];

        // -- prand_k: PySR algebraic expression (Davidson Eq. 22) or NN --
        if (usePySR_)
        {
            const scalar xv = min(max(voy, voyMin), voyMax);
            const scalar xu = min(max(uv,  uvMin),  uvMax);
            const scalar sigma_pySR =
                0.469*xv
              + (0.574 + 1.0/(49.3 + 1.0/max(xv*xu*xu - 0.362, SMALL)))
              / (xv + 0.246 + 0.0516*xu/max(xv, SMALL));
            sigmakNN_[cellI] =
                a*sigmakNN_[cellI] + b*min(max(sigma_pySR, sigmakMin), sigmakMax);
        }
        else
        {
            // -- prand_k NN forward pass --
            for (int i = 0; i < 10; ++i)
                h1[i] = max(sk_W1[i][0]*x0 + sk_W1[i][1]*x1 + sk_b1[i], 0.0);
            for (int i = 0; i < 10; ++i)
            {
                scalar s = sk_b2[i];
                for (int j = 0; j < 10; ++j) s += sk_W2[i][j]*h1[j];
                h2[i] = max(s, 0.0);
            }
            {
                scalar s = sk_b3;
                for (int j = 0; j < 10; ++j) s += sk_W3[j]*h2[j];
                sigmakNN_[cellI] =
                    a*sigmakNN_[cellI] + b*min(max(s, sigmakMin), sigmakMax);
            }
        }

        // -- c_k forward pass --
        for (int i = 0; i < 10; ++i)
            h1[i] = max(ck_W1[i][0]*x0 + ck_W1[i][1]*x1 + ck_b1[i], 0.0);
        for (int i = 0; i < 10; ++i)
        {
            scalar s = ck_b2[i];
            for (int j = 0; j < 10; ++j) s += ck_W2[i][j]*h1[j];
            h2[i] = max(s, 0.0);
        }
        {
            scalar s = ck_b3;
            for (int j = 0; j < 10; ++j) s += ck_W3[j]*h2[j];
            CkNN_[cellI] =
                a*CkNN_[cellI] + b*min(max(s, CkMin), CkMax);
        }

        // -- c_omega_2 forward pass --
        for (int i = 0; i < 10; ++i)
            h1[i] = max(co_W1[i][0]*x0 + co_W1[i][1]*x1 + co_b1[i], 0.0);
        for (int i = 0; i < 10; ++i)
        {
            scalar s = co_b2[i];
            for (int j = 0; j < 10; ++j) s += co_W2[i][j]*h1[j];
            h2[i] = max(s, 0.0);
        }
        {
            scalar s = co_b3;
            for (int j = 0; j < 10; ++j) s += co_W3[j]*h2[j];
            Comega2NN_[cellI] =
                a*Comega2NN_[cellI] + b*min(max(s, Comega2Min), Comega2Max);
        }
    }

    sigmakNN_.correctBoundaryConditions();
    CkNN_.correctBoundaryConditions();
    Comega2NN_.correctBoundaryConditions();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaDavidsonNN<BasicTurbulenceModel>::kOmegaDavidsonNN
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Cmu_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    beta_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta",
            this->coeffDict_,
            0.072
        )
    ),
    gamma_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma",
            this->coeffDict_,
            5.0/9.0
        )
    ),
    alphaK_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega",
            this->coeffDict_,
            0.5
        )
    ),
    ewmaM_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "ewmaM",
            this->coeffDict_,
            500
        )
    ),
    usePySR_(false),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    sigmakNN_
    (
        IOobject
        (
            "sigmakNN",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("sigmakNN", dimless, 2.0)
    ),
    CkNN_
    (
        IOobject
        (
            "CkNN",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("CkNN", dimless, 1.0)
    ),
    Comega2NN_
    (
        IOobject
        (
            "Comega2NN",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("Comega2NN", dimless, 0.075) // Davidson standard Cω2 = 3/40
    )
{
    this->coeffDict_.readIfPresent("usePySR", usePySR_);

    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kOmegaDavidsonNN<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        beta_.readIfPresent(this->coeffDict());
        gamma_.readIfPresent(this->coeffDict());
        alphaK_.readIfPresent(this->coeffDict());
        alphaOmega_.readIfPresent(this->coeffDict());
        ewmaM_.readIfPresent(this->coeffDict());
        this->coeffDict().readIfPresent("usePySR", usePySR_);

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void kOmegaDavidsonNN<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    const volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    computeNNCoefficients();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))().v()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField::Internal GbyNu
    (
        tgradU().v() && devTwoSymm(tgradU().v())
    );
    const volScalarField::Internal G(this->GName(), nut()*GbyNu);
    tgradU.clear();

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();
    // Push any changed cell values to coupled neighbours
    omega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();

    // Wall omega is set by kOmegaDavidsonNNWallFunction (Davidson Eq. 7)
    // via the patch BC — no cell-centre overwrite needed here.

    // Turbulence specific dissipation rate equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(), omega_)
     ==
        gamma_*alpha()*rho()*GbyNu
      - fvm::SuSp(((2.0/3.0)*gamma_)*alpha()*rho()*divU, omega_)
      - fvm::Sp(Comega2NN_()*alpha()*rho()*omega_(), omega_)
      + fvOptions(alpha, rho, omega_)
    );

    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);


    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*G
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(CkNN_()*Cmu_*alpha()*rho()*omega_(), k_)
      + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);

    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
