/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

Class
    Foam::RASModels::kOmegaDavidsonNN

Group
    grpRASTurbulence

Description
    k-omega turbulence model with PINN/NN corrections, derived from the
    standard Wilcox k-omega model.

    Based on:
    \verbatim
        Davidson, L. (2026).
        Using physics informed neural network (PINN) and neural network (NN)
        to improve a k-omega turbulence model.
        Journal of Turbulence.
        DOI:10.1080/14685248.2026.2665148
    \endverbatim

    Parent model:
    \verbatim
        Wilcox, D. C. (1988).
        Reassessment of the scale-determining equation for advanced
        turbulence models.
        AIAA Journal, 26(11), 1299-1310.
        DOI:10.2514/3.10041
    \endverbatim

    The default model coefficients are
    \verbatim
        kOmegaDavidsonNNCoeffs
        {
            betaStar    0.09;
            gamma       0.5556;  // 5/9
            alphaOmega  0.5;
            ewmaM       500;   // EWMA window (Davidson Eq. 18): a = exp(-1/m)
        }
    \endverbatim

SourceFiles
    kOmegaDavidsonNN.C

\*---------------------------------------------------------------------------*/

#ifndef kOmegaDavidsonNN_H
#define kOmegaDavidsonNN_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class kOmegaDavidsonNN Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kOmegaDavidsonNN
:
    public eddyViscosity<RASModel<BasicTurbulenceModel>>
{

protected:

    // Protected data

        // Model coefficients

            dimensionedScalar Cmu_;
            dimensionedScalar gamma_;
            dimensionedScalar alphaOmega_;
            dimensionedScalar ewmaM_;   // EWMA window m; a = exp(-1/m)
            bool usePySR_;              // use PySR expression for sigmakNN (Davidson Eq. 22)


        // Fields

            volScalarField k_;
            volScalarField omega_;

        // NN correction fields (uniform, updated by network each iteration)

            volScalarField sigmakNN_;    // turbulent Prandtl number for k (std: 2.0)
            volScalarField CkNN_;        // k production multiplier        (std: 1.0)
            volScalarField Comega2NN_;   // omega destruction coefficient   (std: 0.075 = 3/40)


    // Protected Member Functions

        virtual void correctNut();

        virtual void computeNNCoefficients();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("kOmegaDavidsonNN");


    // Constructors

        //- Construct from components
        kOmegaDavidsonNN
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~kOmegaDavidsonNN() = default;


    // Member Functions

        //- Read RASProperties dictionary
        virtual bool read();

        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff() const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    "DkEff",
                    this->nut_/sigmakNN_ + this->nu()
                )
            );
        }

        //- Return the effective diffusivity for omega
        tmp<volScalarField> DomegaEff() const
        {
            const dimensionedScalar sigmaOmega
            (
                "sigmaOmega", dimless, scalar(1)/alphaOmega_.value()
            );
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    "DomegaEff",
                    this->nut_/min(scalar(2)*sigmakNN_, sigmaOmega)
                  + this->nu()
                )
            );
        }

        //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the turbulence specific dissipation rate
        virtual tmp<volScalarField> omega() const
        {
            return omega_;
        }

        //- Return the NN omega destruction coefficient field (Davidson Eq. 7)
        const volScalarField& Comega2NN() const
        {
            return Comega2NN_;
        }

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#ifdef NoRepository
    #include "kOmegaDavidsonNN.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
