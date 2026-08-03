/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef kOmegaDavidsonNNWallFunction_H
#define kOmegaDavidsonNNWallFunction_H

#include "omegaWallFunctionFvPatchScalarField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
              Class kOmegaDavidsonNNWallFunction Declaration
\*---------------------------------------------------------------------------*/

class kOmegaDavidsonNNWallFunction
:
    public omegaWallFunctionFvPatchScalarField
{

protected:

    // Protected Member Functions

        //- Calculate omega (and G) at wall-adjacent cells.
        //- If the Comega2NN field is present in the registry (i.e. the
        //- turbulence model is kOmegaDavidsonNN), uses Davidson (2026) Eq. 7:
        //-   omega_w = 6*nu / (Comega2NN * y^2)
        //- Otherwise falls back to the standard omegaWallFunction.
        virtual void calculate
        (
            const turbulenceModel& turbModel,
            const List<scalar>& cornerWeights,
            const fvPatch& patch,
            scalarField& G,
            scalarField& omega
        );


public:

    //- Runtime type information
    TypeName("kOmegaDavidsonNNWallFunction");


    // Constructors

        //- Construct from patch and internal field
        kOmegaDavidsonNNWallFunction
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        kOmegaDavidsonNNWallFunction
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping onto a new patch
        kOmegaDavidsonNNWallFunction
        (
            const kOmegaDavidsonNNWallFunction&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy setting internal field reference
        kOmegaDavidsonNNWallFunction
        (
            const kOmegaDavidsonNNWallFunction&,
            const DimensionedField<scalar, volMesh>&
        );

        //- No copy without an internal field
        kOmegaDavidsonNNWallFunction
        (
            const kOmegaDavidsonNNWallFunction&
        ) = delete;

        //- Clone with an internal field reference
        virtual tmp<fvPatchField<scalar>> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return fvPatchField<scalar>::Clone(*this, iF);
        }


    //- Destructor
    virtual ~kOmegaDavidsonNNWallFunction() = default;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
