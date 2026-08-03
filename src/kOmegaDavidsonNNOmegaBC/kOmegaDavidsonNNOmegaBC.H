/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef kOmegaDavidsonNNOmegaBC_H
#define kOmegaDavidsonNNOmegaBC_H

#include "fixedValueFvPatchFields.H"
#include "turbulenceModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class kOmegaDavidsonNNOmegaBC Declaration
\*---------------------------------------------------------------------------*/

//- True Dirichlet omega wall BC implementing Davidson (2026) Eq. 7:
//-   omega_w = 6*nu / (Comega2NN * y^2)
//- Falls back to 6*nu/(beta1*y^2) if Comega2NN is not in the registry.
class kOmegaDavidsonNNOmegaBC
:
    public fixedValueFvPatchScalarField
{

public:

    //- Runtime type information
    TypeName("kOmegaDavidsonNNOmegaBC");


    // Constructors

        //- Construct from patch and internal field
        kOmegaDavidsonNNOmegaBC
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        kOmegaDavidsonNNOmegaBC
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping onto a new patch
        kOmegaDavidsonNNOmegaBC
        (
            const kOmegaDavidsonNNOmegaBC&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy setting internal field reference
        kOmegaDavidsonNNOmegaBC
        (
            const kOmegaDavidsonNNOmegaBC&,
            const DimensionedField<scalar, volMesh>&
        );

        //- No copy without an internal field
        kOmegaDavidsonNNOmegaBC
        (
            const kOmegaDavidsonNNOmegaBC&
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
    virtual ~kOmegaDavidsonNNOmegaBC() = default;


    // Member Functions

        //- Update the coefficients associated with the patch field.
        //- Sets omega_w = 6*nu/(Comega2NN*y^2) on each wall face.
        virtual void updateCoeffs();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
