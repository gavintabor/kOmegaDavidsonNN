/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "kOmegaDavidsonNNOmegaBC.H"
#include "turbulenceModel.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void kOmegaDavidsonNNOmegaBC::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const turbulenceModel& turbModel =
        db().lookupObject<turbulenceModel>(turbulenceModel::propertiesName);

    const label patchi = patch().index();
    const scalarField& y = turbModel.y()[patchi];

    const tmp<scalarField> tnuw = turbModel.nu(patchi);
    const scalarField& nuw = tnuw();

    const labelUList& faceCells = patch().faceCells();
    scalarField omegaw(patch().size());

    if (db().foundObject<volScalarField>("Comega2NN"))
    {
        // Davidson (2026) Eq. 7: omega_w = 6*nu / (Comega2NN * y^2)
        const volScalarField& Comega2NN =
            db().lookupObject<volScalarField>("Comega2NN");

        forAll(faceCells, facei)
        {
            const label celli = faceCells[facei];
            const scalar Cw2 = max(Comega2NN[celli], scalar(0.01));
            omegaw[facei] =
                6.0*nuw[facei]
              / max(Cw2*sqr(y[facei]), SMALL);
        }
    }
    else
    {
        // Fallback: standard viscous-sublayer formula with beta1 = 0.072
        const scalar beta1 = 0.072;
        forAll(faceCells, facei)
        {
            omegaw[facei] =
                6.0*nuw[facei]
              / max(beta1*sqr(y[facei]), SMALL);
        }
    }

    operator==(omegaw);
    fixedValueFvPatchScalarField::updateCoeffs();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kOmegaDavidsonNNOmegaBC::kOmegaDavidsonNNOmegaBC
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedValueFvPatchScalarField(p, iF)
{}


kOmegaDavidsonNNOmegaBC::kOmegaDavidsonNNOmegaBC
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchScalarField(p, iF, dict)
{}


kOmegaDavidsonNNOmegaBC::kOmegaDavidsonNNOmegaBC
(
    const kOmegaDavidsonNNOmegaBC& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchScalarField(ptf, p, iF, mapper)
{}


kOmegaDavidsonNNOmegaBC::kOmegaDavidsonNNOmegaBC
(
    const kOmegaDavidsonNNOmegaBC& bc,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedValueFvPatchScalarField(bc, iF)
{}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField
(
    fvPatchScalarField,
    kOmegaDavidsonNNOmegaBC
);

} // End namespace Foam

// ************************************************************************* //
