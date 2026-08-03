/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "kOmegaDavidsonNNWallFunction.H"
#include "turbulenceModel.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void kOmegaDavidsonNNWallFunction::calculate
(
    const turbulenceModel& turbModel,
    const List<scalar>& cornerWeights,
    const fvPatch& patch,
    scalarField& G0,
    scalarField& omega0
)
{
    const fvMesh& mesh = patch.boundaryMesh().mesh();

    if (!mesh.foundObject<volScalarField>("Comega2NN"))
    {
        // Model is not kOmegaDavidsonNN — fall back to standard behaviour
        omegaWallFunctionFvPatchScalarField::calculate
        (
            turbModel, cornerWeights, patch, G0, omega0
        );
        return;
    }

    const volScalarField& Comega2NN =
        mesh.lookupObject<volScalarField>("Comega2NN");

    const label patchi = patch.index();
    const scalarField& y = turbModel.y()[patchi];
    const labelUList& faceCells = patch.faceCells();

    const tmp<scalarField> tnuw = turbModel.nu(patchi);
    const scalarField& nuw = tnuw();

    // Davidson (2026) Eq. 7: omega_w = 6*nu / (Comega2NN * y^2)
    // Purely viscous-sublayer formula — appropriate for y+ ~ O(1).
    // No G contribution is added (consistent with viscous-sublayer treatment
    // where y+ << yPlusLam and the log-law production correction does not apply).
    forAll(faceCells, facei)
    {
        const label celli = faceCells[facei];
        const scalar Cw2 = max(Comega2NN[celli], scalar(0.01));
        omega0[celli] +=
            cornerWeights[facei]*6.0*nuw[facei]
          / max(Cw2*sqr(y[facei]), SMALL);
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kOmegaDavidsonNNWallFunction::kOmegaDavidsonNNWallFunction
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    omegaWallFunctionFvPatchScalarField(p, iF)
{}


kOmegaDavidsonNNWallFunction::kOmegaDavidsonNNWallFunction
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    omegaWallFunctionFvPatchScalarField(p, iF, dict)
{}


kOmegaDavidsonNNWallFunction::kOmegaDavidsonNNWallFunction
(
    const kOmegaDavidsonNNWallFunction& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    omegaWallFunctionFvPatchScalarField(ptf, p, iF, mapper)
{}


kOmegaDavidsonNNWallFunction::kOmegaDavidsonNNWallFunction
(
    const kOmegaDavidsonNNWallFunction& wf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    omegaWallFunctionFvPatchScalarField(wf, iF)
{}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField
(
    fvPatchScalarField,
    kOmegaDavidsonNNWallFunction
);

} // End namespace Foam

// ************************************************************************* //
