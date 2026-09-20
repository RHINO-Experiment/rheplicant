"""The public surface, pinned name by name.

``__all__`` is the list a reader trusts and a consumer imports from, and until
now nothing said what was in it. ``test_published_contracts.py`` checks that a
listed name RESOLVES -- a real check, and a different one: it cannot see a name
arriving, and an accidental export looks exactly like a deliberate one to it.

So this file holds the six lists, sorted, and compares them. The point is the
DIFF. A name added, removed or renamed shows up here as a line in the review,
which is the moment to ask whether it was meant -- adding to ``__all__`` is a
promise, and removing from it breaks whoever believed the last one.

**Sorted here, not in the source.** The modules build their ``__all__`` in
groups that mean something (``radio`` appends per subsystem, ``inference`` by
layer), and flattening them into alphabetical order would lose that and make
the source diff of a real addition harder to read, not easier. Sorting happens
on the way into the comparison, so the snapshot is stable under a regrouping
that changes no names.

**What this does NOT pin** is anything about the objects. A name may keep its
spelling while its signature changes underneath, and that is what
``test_published_contracts.py`` and the per-module guards are for. This is the
census of names, and it is only the census.
"""

from __future__ import annotations

import importlib

import pytest

#: rheplicant.__all__, sorted. 25 names.
RHEPLICANT = (
    'AbstractOperator', 'AmbiguousNodeError', 'Assembly', 'AssemblyError',
    'At', 'BASIS_KINDS', 'Coordinates', 'DataIngestionError', 'DirtError',
    'Environment', 'FrozenMapping', 'LambdaOperator', 'Maturity',
    'MissingKeyError', 'Pipeline', 'PipelineError', 'SelectOperator',
    'SeparableBasis', 'SignalGraph', 'SnapshotOperator', 'State',
    'StateValidationError', 'SumOperator', '__version__', 'basis_matrix',
)

#: rheplicant.core.__all__, sorted. 31 names.
RHEPLICANT_CORE = (
    'AbstractOperator', 'AmbiguousNodeError', 'Assembly', 'AssemblyError',
    'At', 'BASIS_KINDS', 'Coordinates', 'DataIngestionError', 'DirtError',
    'Environment', 'FrozenMapping', 'LambdaOperator', 'Maturity',
    'MissingKeyError', 'NodeSpec', 'Pipeline', 'PipelineError',
    'RANDOMNESS', 'SelectOperator', 'SeparableBasis', 'SignalGraph',
    'SnapshotOperator', 'State', 'StateValidationError', 'SumOperator',
    'assemble', 'basis_matrix', 'get_graph', 'register_graph',
    'stages_requiring', 'walk_operators',
)

#: rheplicant.radio.__all__, sorted. 61 names.
RHEPLICANT_RADIO = (
    'ADCOperator', 'AbstractLinearFilter', 'AbstractSkyModel',
    'AbstractSkyProjector', 'AntennaLossOperator',
    'ApplyCalibrationOperator', 'AtmosphericEmissionOperator',
    'BackendOperator', 'BasisTemperatureOperator', 'BeamSpillOperator',
    'CWCalibrationOperator', 'CalLoadOperator', 'DriftScanProjector',
    'EMIOperator', 'FlaggingOperator', 'ForegroundOperator',
    'FourierBandFilter', 'GainOperator', 'GeneralPointingProjector',
    'GlobalSignalOperator', 'GroundPickupOperator', 'IonosphereOperator',
    'MapSky', 'MatrixProjector', 'MomentRFIFlaggingOperator',
    'NeuralOperator', 'NoiseOperator', 'NoiseWaveOperator', 'PROTECTED_KEY',
    'PointSourceOperator', 'PowerLawSkyModel', 'RADIO_GRAPH', 'RFIOperator',
    'RadiometerNoiseOperator', 'ReceiverOperator', 'RhinoObservation',
    'SiderealFilter', 'SkyOperator', 'SkySourceOperator', 'SkySpaceFilter',
    'Touchstone', 'UniformSkyModel', 'assemble', 'at_level',
    'cal_load_operators', 'capabilities', 'capability_classes',
    'cst_beam_maps', 'cst_frequency_table', 'horizon_truncated_beam',
    'interpolate_onto', 'lst_grid_deg', 'protect', 'read_cst_farfield',
    'read_rhino_observation', 'read_touchstone', 'reduce_protection',
    'rhino_to_state', 'unflag_protected', 'unit_mean_bandpass',
    'unit_mean_free',
)

#: rheplicant.inference.__all__, sorted. 106 names.
RHEPLICANT_INFERENCE = (
    'AdamCalibrator', 'AmbiguousFanWarning', 'BROADCAST', 'BayesMemory',
    'Bind', 'Block', 'COEFFICIENTS', 'CRITERION_SHIFT', 'ChainMemory',
    'CompressedLikelihood', 'DEFAULT_RANK_RTOL', 'DISTRIBUTE', 'Draws',
    'EpochResidual', 'Estimate', 'FIRST_ORDER_MAX_FRACTIONAL',
    'Factorization', 'FidelityReport', 'FlaggedNoise', 'GLSResult',
    'GaussianLikelihood', 'GradientCalibrator', 'HeldOut',
    'HomoscedasticNoise', 'HyperTransition', 'IdentifiabilityReport',
    'JeffreysPrior', 'LOG_DEFAULT_SCALES', 'Latent', 'Likelihood',
    'LinearBlock', 'LinearGaussianTransition', 'LinearityRefused',
    'LogSpaceUnavailable', 'MIN_DRAWS', 'MaskedGaussianLikelihood',
    'NeuralPosterior', 'NoiseModel', 'NoiseModelLikelihood',
    'ParameterSpace', 'ParameterSpaceError', 'PlanDiagnostics',
    'PlanResult', 'PriorSensitivityReport', 'QuadraticLikelihood',
    'REQUIRED_TERM_MEMBERS', 'RadiometerNoise', 'RawLikelihood',
    'ReducedBasis', 'ReducedBasisLikelihood', 'SamplingPlan', 'SqrtInfo',
    'as_noise_model', 'auto_blocks', 'basis_fidelity', 'build_forward_fn',
    'build_reduced_basis', 'chain_log_likelihood', 'chain_marginal',
    'check_linearity', 'check_log_linearity', 'check_observed_shape',
    'coherent_mode', 'compress', 'compress_linear',
    'compress_reduced_basis', 'condition_bound', 'condition_estimate',
    'epoch_residuals', 'fisher_information', 'gcr_sample', 'held_out_z',
    'identifiability', 'init_to_declared', 'inverse_variance',
    'iterative_gls', 'linear_operator', 'load_memory',
    'log_linear_operator', 'marginalise', 'marginalise_arrays',
    'mean_squared_error', 'numerical_rank', 'ornstein_uhlenbeck',
    'orthonormal_transform', 'orthonormalise', 'parameter_covariance',
    'predict_from_samples', 'prior_sensitivity', 'propagate_covariance',
    'push_forward', 'refuse_stochastic_stages', 'save_memory',
    'score_directions', 'select_greedy', 'select_svd', 'shrinkage_power',
    'shrinkage_report', 'simulate_pairs', 'smooth', 'split_rhat',
    'systematic_floor', 'to_log_space', 'to_numpyro_model',
    'train_posterior', 'wiener_solve',
)

#: rheplicant.config.__all__, sorted. 41 names.
RHEPLICANT_CONFIG = (
    'ACCEPTED_UNITS', 'BuiltResources', 'ConfigError', 'ConfigWarning',
    'ConfiguredRun', 'DERIVATIONS', 'FILE_FORMATS', 'FieldSpec', 'Finding',
    'Gate', 'InferenceBuild', 'RESOURCE_KINDS', 'Report',
    'ResolutionContext', 'ResolvedPath', 'ResolvedValue', 'RunResult',
    'SHAPE_SYMBOLS', 'ShapeScope', 'Unit', 'VALUE_FORMS', 'VALUE_MODIFIERS',
    'apply_variant', 'build_resources', 'canonical_unit', 'compile_path',
    'convert_to_canonical', 'deliver', 'field_specs', 'gates',
    'load_document', 'parse_path', 'preflight', 'recursive_update',
    'register_dimension', 'register_dimension_formula', 'resolve_extent',
    'resolve_path_on', 'resolve_value', 'run_document', 'run_forward',
)

#: rheplicant.gui.__all__, sorted. 74 names.
RHEPLICANT_GUI = (
    'AuditArtifact', 'AxisPreview', 'CatalogDrift', 'EditorSession',
    'EditorSnapshot', 'FormCatalog', 'FormProjection', 'FormRule',
    'ForwardCost', 'GraphCounts', 'GraphDiagram', 'JobProjection',
    'JobRecord', 'JobStore', 'LedgerFinding', 'NodeCard', 'NodeInstance',
    'OutputProductProjection', 'OutputReportProjection', 'OutputState',
    'OutputWorkflowProjection', 'PresetChange', 'PreviewClass',
    'PreviewProjection', 'ProjectedSection', 'ProjectedWidget',
    'RevisionConflict', 'SectionBadge', 'SectionMetadata', 'ShapePreview',
    'SourceRef', 'ValidationProjection', 'WidgetMetadata',
    'assert_catalog_closed', 'classify_output_state',
    'completed_output_summary', 'compose_node', 'compose_session_node',
    'edit_session_many_node', 'edit_session_node', 'execute_job',
    'forward_preview_document', 'load_session_file', 'load_session_yaml',
    'mark_saved', 'mark_validated', 'move_node_instance',
    'move_session_node_instance', 'new_session', 'output_summary_at_path',
    'place_node', 'place_session_node', 'project_forms',
    'project_output_workflow', 'project_previews', 'read_audit_artifact',
    'redo', 'replace_session_yaml', 'replace_yaml', 'run_forward_preview',
    'run_priced_validation', 'save_session_file', 'set_many_node',
    'set_node', 'set_output_product', 'set_output_report',
    'set_session_output_product', 'set_session_output_report',
    'set_session_snapshot_before', 'set_snapshot_before', 'snapshot',
    'undo', 'validate_document', 'widget_catalog',
)

SNAPSHOTS = {
    "rheplicant": RHEPLICANT,
    "rheplicant.core": RHEPLICANT_CORE,
    "rheplicant.radio": RHEPLICANT_RADIO,
    "rheplicant.inference": RHEPLICANT_INFERENCE,
    "rheplicant.config": RHEPLICANT_CONFIG,
    "rheplicant.gui": RHEPLICANT_GUI,
}


@pytest.mark.parametrize("namespace", sorted(SNAPSHOTS), ids=sorted(SNAPSHOTS))
def test_the_public_surface_is_what_the_snapshot_says(namespace):
    """Sorted equality, and the failure names both directions.

    A bare ``==`` would print two long tuples and leave the reader to diff
    them; what matters is which names moved and which way.
    """
    module = importlib.import_module(namespace)
    live = sorted(module.__all__)
    pinned = sorted(SNAPSHOTS[namespace])
    added = sorted(set(live) - set(pinned))
    removed = sorted(set(pinned) - set(live))
    assert live == pinned, (
        f"{namespace}.__all__ has changed.\n"
        f"  added:   {added}\n"
        f"  removed: {removed}\n"
        "If that was deliberate, update the snapshot in this file in the same "
        "commit -- adding a name is a promise, and removing one breaks whoever "
        "believed the last promise. If it was not, the export is accidental."
    )


@pytest.mark.parametrize("namespace", sorted(SNAPSHOTS), ids=sorted(SNAPSHOTS))
def test_no_namespace_lists_a_name_twice(namespace):
    """``__all__`` is built by appending groups, which is where duplicates come from.

    A duplicate is invisible to the comparison above -- both tuples sort the
    same -- and it is invisible to ``import *``. It shows up as a name
    documented twice in the API reference.
    """
    live = list(importlib.import_module(namespace).__all__)
    seen = [name for name in set(live) if live.count(name) > 1]
    assert not seen, f"{namespace}.__all__ lists {sorted(seen)} more than once"


@pytest.mark.parametrize("namespace", sorted(SNAPSHOTS), ids=sorted(SNAPSHOTS))
def test_every_pinned_name_resolves(namespace):
    """The snapshot cannot outlive what it names.

    Without this, a name deleted from the package AND from ``__all__`` in one
    commit, but left in the snapshot, would fail the comparison above with a
    confusing message about a removal that was intended. With it, a snapshot
    entry that no longer exists says so directly.
    """
    module = importlib.import_module(namespace)
    missing = [name for name in SNAPSHOTS[namespace] if not hasattr(module, name)]
    assert not missing, (
        f"the snapshot for {namespace} names {missing}, which the module does "
        "not provide"
    )
