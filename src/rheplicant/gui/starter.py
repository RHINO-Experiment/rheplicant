"""The canonical, server-owned starting document for the GUI workbench."""

STARTER_YAML = """\
schema_version: 1
runtime:
  seed: 20260820
observation:
  meta:
    telescope: RHINO
  freq:
    grid:
      linspace:
        start: 60.0
        stop: 85.0
        num: 8
        endpoint: true
      unit: MHz
  time:
    grid:
      arange:
        start: 0.0
        step: 2.0
        num: 16
      unit: s
  environment:
    temperature: {value: 280.0, unit: K}
resources:
  arrays:
    flat:
      ones: [n_freq]
model:
  global_signal:
    depth: {value: 0.5, unit: K}
    centre: {value: 75.0, unit: MHz}
    width: {value: 5.0, unit: MHz}
  gain:
    gain: {value: 1.1, unit: dimensionless}
  noise:
    type: NoiseOperator
    sigma: {value: 0.05, unit: K}
runs:
  - name: forward
    kind: forward
"""
