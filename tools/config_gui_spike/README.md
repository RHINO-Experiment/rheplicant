# Config editor stack spike

This directory preserves the Task 1 comparison from Config Plan 5. Both
candidates consume the same framework-free `rheplicant.gui` document engine,
the same 33-node SVG, and the same `gain` edit. YAML returned for the measured
interaction is byte-identical.

## Decision

**Selected: FastAPI + React/TypeScript.** Criterion 1 is lexicographically
prior to criterion 2; both candidates tie on criterion 1, so React's stronger
canvas score decides the stack. The Panel candidate remains here as executable
evidence and is not a second implementation of the document rules.

| Candidate | pytest without a browser | Canvas interactivity |
| --- | ---: | ---: |
| Panel 1.9 | 5/5 | 3/5 |
| FastAPI + React/TypeScript | 5/5 | 5/5 |

Criterion 1 evidence is in `tests/gui/`: document transitions, exact no-op
stability, refusals, adapter delegation, API translation, and byte-identical
candidate parity all run under pytest without a browser. The optional-stack
imports happen inside tests so portable collection counts do not depend on
which GUI extra is installed.

Both canvases passed real-browser click, hover, keyboard activation, forward
walk, settings edit, and YAML-mirror checks. Panel's `ReactiveHTML` candidate
synchronizes click and every hover through the server-side parameter model;
selection feedback therefore takes a websocket round trip, and richer direct
manipulation would grow a string-script/Python synchronization seam. React
keeps selection and hover as typed client-local view state, delegates through
stable `data-node-id` attributes, and memoizes the SVG canvas so those view
updates preserve DOM identity. That is the concrete 3/5 versus 5/5 difference.

## Reproduce

From the repository root:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m pytest tests/gui --no-cov
npm --prefix tools/config_gui_spike/react test -- --run
npm --prefix tools/config_gui_spike/react run build
```

Run the candidates with:

```bash
PYTHONPATH="$PWD/src" .venv/bin/panel serve tools/config_gui_spike/panel_app.py
PYTHONPATH="$PWD/src" .venv/bin/uvicorn tools.config_gui_spike.react_api:app
```

The FastAPI server accepts a built frontend directory through
`create_app(frontend_dir=...)`; the durable launcher and packaged assets are
Plan 5 Task 8 work.
