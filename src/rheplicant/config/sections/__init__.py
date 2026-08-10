"""Section loaders: one module per top-level config section concern.

Each module turns already-parsed Python data into the package's own objects.
:mod:`rheplicant.config.document` owns the build ORDER and is the only module
that imports across sections; nothing here imports ``document`` back.
"""
