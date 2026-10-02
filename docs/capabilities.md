# Capabilities: what is real, what is a stand-in

[Stability and capabilities](stability.md) says what this package
promises and for how long. This page says which parts of it are finished.

A placeholder in a digital twin produces numbers. It does not raise, it does
not warn at the call site, and the array it returns has the right shape and a
plausible structure. Under Philosophy 6 (*interfaces first, physics second*)
a placeholder's contract is real and tested, so a document that places one is
doing nothing wrong. It must not read the numbers as physics.

Three things read the table below, and all three are views of the same
registry:

- check **A53** tells a document, once, which of its nodes are not maintained;
- `capabilities.json` in every published run records the level of every node
  that run placed, so an archived result says which of its physics was a
  stand-in;
- the GUI disables a reserved key and gives the registry's own reason.

```{include} _generated/capabilities.md
```

## Reading a level

A level describes the implementation, not the interface. By Philosophy 6,
real physics replaces bodies, never contracts, so a node at `placeholder`
today and `maintained` tomorrow keeps its shapes, its ordering, its PRNG
consumption and its name; what changes is whether the numbers mean anything.

Nothing here is a promise about dates. A level is a statement about the code
as it stands: it is read off the class, and a node stops being reported in
the same commit that raises it.
