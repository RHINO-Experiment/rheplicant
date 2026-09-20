# Capabilities: what is real, what is a stand-in

[Stability and the public surface](stability.md) says what this package
promises and for how long. This page says which parts of it are finished.

The distinction matters more here than in most packages, because a placeholder
in a digital twin produces numbers. It does not raise, it does not warn at the
call site, and the array it returns has the right shape and a plausible
structure. Philosophy 6 — *interfaces first, physics second* — is what makes
that safe to ship: a placeholder's **contract** is real and tested, so a
document that places one is doing nothing wrong. What it must not do is read
the numbers as physics.

Three things read the table below, and all three are views of the same
registry rather than copies of it:

- check **A53** tells a document, once, which of its nodes are not maintained;
- `capabilities.json` in every published run records the level of every node
  that run placed, so an archived result says which of its physics was a
  stand-in without anyone reading prose;
- the GUI disables a reserved key and gives the registry's own reason.

```{include} _generated/capabilities.md
```

## Reading a level

A level describes the IMPLEMENTATION, never the interface. Philosophy 6 again:
real physics replaces bodies, never contracts. So a node at `placeholder`
today and `maintained` tomorrow keeps its shapes, its ordering, its PRNG
consumption and its name; what changes is whether the numbers mean anything.

That is also why nothing here is a promise about dates. A level is a statement
about the code as it stands, made by the code: it is read off the class, and a
node stops being reported in the same commit that raises it.
