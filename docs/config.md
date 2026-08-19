# Configuration documents

Configuration is a declarative route from exact YAML bytes to a validated,
audited run. Start with the [configuration command line](config-cli.md), then
use these pages for the document itself:

- [values and physical units](config-values.md);
- [resources and captured inputs](config-resources.md);
- [observation and model sections](config-sections.md);
- [inference, likelihoods, and exits](config-inference.md);
- [validation passes and findings](config-validation.md).

The loader parses the base and every declared variant before execution. It
records origins through preset and variant overlays, parses every run kind's
options once, and completes all selected layers through post-flight before the
first executor is called. Checks that genuinely need a built model are marked
as deferred during parsing and resolved at their named later boundary; they are
not silently postponed to an executor.

The mapping API remains available for Python callers. The CLI adds exact-byte
source identity, runtime ordering, output security, resolved YAML, and
provenance around that same orchestration rather than implementing a second
configuration language.
