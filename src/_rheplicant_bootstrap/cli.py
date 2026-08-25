"""Argument and stream boundary for the Config Plan 4A commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import BinaryIO, TextIO

from _rheplicant_bootstrap.entry import _render_exception, dispatch_request
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output.manager import parse_output_grammar, resolve_output_request
from _rheplicant_bootstrap.prepare import prepare_config
from _rheplicant_bootstrap.presets import read_installed_preset
from _rheplicant_bootstrap.script import publish_script, render_script
from _rheplicant_bootstrap.source import read_source


class _Parser(argparse.ArgumentParser):
    """An argparse parser that refuses through this package's error type.

    The override exists so a usage mistake arrives as a ``ConfigError`` the
    caller can catch, rather than as argparse's own ``SystemExit`` -- which
    matters because ``main()`` is a library entry point with a documented
    ``int`` return, and a process-exiting parser would make that signature a
    fiction for every argument mistake.

    It used to raise ``f"usage: {message}"``, which promised a usage line and
    then did not print one: the reader got ``usage: unrecognized arguments:
    --zzz`` and no indication of what the usage IS. ``format_usage()`` is
    what argparse would have written before exiting, so including it makes
    the word true. The message still begins ``usage: `` -- that is the first
    token of ``format_usage()`` -- so nothing matching on the prefix moves.

    ``--help`` is deliberately left alone. It still raises ``SystemExit(0)``
    after printing, which is what a CLI should do and what ``main`` reports
    as its own exit code; the review paired the two, but only the error path
    was discarding anything.
    """

    def error(self, message: str) -> None:
        raise ConfigError(f"{self.format_usage().strip()}\n{message}")


def _parser() -> _Parser:
    parser = _Parser(prog="rheplicant", add_help=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "script"):
        command = commands.add_parser(name)
        command.add_argument("config", metavar="CONFIG")
        command.add_argument("--base-dir", dest="base_dir")
        if name == "script":
            command.add_argument("-o", "--output", dest="output")
    return parser


def _binary_stdout(stream: TextIO, explicit: BinaryIO | None) -> BinaryIO:
    if explicit is not None:
        return explicit
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        raise ConfigError("script stdout does not expose a binary stream.")
    return buffer


def _main(
    argv: Sequence[str] | None = None,
    *,
    stdin: BinaryIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdout_bytes: BinaryIO | None = None,
) -> int:
    """Parse one invocation and return its stable process status."""
    chosen_stdout = sys.stdout if stdout is None else stdout
    chosen_stderr = sys.stderr if stderr is None else stderr
    try:
        arguments = _parser().parse_args(None if argv is None else tuple(argv))
        if arguments.config == "-" and arguments.base_dir is None:
            raise ConfigError("CONFIG '-' requires --base-dir DIR.")
        source = read_source(
            arguments.config,
            base_dir=arguments.base_dir,
            stdin=stdin,
        )
        if arguments.command in ("validate", "run"):
            return dispatch_request(
                arguments.command,
                source,
                stdout=chosen_stdout,
                stderr=chosen_stderr,
            )

        prepared = prepare_config(
            source,
            preset_provider=read_installed_preset,
            parse_outputs=parse_output_grammar,
        )
        request = resolve_output_request(
            prepared.process.outputs,
            source=source,
            command="script",
        )
        if source.source_path == "<stdin>" and request.target_path is None:
            raise ConfigError(
                "outputs.dir: script from stdin requires an explicit directory."
            )
        payload = render_script(
            source,
            prepared.source.bootstrap_manifest.presets,
        )
        if arguments.output is None:
            stream = _binary_stdout(chosen_stdout, stdout_bytes)
            stream.write(payload)
            stream.flush()
        else:
            publish_script(payload, arguments.output)
        return 0
    except ConfigError as error:
        return _render_exception(error, chosen_stderr, traceback_error=False)
    except Exception as error:
        return _render_exception(error, chosen_stderr, traceback_error=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run against the process streams with the stable public signature."""
    return _main(argv)


__all__ = ["main"]
