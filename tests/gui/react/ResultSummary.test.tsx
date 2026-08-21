import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ResultSummary } from "../../../src/rheplicant/gui/react/ResultSummary";
import type { JobProjection } from "../../../src/rheplicant/gui/react/types";

afterEach(cleanup);

const TRUNCATION_MARKER = "…[truncated]";
// The exact contract the DOM must honour, pinned rather than bounded by a loose inequality: a
// value over the cap is cut so that the marker fits INSIDE the cap, never appended past it.
const MAX_RENDERED_CHARACTERS = 4_000;
const MAX_RENDERED_FINDINGS = 100;

function truncatedTo(character: string) {
  return `${character.repeat(MAX_RENDERED_CHARACTERS - TRUNCATION_MARKER.length)}${TRUNCATION_MARKER}`;
}

function job(overrides: Partial<JobProjection> = {}): JobProjection {
  return {
    job_id: "job-1",
    session_id: "session-1",
    kind: "validate",
    revision: 3,
    yaml_digest: "digest-3",
    status: "refused",
    result: null,
    message: null,
    stale: false,
    ...overrides,
  };
}

function finding(index: number, message: string) {
  return {
    check: `C${index}`,
    severity: "report",
    where: "document",
    message,
    layer: "base",
  };
}

function summary() {
  return screen.getByRole("region", { name: "Result summary" });
}

describe("the rendered result is bounded even when the payload is not", () => {
  it("caps a megabyte message in the DOM", () => {
    render(<ResultSummary job={job({ message: "m".repeat(1_000_000) })} />);

    const text = summary().textContent ?? "";
    // Pinned to the cap itself: a threshold raised to anything above 4000 leaves this megabyte
    // message far longer than the exact string the DOM is required to hold.
    expect(screen.getByText(truncatedTo("m")).textContent).toHaveLength(MAX_RENDERED_CHARACTERS);
    expect(text.length).toBeLessThan(MAX_RENDERED_CHARACTERS + 200);
    expect(text).toContain(TRUNCATION_MARKER);
  });

  it("renders a bounded number of findings and names the rest", () => {
    const findings = Array.from({ length: 5_000 }, (_, index) => (
      finding(index, "long ".repeat(20_000))
    ));

    render(<ResultSummary job={job({ result: { findings } })} />);

    const items = within(summary()).getAllByRole("listitem");
    // Pinned to the cap itself: a threshold lowered to 50 renders half this list and still
    // satisfies "at most 100".
    expect(items).toHaveLength(MAX_RENDERED_FINDINGS);
    for (const item of items) {
      expect((item.textContent ?? "").length).toBeLessThan(20_000);
    }
    expect(
      within(summary()).getByText(/further findings were not rendered/),
    ).toBeInTheDocument();
    expect(summary().textContent).toContain(TRUNCATION_MARKER);
  });

  it("leaves a legal message and finding list untouched", () => {
    render(
      <ResultSummary
        job={job({
          message: "priced refusal",
          result: { findings: [finding(13, "check the declared gain")] },
        })}
      />,
    );

    const text = summary().textContent ?? "";
    expect(text).toContain("priced refusal");
    expect(text).toContain("check the declared gain");
    expect(text).not.toContain(TRUNCATION_MARKER);
    expect(within(summary()).getAllByRole("listitem")).toHaveLength(1);
  });

  it("truncates one character over the cap and leaves the cap itself untouched", () => {
    // Both sides of the boundary, so neither raising nor lowering MAX_RENDERED_CHARACTERS survives.
    const view = render(<ResultSummary job={job({ message: "m".repeat(MAX_RENDERED_CHARACTERS) })} />);
    expect(screen.getByText("m".repeat(MAX_RENDERED_CHARACTERS)).textContent)
      .toHaveLength(MAX_RENDERED_CHARACTERS);
    expect(summary().textContent).not.toContain(TRUNCATION_MARKER);
    view.unmount();

    render(<ResultSummary job={job({ message: "m".repeat(MAX_RENDERED_CHARACTERS + 1) })} />);
    // The marker is accounted for INSIDE the cap: 3987 characters plus the 13-character marker
    // (U+2026 plus the twelve of "[truncated]" and its brackets), which is 4000 exactly.
    expect(screen.getByText(truncatedTo("m")).textContent).toHaveLength(MAX_RENDERED_CHARACTERS);
  });

  it("renders the hundredth finding and names the hundred-and-first", () => {
    // Both sides of the boundary, so neither raising nor lowering MAX_RENDERED_FINDINGS survives.
    const rows = (count: number) => Array.from({ length: count }, (_, index) => (
      finding(index, "legible")
    ));
    const view = render(<ResultSummary job={job({ result: { findings: rows(MAX_RENDERED_FINDINGS) } })} />);
    expect(within(summary()).getAllByRole("listitem")).toHaveLength(MAX_RENDERED_FINDINGS);
    expect(within(summary()).queryByText(/further findings were not rendered/)).toBeNull();
    view.unmount();

    render(<ResultSummary job={job({ result: { findings: rows(MAX_RENDERED_FINDINGS + 1) } })} />);
    expect(within(summary()).getAllByRole("listitem")).toHaveLength(MAX_RENDERED_FINDINGS);
    expect(within(summary()).getByText(`1 further findings were not rendered ${TRUNCATION_MARKER}`))
      .toBeVisible();
  });

  it("bounds every rendered finding field, not the message alone", () => {
    // Kills bounding `message` only. check, where and layer are server strings on the same
    // surface, and every long fixture in this file used to be a message.
    render(
      <ResultSummary
        job={job({
          result: {
            findings: [{
              check: "c".repeat(1_000_000),
              severity: "report",
              where: "w".repeat(1_000_000),
              message: "m".repeat(1_000_000),
              layer: "l".repeat(1_000_000),
            }],
          },
        })}
      />,
    );

    const rendered = within(summary()).getAllByRole("listitem")[0].textContent ?? "";
    for (const character of ["c", "w", "m", "l"]) {
      expect(rendered).toContain(truncatedTo(character));
    }
    expect(rendered.length).toBeLessThan(4 * MAX_RENDERED_CHARACTERS + 200);
  });

  it("bounds the published output summary the server fills", () => {
    // Kills rendering the output path raw, which put 2,000,140 characters into the DOM.
    render(
      <ResultSummary
        job={job({
          status: "succeeded",
          result: {
            output: {
              state: "ready_new",
              state_message: "s".repeat(1_000_000),
              target_path: "t".repeat(1_000_000),
            },
          },
        })}
      />,
    );

    const rendered = screen.getByRole("region", { name: "Published output summary" }).textContent ?? "";
    expect(rendered).toContain(truncatedTo("s"));
    expect(rendered).toContain(truncatedTo("t"));
    expect(rendered.length).toBeLessThan(2 * MAX_RENDERED_CHARACTERS + 200);
    expect((summary().textContent ?? "").length).toBeLessThan(20_000);
  });

  it("bounds the waterfall summary the server fills", () => {
    // Kills rendering the waterfall path raw, which put 2,000,138 characters into the DOM.
    render(
      <ResultSummary
        job={job({
          kind: "preview_forward",
          status: "succeeded",
          result: {
            waterfall: {
              shape: [2, 2],
              dtype: "d".repeat(1_000_000),
              statistic: "q".repeat(1_000_000),
            },
            taps: { ["k".repeat(1_000_000)]: { shape: [4], dtype: "f".repeat(1_000_000) } },
            uniform_sky_mean: { ["u".repeat(1_000_000)]: 1 },
          },
        })}
      />,
    );

    const rendered = screen.getByRole("region", { name: "Waterfall summary" }).textContent ?? "";
    for (const character of ["d", "q", "k", "f", "u"]) {
      expect(rendered).toContain(truncatedTo(character));
    }
    expect(rendered.length).toBeLessThan(5 * MAX_RENDERED_CHARACTERS + 400);
    expect((summary().textContent ?? "").length).toBeLessThan(30_000);
  });

  it("names a warning whose only finding sits past the render cap", () => {
    // Kills reading the sliced list for the severity question. MAX_RENDERED_FINDINGS bounds what
    // the DOM holds, not what the job means: a succeeded job whose only `warn` sat at index 149
    // was labelled a green "Succeeded" and lost §9's corrective action entirely.
    const rows: unknown[] = Array.from({ length: 150 }, (_, index) => finding(index, "legible"));
    rows[149] = { ...finding(149, "check the declared gain"), severity: "warn" };

    render(<ResultSummary job={job({ status: "succeeded", result: { findings: rows } })} />);

    const chip = screen.getByText("Warning").closest("[role]");
    expect(chip).toHaveAttribute("role", "status");
    expect(chip).toHaveClass("status-warning");
    expect(screen.getByText("Review the warning before continuing.")).toBeVisible();
    expect(screen.queryByText("Succeeded")).toBeNull();
    // The bound is untouched: the warning changes the words, never what the DOM carries.
    expect(within(summary()).getAllByRole("listitem")).toHaveLength(MAX_RENDERED_FINDINGS);
    expect(within(summary()).queryByText("check the declared gain")).toBeNull();
  });

  it("leaves a succeeded job green when no row anywhere in the payload warns", () => {
    // The other side: scanning the whole payload must not invent a warning out of report rows.
    const rows = Array.from({ length: 150 }, (_, index) => finding(index, "legible"));

    render(<ResultSummary job={job({ status: "succeeded", result: { findings: rows } })} />);

    expect(screen.getByText("Succeeded").closest("[role]")).toHaveClass("status-success");
    expect(screen.queryByText("Warning")).toBeNull();
    expect(screen.queryByText("Review the warning before continuing.")).toBeNull();
  });

  it("ignores a malformed row past the cap that only looks like a warning", () => {
    // The severity question is asked with the same predicate the list renders by, so a row the
    // list would have rejected cannot turn a succeeded job amber from beyond the cap either.
    const rows: unknown[] = Array.from({ length: 150 }, (_, index) => finding(index, "legible"));
    rows[149] = { severity: "warn" };

    render(<ResultSummary job={job({ status: "succeeded", result: { findings: rows } })} />);

    expect(screen.getByText("Succeeded").closest("[role]")).toHaveClass("status-success");
    expect(screen.queryByText("Warning")).toBeNull();
  });

  it("counts every finding it did not render, not only the ones past the cap", () => {
    // Kills a count taken as raw.length - 100: rows dropped for being malformed inside the first
    // hundred were not rendered either, and under-reporting them hides what the user never saw.
    const findings = Array.from({ length: MAX_RENDERED_FINDINGS }, (_, index) => (
      index < 40 ? { check: index } : finding(index, "legible")
    ));

    render(<ResultSummary job={job({ result: { findings } })} />);

    expect(within(summary()).getAllByRole("listitem")).toHaveLength(60);
    expect(within(summary()).getByText(`40 further findings were not rendered ${TRUNCATION_MARKER}`))
      .toBeVisible();
  });

  it("still names the dropped findings when every row is malformed", () => {
    // Kills parking the notice inside `knownFindings.length > 0`, where a wholly malformed payload
    // renders nothing at all and tells the user nothing was found.
    render(<ResultSummary job={job({ result: { findings: [{ check: 1 }, { check: 2 }, null, "x", []] } })} />);

    expect(within(summary()).queryAllByRole("listitem")).toHaveLength(0);
    expect(within(summary()).getByText(`5 further findings were not rendered ${TRUNCATION_MARKER}`))
      .toBeVisible();
  });

  it("counts the cap in code points, as the backend does, and never cuts a surrogate pair", () => {
    // boundedText sliced UTF-16 code units while Python's len() counts code points, so a message
    // the backend never considered oversized was truncated here, and the cut could land between
    // the halves of a surrogate pair and put a lone surrogate in the DOM.
    const emoji = "😀";
    const view = render(<ResultSummary job={job({ message: emoji.repeat(3_000) })} />);
    // 3000 code points is inside the cap; 6000 code units is not. Python forwards it whole.
    expect(summary().textContent).not.toContain(TRUNCATION_MARKER);
    expect(screen.getByText(emoji.repeat(3_000))).toBeVisible();
    view.unmount();

    render(<ResultSummary job={job({ message: emoji.repeat(MAX_RENDERED_CHARACTERS + 1) })} />);
    const rendered = [...summary().querySelectorAll("p")]
      .map((paragraph) => paragraph.textContent ?? "")
      .find((text) => text.endsWith(TRUNCATION_MARKER)) ?? "";
    expect([...rendered]).toHaveLength(MAX_RENDERED_CHARACTERS);
    expect(rendered.startsWith(emoji)).toBe(true);
    // No half of a pair left behind on either side of the cut.
    expect(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(rendered))
      .toBe(false);
  });

  it("names the forward taps and uniform-sky probes it did not render", () => {
    // MAX_PREVIEW_DIMENSION is 64 while the backend forwards up to 256, so a preview carrying 200
    // taps dropped 136 of them and said nothing — silence a reader cannot tell apart from a model
    // that declared 64 taps. The finding list has named its own drops all along.
    const taps = Object.fromEntries(Array.from({ length: 200 }, (_, index) => (
      [`tap-${index}`, { shape: [4], dtype: "float32" }]
    )));
    const uniform = Object.fromEntries(Array.from({ length: 100 }, (_, index) => (
      [`probe-${index}`, index]
    )));

    render(
      <ResultSummary
        job={job({
          kind: "preview_forward",
          status: "succeeded",
          result: { waterfall: { shape: [2, 2], dtype: "float32" }, taps, uniform_sky_mean: uniform },
        })}
      />,
    );

    const waterfall = screen.getByRole("region", { name: "Waterfall summary" });
    expect(within(waterfall).getAllByRole("list")).toHaveLength(2);
    expect(within(waterfall).getByRole("list", { name: "Forward taps" })
      .querySelectorAll("li")).toHaveLength(64);
    expect(within(waterfall).getByRole("list", { name: "Uniform-sky probes" })
      .querySelectorAll("li")).toHaveLength(64);
    expect(within(waterfall)
      .getByText(`136 further forward taps were not rendered ${TRUNCATION_MARKER}`)).toBeVisible();
    expect(within(waterfall)
      .getByText(`36 further uniform-sky probes were not rendered ${TRUNCATION_MARKER}`))
      .toBeVisible();
  });

  it("claims nothing was dropped when exactly the cap arrived", () => {
    // The other side of that boundary, so a notice invented for a payload inside the cap fails.
    const taps = Object.fromEntries(Array.from({ length: 64 }, (_, index) => (
      [`tap-${index}`, { shape: [4], dtype: "float32" }]
    )));

    render(
      <ResultSummary
        job={job({
          kind: "preview_forward",
          status: "succeeded",
          result: { waterfall: { shape: [2, 2], dtype: "float32" }, taps },
        })}
      />,
    );

    const waterfall = screen.getByRole("region", { name: "Waterfall summary" });
    expect(within(waterfall).getByRole("list", { name: "Forward taps" })
      .querySelectorAll("li")).toHaveLength(64);
    expect(within(waterfall).queryByText(/further forward taps were not rendered/)).toBeNull();
    expect(within(waterfall).queryByText(/further uniform-sky probes were not rendered/)).toBeNull();
  });

  it("counts a malformed tap inside the cap among the taps it did not render", () => {
    // Counted like droppedFindings: rows past the cap AND rows inside it too malformed to render,
    // because both are rows the reader has no other way of learning about.
    const taps = {
      ...Object.fromEntries(Array.from({ length: 10 }, (_, index) => [`tap-${index}`, null])),
      good: { shape: [4], dtype: "float32" },
    };

    render(
      <ResultSummary
        job={job({
          kind: "preview_forward",
          status: "succeeded",
          result: { waterfall: { shape: [2, 2], dtype: "float32" }, taps },
        })}
      />,
    );

    const waterfall = screen.getByRole("region", { name: "Waterfall summary" });
    expect(within(waterfall).getByRole("list", { name: "Forward taps" })
      .querySelectorAll("li")).toHaveLength(1);
    expect(within(waterfall)
      .getByText(`10 further forward taps were not rendered ${TRUNCATION_MARKER}`)).toBeVisible();
  });

  it("still renders a whole 64 by 64 preview grid", () => {
    const values = Array.from({ length: 64 }, (_, row) => (
      Array.from({ length: 64 }, (_, column) => row + column)
    ));

    render(
      <ResultSummary
        job={job({
          kind: "preview_forward",
          status: "succeeded",
          result: {
            waterfall: {
              shape: [64, 64],
              dtype: "float32",
              minimum: 0,
              maximum: 126,
              mean: 63,
              values,
            },
          },
        })}
      />,
    );

    const table = screen.getByRole("table", { name: "Predicted waterfall" });
    const rows = table.querySelectorAll("tr");
    expect(rows).toHaveLength(64);
    expect(rows[0].querySelectorAll("td")).toHaveLength(64);
    expect(summary().textContent).not.toContain(TRUNCATION_MARKER);
  });
});
