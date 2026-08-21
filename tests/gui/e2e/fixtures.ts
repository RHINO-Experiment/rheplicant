import { lstat, mkdtemp, realpath, rm } from "node:fs/promises";
import { basename, join, resolve, sep } from "node:path";
import { tmpdir } from "node:os";

import AxeBuilder from "../../../tools/config_gui_spike/react/node_modules/@axe-core/playwright/dist/index.js";
import {
  expect,
  test as base,
  type Locator,
  type Page,
} from "../../../tools/config_gui_spike/react/node_modules/@playwright/test/index.js";

export const STARTER_YAML = `schema_version: 1
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
`;

export const COMPLETE_YAML = `schema_version: 1
defaults: [rhino_v1]
runtime:
  seed: 20260820
observation:
  meta:
    telescope: RHINO
  pointing:
    materialise: []
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
variants:
  low_gain:
    model:
      gain:
        gain: {value: 0.9, unit: dimensionless}
runs:
  - name: forward
    kind: forward
`;

export const GRAPH_VARIANT_YAML = COMPLETE_YAML;
export const INVALID_YAML_DRAFT = "schema_version: [\n";

export function outputYaml(target: string): string {
  return `${COMPLETE_YAML}outputs:\n  dir: ${JSON.stringify(target)}\n  clobber: false\n  write:\n    arrays: {format: npz}\nplugins:\n  - json\n`;
}

export function successfulJobYaml(target: string): string {
  return outputYaml(target);
}

export function refusalJobYaml(existingTarget: string): string {
  return outputYaml(existingTarget);
}

export const STALE_JOB_SOURCE_YAML = COMPLETE_YAML;
export const STALE_JOB_CURRENT_YAML = `${COMPLETE_YAML}# accepted after job submission\n`;

export interface AllocatedOutputTarget {
  readonly directory: string;
  readonly target: string;
  cleanup(): Promise<void>;
}

interface DirectoryIdentity {
  readonly dev: number;
  readonly ino: number;
}

export function sameDirectoryIdentity(
  expected: DirectoryIdentity,
  found: DirectoryIdentity,
): boolean {
  return expected.dev === found.dev && expected.ino === found.ino;
}

export async function allocateOutputTarget(): Promise<AllocatedOutputTarget> {
  const temporaryRoot = await realpath(tmpdir());
  const directory = await realpath(
    await mkdtemp(join(temporaryRoot, "rheplicant-e2e-")),
  );
  const expectedPrefix = `${temporaryRoot}${sep}`;
  if (
    !directory.startsWith(expectedPrefix)
    || !basename(directory).startsWith("rheplicant-e2e-")
  ) {
    throw new Error("Temporary output allocation escaped the OS temporary root.");
  }
  const allocated = await lstat(directory);
  if (!allocated.isDirectory()) {
    throw new Error("Temporary output allocation is not a directory.");
  }
  const identity = { dev: allocated.dev, ino: allocated.ino };

  let removed = false;
  return {
    directory,
    target: resolve(directory, "output"),
    async cleanup() {
      if (removed) return;
      const found = await lstat(directory).catch((error: unknown) => {
        if (
          typeof error === "object"
          && error !== null
          && "code" in error
          && error.code === "ENOENT"
        ) return null;
        throw error;
      });
      if (found === null) {
        removed = true;
        return;
      }
      if (!found.isDirectory() || !sameDirectoryIdentity(identity, found)) {
        throw new Error("Refusing to clean a replaced temporary output allocation.");
      }
      await rm(directory, { recursive: true });
      removed = true;
    },
  };
}

interface BrowserFixtures {
  outputTarget: string;
  outputDocument: string;
}

export const test = base.extend<BrowserFixtures>({
  outputTarget: async ({}, use) => {
    const allocation = await allocateOutputTarget();
    try {
      await use(allocation.target);
    } finally {
      await allocation.cleanup();
    }
  },
  outputDocument: async ({ outputTarget }, use) => {
    await use(outputYaml(outputTarget));
  },
});

export { AxeBuilder, expect };
export type { Locator, Page };
