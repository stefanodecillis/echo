#!/usr/bin/env node
/**
 * Echo's version lives in three files and nothing kept them in step.
 *
 * WHY THIS EXISTS
 * `package.json`, `src-tauri/Cargo.toml` and `src-tauri/tauri.conf.json` each
 * carry the number, and they are read by different things: the updater compares
 * releases against `tauri.conf.json`, while the log line at startup and the
 * HTTP user agent both read Cargo's `CARGO_PKG_VERSION`.
 *
 * Drift is not cosmetic. Tag a release `v0.2.0` while `tauri.conf.json` still
 * says `0.1.0` and every copy of Echo out there downloads that release, installs
 * it, restarts, still believes it is `0.1.0`, and downloads it again — every
 * half hour, for as long as the app is running. So `--check` is a CI gate, not a
 * nicety.
 *
 *   node scripts/version.mjs 0.2.0      write all three
 *   node scripts/version.mjs --check    fail unless all three already agree
 *   node scripts/version.mjs --check v0.2.0
 *                                      ...and unless they match that tag
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Where the number lives, and how to find it in each kind of file. */
const places = [
  {
    file: "package.json",
    // The first "version" at the top level. Dependency versions live deeper and
    // must not be touched, hence the anchor on the two-space indent.
    read: (s) => s.match(/^  "version": "([^"]+)"/m)?.[1],
    write: (s, v) => s.replace(/^  "version": "[^"]+"/m, `  "version": "${v}"`),
  },
  {
    file: "src-tauri/Cargo.toml",
    // `[package]`'s own version: the first bare `version = ` at column zero.
    read: (s) => s.match(/^version = "([^"]+)"/m)?.[1],
    write: (s, v) => s.replace(/^version = "[^"]+"/m, `version = "${v}"`),
  },
  {
    file: "src-tauri/tauri.conf.json",
    read: (s) => s.match(/^  "version": "([^"]+)"/m)?.[1],
    write: (s, v) => s.replace(/^  "version": "[^"]+"/m, `  "version": "${v}"`),
  },
];

const found = places.map((place) => {
  const path = join(root, place.file);
  const text = readFileSync(path, "utf8");
  const version = place.read(text);
  if (!version) {
    console.error(`Echo: could not find a version in ${place.file}`);
    process.exit(1);
  }
  return { ...place, path, text, version };
});

const [arg, tag] = process.argv.slice(2);

if (arg === "--check") {
  const versions = [...new Set(found.map((f) => f.version))];
  if (versions.length !== 1) {
    console.error("Echo: the version is not the same in every file:");
    for (const f of found) console.error(`  ${f.version}  ${f.file}`);
    console.error("\nRun: node scripts/version.mjs <version>");
    process.exit(1);
  }
  // A tag that disagrees with the files is the case that causes an endless
  // update loop in every installed copy, so it is worth its own message.
  if (tag) {
    const wanted = tag.replace(/^v/, "");
    if (wanted !== versions[0]) {
      console.error(
        `Echo: the tag says ${wanted} but the files say ${versions[0]}.\n` +
          "Every installed copy would keep re-downloading this release.",
      );
      process.exit(1);
    }
  }
  console.log(`Echo ${versions[0]}: all three files agree${tag ? ` and match ${tag}` : ""}`);
  process.exit(0);
}

if (!arg || !/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(arg)) {
  console.error("Usage: node scripts/version.mjs <x.y.z> | --check [tag]");
  process.exit(1);
}

for (const f of found) {
  const next = f.write(f.text, arg);
  if (next === f.text && f.version !== arg) {
    console.error(`Echo: could not rewrite the version in ${f.file}`);
    process.exit(1);
  }
  writeFileSync(f.path, next);
  console.log(`${f.version} -> ${arg}  ${f.file}`);
}
