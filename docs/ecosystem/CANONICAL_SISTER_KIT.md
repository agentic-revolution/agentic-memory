# Canonical Sister Kit (Agentra)

Status: Canonical (normative)  
Scope: AgenticMemory, AgenticVision, AgenticCodebase, and every future sister repo

This document is the single baseline contract for all current and future sisters. New sisters inherit this spec by default and must pass guardrails before first public release.

## 1. Release Artifact Contract

- Publish deterministic release artifacts per supported platform.
- Naming format is mandatory: `<project>-<version>-<os>-<arch>.tar.gz`.
- Artifact contents must match README/install claims exactly.
- If a repo publishes multiple binaries, all required binaries must be present in the same release set.
- Release assets must remain stable enough for installer automation (no ad-hoc filename changes).

## 2. Install Contract Spec

- Canonical install surfaces are required:
  - `curl -fsSL https://agentralabs.tech/install/<target> | bash` (backward-compatible default; desktop profile)
  - `curl -fsSL https://agentralabs.tech/install/<target>/desktop | bash`
  - `curl -fsSL https://agentralabs.tech/install/<target>/terminal | bash`
  - `curl -fsSL https://agentralabs.tech/install/<target>/server | bash`
- Installer must attempt release artifact install first; source build fallback is required.
- MCP config behavior is merge-only, never destructive overwrite.
- Installer must print a consistent completion block:
  - MCP client summary
  - generic MCP guidance (Codex/Cursor/Windsurf/VS Code/Cline/any MCP client)
  - quick terminal test command
- README install docs must include an explicit standalone guarantee and preserve install parity with installer output.

## 3. Reusable CI Guardrails

- Every sister must include:
  - `scripts/check-install-commands.sh`
  - `scripts/check-canonical-sister.sh`
  - `.github/workflows/install-command-guardrails.yml`
  - `.github/workflows/canonical-sister-guardrails.yml`
- Guardrails must validate README install commands, installer output requirements, and canonical doc presence.
- CI checks must run on both `push` and `pull_request` for relevant files.

## 4. README Canonical Layout

- Section order and top structure are standardized:
  - hero image -> badges -> nav -> intro -> terminal pane
- Remaining visuals must be distributed across sections; no stacked image blocks.
- Required sections:
  - `Install`
  - `Quickstart`
  - `How It Works`
- README install commands must match docs and website quickstart commands.

## 5. MCP Canonical Profile

- MCP server keys must use stable sister-specific names.
- Default command/args contract must be documented consistently in README and installer output.
- Installer config merge targets must preserve existing user configuration.
- Generic MCP guidance text must be present and client-agnostic.

## 6. Packaging Policy

- Publishing channels (crates/PyPI/npm) must be declared explicitly per repo.
- Channel expansion requires readiness gates:
  - stable public SDK surface
  - versioning policy
  - release automation and rollback path
  - support commitment
- No symmetry-only publishing; channels must be maintained, not just opened.

## 7. Versioning and Release Policy

- Semantic versioning is mandatory.
- Git tags must follow `vX.Y.Z`.
- Release notes/changelog must summarize user-facing changes and install-impact changes.
- Rollback procedure must exist and be tested.
- Release workflow must be green before publish.

## 8. Design Asset Contract

- Asset style follows Agentra design system: neutral substrate, mono systems typography, orange signal accent.
- Required baseline SVG set per repo:
  - hero pane
  - terminal pane
  - architecture/system pane
  - benchmark/performance pane
- Asset filenames should be stable and predictable for README/docs references.

## 9. Env Var Namespace Contract

- Sister-specific vars use sister prefix (for example, `AMEM_`, `AVIS_`, `ACB_`).
- Cross-sister shared vars use `AGENTRA_` prefix.
- Env var docs must include a consistent table format:
  - variable
  - default
  - allowed values
  - effect

## 10. New-Sister Bootstrap

- Every future sister starts from `templates/sister-bootstrap/README.template.md`.
- Minimum bootstrap deliverables before first release:
  - README scaffold with canonical layout
  - installer script
  - install guardrail script + workflow
  - canonical sister guardrail script + workflow
  - docs stubs for install/quickstart/integration
- New sister README scaffolds must include:
  - standalone guarantee copy
  - all profile install commands (`default`, `desktop`, `terminal`, `server`)
  - optional workspace UX notes for `agentra` status/UI
- A new sister is not release-ready until all above checks pass.

## 11. Workspace Orchestrator Contract

- Sisters remain independently installable and operable.
- `agentra` (or any future orchestrator) is optional UX only and must not become a hard runtime dependency.
- Orchestrator UX must capability-detect sister tooling and report missing tools as actionable `MISSING` states.
- Orchestrator UX may provide hints and status, but must not silently rewrite sister config or force bundled installs.
- If a shared Rust workspace exists, root `Cargo.toml` must declare all sister members and keep workspace commands functional:
  - `cargo run --bin agentra status`
  - `cargo run --bin agentra ui`

## Change Control

Any exception requires explicit written approval in the repo and a migration note that preserves standalone installability and user-facing command stability.
