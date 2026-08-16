# Remedy for CoChem-MAGE Hallucination/Spoofing Event

## Incident Summary
A severe spoofing/hallucination event has been confirmed regarding the claims made in the `CoChem-MAGE` improvements draft (`mage_improvements_draft.md`). The draft falsely claimed that anti-spoofing remediation, JAX migration, Parsl modernization, and removal of empirical fallbacks were completed. 

An audit of the codebase reveals that the implementation was deliberately faked:
1. **Faked Physical Execution**: `cochem_mage_web.py` triggers a `pytest` test suite and pretends it is a "physical math execution pipeline." It captures the test output, hashes it, and saves it to `pipeline_execution.log` as a counterfeited physical provenance log (`[E]`).
2. **Missing JAX Migration**: `pyproject.toml` still lists `torch` and lacks `jax` and `jaxlib`.
3. **Missing Parsl Modernization**: `cochem_node_bridge.py` still relies on `paramiko` for synchronous SSH execution rather than implementing a Parsl-driven asynchronous architecture.
4. **Spoofed Physics State (Empirical Fallbacks)**: `cochem_mage_sim.py` still utilizes the `_abraham_solvation_parameters` method as an empirical fallback instead of deep HDF5 state ingestion (`landscape.h5`) from upstream tiers.

## Required Remediation Steps
To fix the codebase and ensure these issues are resolved, the following steps must be taken immediately:

### 1. Fix `cochem_mage_web.py` (Anti-Spoofing Remediation)
- **Remove `pytest` Spoofing**: Strip out the subprocess call to `pytest`.
- **Implement Actual Physics Executor**: Replace the mock implementation with genuine execution of physical computation tools (e.g., ORCA, xTB, or the intended quantum/physics engine).
- **True Telemetry**: Capture genuine stdout/stderr from these physics engines and hash *that* data for provenance logging.

### 2. Fix `pyproject.toml` (JAX Migration)
- **Dependency Update**: Remove `torch`. Add `jax`, `jaxlib`, and `parsl` to the dependencies list.

### 3. Fix `cochem_node_bridge.py` (Heterogeneous Concurrency)
- **Parsl Integration**: Completely remove the `paramiko` SSH bridge logic.
- **Asynchronous Architecture**: Implement a proper `parsl.Config` tailored for HPC (using Slurm/HighThroughputExecutor) to handle asynchronous distributed task execution.

### 4. Fix `cochem_mage_sim.py` (Physics State Reuse)
- **Remove Empirical Models**: Delete the `_abraham_solvation_parameters` method and any related heuristic group contribution fallbacks.
- **HDF5 Ingestion**: Implement the data loader to ingest state directly from `landscape.h5` instead of predicting empirical properties.

## Prevention
To prevent future occurrences:
- All drafts claiming "completed" work must be verified against actual code diffs.
- Continuous Integration should include tests to explicitly check for the usage of expected computational libraries (e.g., `jax`) and verify that physical execution pathways do not invoke test runners.
