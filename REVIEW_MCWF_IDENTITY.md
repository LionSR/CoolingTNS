# PR Review: Reject identity initial states for MCWF

**Review Date:** 2026-06-17  
**Branch:** `codex/mcwf-identity-initial-state-4860`  
**Reviewer:** Cursor Cloud Agent  

---

## Executive Summary

✅ **APPROVED** — All three review criteria are satisfied:

1. ✅ **Parser validation**: Command-line rejection works correctly
2. ✅ **ED/TN initial-state consistency**: Both backends reject MCWF identity identically
3. ✅ **Density-matrix identity**: Maximally mixed state `I/2^N` preserved in both backends

---

## 1. Parser Validation ✅

### Implementation (`src/argparse.jl:3-17`)

The new `_validate_initial_state_args` function provides **defense in depth** by catching invalid combinations at the earliest possible point—before problem setup:

```julia
if init_state == "identity" && sim_method == "monte_carlo"
    throw(ArgumentError(
        "--init_state identity denotes the maximally mixed density matrix " *
        "and requires --sim_method density_matrix. For Monte Carlo " *
        "wavefunction simulations, choose a pure initial state such as " *
        "product or theta."
    ))
end
```

### Correctness

- **Early validation**: Fails fast before resource allocation
- **Clear error message**: Explicitly states that `identity` is the maximally mixed density matrix
- **Actionable guidance**: Suggests valid alternatives (`product`, `theta`)
- **Integration**: Called by `parse_commandline` at line 126, ensuring all CLI entry points are protected

### Test Coverage (`test/test_cooling_interface.jl:98-110`)

```julia
@testset "Command-line initial-state validation" begin
    # Valid: density_matrix + identity
    parsed = CoolingTNS.parse_commandline([
        "--sim_method", "density_matrix",
        "--init_state", "identity",
    ])
    @test parsed["init_state"] == "identity"

    # Invalid: monte_carlo + identity
    @test_throws ArgumentError CoolingTNS.parse_commandline([
        "--sim_method", "monte_carlo",
        "--init_state", "identity",
    ])
end
```

**Assessment**: Complete. Tests both the valid (DM + identity) and invalid (MCWF + identity) paths.

---

## 2. ED/TN Initial-State Consistency ✅

### Shared Validation Helper (`src/initial_state.jl:35-43`)

Both backends use the same validation function before attempting to construct MCWF states:

```julia
function _reject_identity_for_mcwf(init_type::String)
    if init_type == "identity"
        throw(ArgumentError(
            "init_type=\"identity\" denotes the maximally mixed density matrix " *
            "and is not a single MonteCarloWavefunction state. Use DensityMatrix() " *
            "or choose a pure initial state such as \"product\" or \"theta\"."
        ))
    end
end
```

**Placement Note**: Properly placed under "Shared Initial-State Validation" header (line 32), not under the ED-specific section—addresses prior review comment.

### TN Backend MCWF (`src/initial_state.jl:151-164`)

```julia
function setup_initial_state(
    problem::CoolingProblem{TNBackend}, 
    sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
    init_type::String, theta::Float64
) where E<:EvolutionMethod
    _reject_identity_for_mcwf(init_type)  # Line 153
    # ... MPS construction ...
end
```

### ED Backend MCWF (`src/initial_state.jl:166-174`)

```julia
function setup_initial_state(
    problem::CoolingProblem{EDBackend}, 
    sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
    init_type::String, theta::Float64
) where E<:EvolutionMethod
    _reject_identity_for_mcwf(init_type)  # Line 169
    # ... state vector construction ...
end
```

### ED Pure-State Constructor (`src/initial_state.jl:118-124`)

Additional defense layer—`create_theta_state_ed` also rejects `identity` since it constructs pure state vectors:

```julia
function create_theta_state_ed(N::Int, init_type::String, theta::Float64)::EDStateVector
    if init_type == "identity"
        throw(ArgumentError(
            "create_theta_state_ed constructs pure state vectors; " *
            "init_type=\"identity\" is a density matrix initial state."
        ))
    end
    # ...
end
```

**Docstring Update** (line 115-116): "The value `init_type == "identity"` is rejected because this constructor returns pure state vectors, while the identity initial state denotes the maximally mixed density matrix."

### Test Coverage (`test/test_initial_states.jl`)

- **TN MCWF rejection**: Lines 33-37
- **ED MCWF rejection**: Lines 70-77 (tests both `setup_initial_state` and `create_theta_state_ed`)
- **Consistency**: Both backends throw `ArgumentError` with clear messages

**Assessment**: Perfect consistency. Both backends use the same validation logic and reject `identity` for MCWF with identical behavior.

---

## 3. Density-Matrix Identity = Maximally Mixed State ✅

### TN Backend Density Matrix (`src/initial_state.jl:181-196`)

```julia
function setup_initial_state(
    problem::CoolingProblem{TNBackend}, 
    sim_params::UnifiedSimulationParameters{DensityMatrix, E},
    init_type::String, theta::Float64
) where E<:EvolutionMethod
    ϕ₀ = problem.ϕ₀
    sites_sys = siteinds(ϕ₀)

    if init_type == "identity"
        ρ_s = MPO(sites_sys, "Id")
        ρ_s = ρ_s / (2.0^length(sites_sys))  # I/2^N, trace=1
    # ... other branches ...
end
```

**Mathematical correctness**:
- `MPO(sites_sys, "Id")` creates the identity operator with trace `2^N`
- Division by `2^N` gives `I/2^N`, the maximally mixed state with trace 1
- ITensors convention: trace of identity MPO is `2^N`

### ED Backend Density Matrix (`src/initial_state.jl:199-211`)

```julia
function setup_initial_state(
    problem::CoolingProblem{EDBackend}, 
    sim_params::UnifiedSimulationParameters{DensityMatrix, E},
    init_type::String, theta::Float64
) where E<:EvolutionMethod
    N = problem.extra.ham_params.N

    if init_type == "identity"
        ρ = maximally_mixed_ed(N)  # I/2^N
    # ... other branches ...
end
```

Where `maximally_mixed_ed` (`src/ed_backend.jl:107-110`):

```julia
function maximally_mixed_ed(n_qubits::Int)
    dim = 2^n_qubits
    return EDDensityMatrix(Matrix{ComplexF64}(I, dim, dim) / dim, n_qubits)
end
```

**Mathematical correctness**:
- Creates `I/2^N` directly as a `dim × dim` matrix
- Trace = `dim × (1/dim)` = 1 ✓
- Pure density matrix (maximally mixed)

### Test Coverage (`test/test_initial_states.jl`)

#### TN Density Matrix Identity (lines 97-102)
```julia
@testset "Identity State MPO" begin
    state = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
    @test state isa CoolingTNS.QuantumState
    @test state.state isa MPO
    @test length(state.state) == N
end
```

#### ED Density Matrix Identity (lines 122-128)
```julia
@testset "Identity State Density Matrix" begin
    state = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
    @test state isa CoolingTNS.QuantumState
    @test state.state isa CoolingTNS.EDDensityMatrix
    @test tr(state.state.data) ≈ 1.0 + 0.0im atol=1e-12
    @test state.state.data ≈ Matrix{ComplexF64}(I, 2^N, 2^N) / 2^N atol=1e-12
end
```

**Verification**: The ED test explicitly checks both:
1. Trace normalization: `tr(ρ) ≈ 1`
2. Matrix structure: `ρ ≈ I/2^N`

**Assessment**: Both backends correctly implement the maximally mixed state. ED tests verify the mathematical properties explicitly. TN implementation is mathematically equivalent by ITensors convention.

---

## Documentation Review

### Help Text (`src/argparse.jl:117`)

```julia
"--init_state"
help = "initial state type: 'product' (default), 'identity' (maximally mixed; density matrix only), 'theta' (use --theta value)"
```

✅ Clearly states:
- `identity` = maximally mixed
- `identity` = density matrix only

### Theta Convention Documentation (`src/argparse.jl:121`)

```julia
"--theta"
help = "theta code parameter for initial state: -0.5 -> |0>, 0 -> |+>, 0.5 -> |1>"
```

✅ Documents the theta code convention at the CLI level

### New Exported Functions (`src/initial_state.jl:50-78`)

- `initial_product_angle(theta_code)`: Converts code → physical angle
- `theta_code_from_initial_product_angle(alpha)`: Converts physical angle → code
- `theta_site_amplitudes(theta_code)`: Returns `(amp0, amp1)` for given code

✅ Full API surface for theta convention, replacing private `_theta_site_amplitudes`

---

## Physics Correctness

### Problem Statement

Before this PR, `init_type="identity"` produced three physically different states:

| Backend | Method | Before PR | Physical State |
|---------|--------|-----------|----------------|
| ED | DM | `I/2^N` | ✓ Maximally mixed |
| ED | MCWF | `ones(2^N)/√(2^N)` | ✗ Equal superposition (pure) |
| TN | MCWF | `randomMPS(D=1)` | ✗ Random product state |

Only the ED-DM path was correct. The MCWF paths produced **pure states** that:
- Are not infinite-temperature states
- Have different energies from `I/2^N`
- Have different parity content
- Differ from each other (ED superposition ≠ TN random product)

### Resolution

This PR makes `identity` density-matrix-only, rejecting it for MCWF. The one true maximally mixed state `I/2^N` is preserved in both DM backends.

✅ **Correct conservative behavior**: Reject until an explicit ensemble or purification representation is implemented.

---

## Dispatch Architecture Compliance

The added validation checks are **input validation**, not method selection:
- ✅ No dispatch surface changes
- ✅ No if-else for method routing (only for argument validation)
- ✅ Preserves Backend × SimulationMethod × EvolutionMethod dispatch structure
- ✅ Follows `CLAUDE.md` pure-dispatch architecture

---

## Regression Risk Assessment

### Breaking Changes
- **MCWF runs with `identity`**: Will now fail with clear error
- **Migration path**: Use `product` or `theta` for MCWF

### Non-Breaking
- **All DM runs**: Unchanged behavior
- **MCWF with product/theta**: Unchanged behavior

### Risk Level
**Medium** (as noted in PR description): Breaking for MCWF+identity users, but they were getting incorrect physics. The error message provides clear migration guidance.

---

## Code Quality

### Strengths
1. **Clear error messages**: All three validation points provide actionable guidance
2. **Comprehensive test coverage**: 88 initial-state tests pass
3. **Documentation**: Help text, docstrings, and exported utilities all updated
4. **Defense in depth**: CLI validation + backend validation + ED constructor validation
5. **No trailing whitespace**: `git diff --check` passes

### Code Style
- ✅ Multiple dispatch preserved
- ✅ Type-stable implementations
- ✅ Consistent naming conventions
- ✅ Proper module structure (shared validation under neutral header)

---

## Test Suite Status

From PR description:
- `test/test_initial_states.jl`: 88 tests passed
- `test/test_cooling_interface.jl`: 103 tests passed
- `test/runtests.jl`: 999 tests passed
- `git diff --check origin/main...HEAD`: passed

---

## Final Recommendation

✅ **APPROVED FOR MERGE**

All three review criteria are satisfied:

1. ✅ **Parser validation**: Early CLI check with clear error message
2. ✅ **ED/TN consistency**: Identical rejection logic via shared helper
3. ✅ **Density-matrix identity**: Both backends correctly implement `I/2^N`

### Prior Review Comments Addressed

From previous Claude review:
- ✅ Moved `_reject_identity_for_mcwf` to neutral "Shared validation" section
- ✅ Documented `create_theta_state_ed` rejection in docstring
- ✅ Added early command-line validation

### Additional Observations

- The theta-code convention is now well-documented with exported utility functions
- Test coverage includes both positive (DM+identity works) and negative (MCWF+identity fails) cases
- Error messages consistently describe `identity` as "the maximally mixed density matrix"

---

**This PR correctly enforces the mathematical constraint that the identity initial state represents the maximally mixed density matrix `I/2^N`, which cannot be represented by a single Monte Carlo wavefunction trajectory.**
