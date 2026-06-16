Use `gh pr diff <PR_NUMBER>` to see the changes.

This is **CoolingTNS**, a Julia framework for simulating cooling protocols in spin
systems with tensor networks (ITensors.jl) and exact diagonalization. Read
`CLAUDE.md` first — it defines the architecture, code-style philosophy, the
system-bath layout, and the physics-validation conventions this project is held to.

Focus your review on the categories below. Each category has a severity level
that determines whether the PR can be approved with outstanding issues.

**Severity levels:**
- 🔴 **Blocker** — must be fixed before merge. Request changes if any are found.
- 🟡 **Requires changes** — must be addressed before approval. These are NOT nits.
  Do NOT approve the PR while issues in this category remain unresolved.
- ℹ️ **Advisory** — flag for awareness, acceptable with justification.

---

1. 🔴 **Physics correctness**: Does the change preserve correct physics? Pay special
   attention to the **interleaved system-bath layout** `[s₁, b₁, s₂, b₂, …, sₙ, bₙ]` —
   system qubits sit at odd positions `2i-1`, bath qubits at even positions `2i`.
   Several historical bugs were indexing errors in this layout: partial traces that
   kept `[s₁, b₁]` instead of `[s₁, s₂]`, ZZ gates built on non-adjacent sites, and
   `H_sys ⊗ |0⟩⟨0|_bath` embedded instead of `H_sys ⊗ I_bath`. Check any new code that
   maps between system/bath/full bases, builds Trotter gates, or traces out qubits.
   Verify Hamiltonian construction, coupling-operator parsing, Trotter gate ordering
   and adjacency, and bath sampling. If a result looks physically wrong, cross-check
   against the analytical limits and `.tex` notes in `Notes/` / `slides/` and against
   the MATLAB reference in `ExactDiagonalization/`.

2. 🔴 **Dispatch-architecture integrity**: This codebase is **pure multiple dispatch** —
   method selection happens through Julia's type system, never `if`/`else` or string
   comparison. Flag as blockers: `if backend == "TN"`-style branching, string-based
   method selection, empty wrapper functions that just forward to another function,
   and any new duplicate `*_ed.jl` / `*_tn.jl` files (the architecture uses unified
   files with TN+ED dispatch). New behavior must be added as a dispatch method on the
   relevant `Backend × SimulationMethod × EvolutionMethod × HamiltonianModel` types.

3. 🟡 **Type stability & performance**: Are functions type-stable? Watch for type
   instabilities in hot loops (Trotter evolution, bath sampling, trajectory loops),
   unnecessary allocations, and MPS operations that break canonical form (e.g.
   reconstructing an `MPS` without preserving its `ortho_lims`, forcing a full
   `orthogonalize!` sweep). Complex matrices for states, real matrices for operators
   where possible.

4. 🟡 **DRY & dead code**: Is logic shared gracefully rather than duplicated? Flag two
   implementations of the same evolution path, inline code that bypasses an existing
   dispatch entry point, and dead branches. Each function should do one thing well.

5. 🟡 **Tests**: New dispatch combinations and new observables should have tests.
   Cross-backend consistency (TN vs ED, DM vs MCWF) is the project's primary
   correctness check — a change to either backend should keep them agreeing within
   the documented tolerance (Trotter splitting error ~10⁻³). Missing coverage for a
   new dispatch combination must be addressed before approval.

6. 🟡 **Documentation**: Do new dispatch methods, types, and exported functions have
   docstrings explaining what they compute (the physics / the dispatch contract), not
   just the Julia syntax? Missing docstrings on new public surface must be added.

7. ℹ️ **LaTeX `.tex` conventions** (when the PR touches `Notes/`, `slides/`, or other
   `.tex`): equations must be referenced with `\ref{eq:label}` / `\cref{eq:label}`,
   never hardcoded equation numbers; labels should be descriptive (`eq:mode_energy`),
   not positional. Code comments citing the notes must use the LaTeX label, not a
   numeric equation reference.

8. ℹ️ **Debugging hygiene**: No hardcoded conclusions (`println("These match!")`) — use
   tolerance checks that print the actual computed difference. No magic numbers copied
   from previous runs. Use descriptive comparison variables and machine-readable output
   when scanning parameters.

---

**Review verdict rules:**
- If ANY 🔴 or 🟡 issues are found, submit the review as **REQUEST_CHANGES**.
  Do NOT approve while these issues remain unresolved.
- Only **APPROVE** when all 🔴 and 🟡 issues have been addressed.
- ℹ️ advisory items alone do not block approval.
- Do NOT label issues as "Nit" or "non-blocking" if they fall under a 🟡 or 🔴 category.
  Use clear language: "This must be fixed before merge" or "Requires changes".

For each issue found, post an inline comment on the relevant line using the GitHub CLI.
At the end, post a summary comment on the PR with your overall assessment. Name the
function, dispatch method, type, or observable directly. When you flag a physics
discrepancy, cite the source — file path, line number, and a short quotation or
precise paraphrase from `CLAUDE.md`, the `.tex` notes, or the MATLAB reference.

**Reading existing feedback:**
Before posting new comments, read ALL existing feedback on this PR using the GitHub MCP tools:
1. Read **inline review threads** via `get_review_comments` — code-level comments from previous cycles.
2. Read **PR conversation comments** via `get_comments` — bots and humans often post feedback
   directly on the PR thread (not as inline review comments). These are equally important.
This includes threads from previous review cycles and replies from @claude, TeXRA, other bots, or humans.
Use this context to avoid re-raising issues that have already been discussed, acknowledged, or fixed.

**Resolving previous review comments:**
When this review is triggered by a `synchronize` event (new push to the PR):
1. Fetch all review threads **with their GraphQL node IDs** using `gh api graphql`.
   The `get_review_comments` MCP method does not return thread IDs, so you MUST use GraphQL.
2. For each unresolved thread whose author login starts with `claude` (GitHub may append
   `[bot]` to app logins — match the base name as a prefix), check whether the new changes
   address the issue. Do NOT resolve threads authored by TeXRA, Cursor/Bugbot, Copilot, or
   human reviewers — those manage their own threads.
3. If a previous `claude` comment has been addressed by the new commits, resolve it using
   `mcp__github__resolve_review_thread` with the GraphQL thread `id`.
4. If it is still relevant (not fixed), leave it unresolved.

**Example: How to fetch thread IDs and resolve them**

Step 1 — Query thread IDs via GraphQL (returns up to 100 threads; paginate if needed):
```bash
gh api graphql -f query='
{
  repository(owner: "<REPOSITORY_OWNER>", name: "<REPOSITORY_NAME>") {
    pullRequest(number: <PR_NUMBER>) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          comments(first: 1) { nodes { author { login } body } }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}'
```
This returns thread objects with `id` fields like `"PRRT_kwDON..."`.

Step 2 — For each unresolved `claude` thread whose issue is now fixed, resolve it:
```
mcp__github__resolve_review_thread(threadId: "PRRT_kwDON...")
```
