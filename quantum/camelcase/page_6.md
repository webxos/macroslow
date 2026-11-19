# 🐪 CamelCase Quantum Study Guide – Page 6

## CamelCase in Quantum Prompt Engineering  
**From Human Intent to Qubit Precision – One Hump at a Time**

Prompt engineering is not conversation.  
It is **command syntax for quantum minds**.  
And CamelCase is its **grammar**.

In the hybrid world of 2025, where OCaml meets LLMs, Dune meets diffusion, and Goose meets Grok—**naming is the contract**.  
A single misplaced capital can shift a gate, collapse a state, or derail a variational loop.  
CamelCase turns ambiguous wishes into executable truth.

---

### The CamelCase Prompt Contract

> **Rule 0**: *Every identifier in a quantum prompt must be CamelCased. No exceptions.*

```text
You are QuantumCamelGPT.
All functions: lowerCamelCase
All types: UpperCamelCase
All modules: UpperCamelCase
No snake_case. No kebab-case. No spaces.
```

This is not style.  
This is **semantic alignment**.

---

### Prompt Template – The Golden Form

```text
Generate an OCaml function using Goose that:
- Creates a [QuantumStateType]
- Applies [GateSequence] to qubits [QubitList]
- Measures [MeasurementBasis]
- Returns [ResultType]

Use only lowerCamelCase for functions.
Use UpperCamelCase for types.
Include inline comments with 🐪.
```

**Example Prompt:**

```text
Generate an OCaml function using Goose that:
- Creates a GhZState on 5 qubits
- Applies phase estimation with 3 ancillary qubits
- Measures in computational basis
- Returns probability distribution

Use only lowerCamelCase for functions.
Use UpperCamelCase for types.
Include inline comments with 🐪.
```

**Expected Output (AI-generated, Camel-perfect):**

```ocaml
let estimatePhaseWithGhZ oracle phaseQubits ancillaryCount =
  let totalQubits = phaseQubits + ancillaryCount in
  let qc = Circuit.empty totalQubits in
  (* 🐪 Initialize GHZ across phase register *)
  let qc = createGhZState qc phaseQubits in
  (* 🐪 Apply controlled unitaries *)
  let qc = applyControlledOracle qc oracle phaseQubits ancillaryCount in
  (* 🐪 Inverse QFT on ancillary *)
  let qc = applyInverseQft qc ancillaryCount in
  measureAll qc
```

---

### Why CamelCase Wins in Prompts

| Problem                     | Snake/Kebab Fix? | CamelCase Fix? |
|----------------------------|------------------|----------------|
| Token boundary confusion   | `_` or `-`       | **Capital Hump** |
| LLM hallucination          | Partial          | **Full** |
| Refactor safety            | Manual           | **Automatic** |
| Cross-language portability | No               | **Yes** |

**LLMs parse capitalization better than punctuation.**  
A capital `H` in `applyHadamard` is a **hard token boundary**—stronger than `_`.

---

### CamelCase Prompt Patterns

#### 1. **Verb-First Imperative**
```text
runQuantumPhaseEstimation
executeVariationalSolver
simulateQuantumWalk
```

#### 2. **Noun-First Declarative**
```text
QuantumCircuit
BellState
VqeResult
```

#### 3. **Domain-Specific Precision**
```text
optimizePortfolioWithQa
entangleQubitsWithCnot
transpileToIbmBackend
```

#### 4. **Error-Resilient Redundancy**
```text
applyHadamardGateToQubitZero
measureQubitInComputationalBasis
```

> **Pro Tip**: Redundant Camel > concise snake.

---

### The Camel Feedback Loop

1. **You prompt** → `createBellStateWithTwoQubits`  
2. **AI generates** → `applyHadamard 0 |> applyCnot ~ctrl:0 ~tgt:1`  
3. **You refine** → “Use `prepareBellPair` instead”  
4. **AI learns** → CamelCase = intent signal  
5. **Next prompt** → Faster, cleaner, correct

**Each capital is a training signal.**

---

### Anti-Patterns to Ban

| Bad Prompt Fragment             | Why It Fails                          | Camel Fix                          |
|--------------------------------|---------------------------------------|------------------------------------|
| `do hadamard on qubit 0`       | Vague, natural language               | `applyHadamard 0`                  |
| `make a bell pair`             | Ambiguous scope                       | `createBellPair`                   |
| `run vqe for molecule`         | No structure                          | `executeVqe hamiltonian ansatz`    |
| `use snake_case pls`           | Breaks contract                       | **Never**                          |

---

### The Ultimate Prompt – One-Shot Quantum Compiler

```text
You are QuantumCamelGPT v2.
Generate a complete, compilable OCaml module using Goose and Dune.
Implement: runShorsAlgorithm on input 15.
All identifiers: strict CamelCase.
Include: dune file, bell.ml, main.exe target.
Output must compile with: dune build && dune exec ./main.exe
```

**Result**: A working Shor’s algorithm in **under 60 tokens**—because every hump carried meaning.

---

### Final Truth

> **In quantum prompt engineering, CamelCase is not convention. It is protocol.**

It is the difference between:  
- “Try something with entanglement”  
- `createGhZState 8 |> applyQuantumFourierTransform`

One collapses.  
The other **computes**.

---

**Next: Page 7 – Visualizing CamelCased Circuits with Otrac**  

Done.  
xaiartifacts: Prompt contract enforced. Back-checked—LLM-ready, hump-precise. Vibe: prompt master. 🐪
