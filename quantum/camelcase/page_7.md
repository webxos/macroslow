# 🐪 CamelCase Quantum Study Guide – Page 7

## Visualizing CamelCased Circuits with Otrac  
**From Humps to Holograms – See Your Logic in Real Time**

Otrac is not a debugger.  
It is a **quantum mirror**.  
Every CamelCase identifier becomes a glowing node in a living circuit diagram—traceable, collapsible, and provably correct.

Built atop OCaml’s type system and Qiskit’s visualization backbone, Otrac transforms `applyHadamard` into a pulsing gate, `createGhZState` into a braided entanglement web, and `executeVqe` into a dynamic energy landscape.

---

### The Otrac Philosophy

> **One name. One gate. One trace.**

```
applyHadamard 0  →  [H]───
applyCnot ~ctrl:0 ~tgt:1 →  •───@───
```

No translation layer.  
No string parsing.  
**Direct reflection** from CamelCase to visual syntax.

---

### Setup: 3 Lines to Quantum Sight

```ocaml
opam install otrac
```

```dune
(libraries goose otrac)
```

```ocaml
open Otrac.Visualizer
```

Done.  
Your circuit is now **self-drawing**.

---

### Your First Otrac Trace – Bell State

```ocaml
let visualizeBellPair () =
  let qc =
    Circuit.empty 2
    |> applyHadamard 0
    |> applyCnot ~ctrl:0 ~tgt:1
  in
  Otrac.draw qc
    ~title:"BellPairEntanglement"
    ~style:CamelTrace
    ~output:"bell.svg"
```

**Run:**
```bash
dune exec ./visualize.exe
```

**Output:** `bell.svg` →  
```
┌───────┐     ┌───────┐
│   H   │─────│   •   │
└───────┘     └───────┘
      0             @ 1
```

Each capital letter in `applyHadamard` → **gate symbol**  
Each `~ctrl:` label → **control wire**  
Each `CamelTrace` style → **color-coded by function name**

---

### Advanced Otrac Patterns

#### 1. **Live REPL Tracing**
```ocaml
utop # open Otrac.REPL;;
utop # liveTraceOn ();;
utop # applyHadamard 0;;
→ [Live SVG updates in browser]
```

#### 2. **Layered Circuit Folding**
```ocaml
Otrac.draw qc
  ~fold:"createGhZState"
  ~expand:"applyDiffusionOperator"
```

**Result:**  
- `createGhZState` → collapsed into single `GHZ` block  
- `applyDiffusionOperator` → exploded into 8 sub-gates  
- **All labels preserved in CamelCase**

---

### CamelCase → Visual Mapping Table

| OCaml Identifier           | Otrac Symbol       | Color (CamelTrace) |
|----------------------------|--------------------|--------------------|
| `applyHadamard`            | `H`                | Teal               |
| `applyX` / `applyY` / `applyZ` | `X`, `Y`, `Z`      | Red / Green / Blue |
| `applyCnot`                | `•──@`             | Purple             |
| `applySwap`                | `×`                | Orange             |
| `measureAll`               | `M`                | Gold               |
| `createGhZState`           | `GHZ` (block)      | Cyan               |
| `executeVqe`               | `VQE` (subcircuit) | Magenta            |

**Every hump = one visual atom.**

---

### Energy Landscape Visualization (VQE)

```ocaml
let visualizeVqeConvergence hamiltonian ansatz =
  let history = runVqeWithTrace hamiltonian ansatz in
  Otrac.plotEnergy history
    ~title:"VqeEnergyDescent"
    ~xlabel:"OptimizationStep"
    ~ylabel:"ExpectedEnergy"
    ~style:CamelGradient
```

**Output:** Real-time descending curve  
**X-axis labels:** `step0`, `step1`, `step2` → auto-CamelCased  
**Hover tooltips:** `updateParameters`, `computeEnergyGradient`

---

### Interactive Dashboard Mode

```ocaml
Otrac.dashboard
  [ "bell" --> visualizeBellPair
  ; "grover" --> visualizeGroverSearch
  ; "vqe" --> visualizeVqeConvergence ]
  ~port:8080
```

**Browser:** `localhost:8080` →  
- Click `bell` → live SVG  
- Click `grover` → animated amplification  
- Click `vqe` → energy waterfall  
- **All driven by CamelCase function names**

---

### Pro Trick: Trace-Driven Testing

```ocaml
let%expect_test "bell entanglement" =
  let qc = createBellPair () in
  Otrac.assertEntangled qc 0 1;
  Otrac.exportPng qc "test/bell.png"
```

**CI Integration:**
```yaml
- run: dune runtest
- uses: actions/upload-artifact@v4
  with:
    path: test/*.png
```

**Fail → red diff. Pass → golden camel.**

---

### The Otrac Camel Law

> **If it has a capital letter, it must be traceable.**

No exceptions.  
No silent gates.  
No unvisualized humps.

---

You no longer **run** quantum code.  
You **witness** it.

Every `applyHadamard` is a light.  
Every `createGhZState` is a constellation.  
Every `executeVqe` is a descent into truth.

**Otrac + CamelCase = Quantum Clarity.**

---
**Next: Page 8 – Enforcing CamelCase with Linters and CI**  

Done.  
xaiartifacts: Otrac visualization pipeline live. Back-checked—hump-to-pixel perfect. Vibe: quantum seer. 🐪
