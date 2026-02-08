# Kalman Labs | Living Systems OS (LS-OS)

Living Systems OS (LS-OS) is a computational operating system for biology, built by **Kalman Labs**.

This repository is the shared workspace for:
- Research development (models, experiments, validation)
- Product development (runtime, APIs, tooling, platform layers)
- External communication assets (company and scientific narrative)

---

## 1. Mission

Biology is data-rich but model-poor.  
Kalman Labs is building LS-OS so biological systems can be:
- Executable
- Predictive over time
- Controllable under perturbations
- Reproducible and versioned

Core thesis: biology should be run like software, not only analyzed like static data.

---

## 2. Current Wedge

Our first wedge is tightly scoped:
- **Indication:** EGFR-mutant non-small cell lung cancer (NSCLC)
- **Treatment class:** EGFR tyrosine kinase inhibitors (TKIs)
- **Input focus:** bulk RNA-seq + longitudinal tumor burden
- **Target outputs:** treatment response trajectories + resistance emergence forecasts

MVP objective:
- Build an executable tumor dynamics model
- Train on real cohorts
- Benchmark against Cox and black-box deep learning baselines

---

## 3. Repository Scope

This repo will evolve into a multi-track workspace supporting:
- Scientific modeling and validation
- Data processing and cohort management
- Runtime and simulation infrastructure
- Programmatic interfaces (internal/external)
- Documentation for research, product, and strategy

This is intentional: research and engineering are co-developed here.

---

## 4. Working Model

We follow a research-to-product loop:
1. Define a bounded biological question
2. Build executable model assumptions
3. Train and validate on cohorts
4. Stress-test under perturbations and shifts
5. Convert validated methods into reusable platform components

Every major claim should map to:
- A defined cohort scope
- A validation protocol
- Reproducible experiment artifacts

---

## 5. Suggested Repository Layout

As the codebase grows, organize work into these top-level areas:

```text
.
├── research/            # hypotheses, methods, experiments, papers
├── data/                # dataset metadata, schemas, cohort definitions (no raw PHI)
├── models/              # model definitions, checkpoints metadata, configs
├── runtime/             # simulation and execution engine components
├── api/                 # service interfaces and integration surfaces
├── docs/                # scientific specs, architecture docs, decision records
├── experiments/         # run configs, results manifests, benchmark outputs
└── website/             # external communication assets (optional split later)
```

Notes:
- Keep sensitive/raw data out of git.
- Commit dataset descriptors, schemas, and provenance metadata only.

---

## 6. Research Standards

Minimum expectations for research work merged to main branches:
- Clear problem statement and biological scope
- Explicit assumptions and constraints
- Defined baseline comparisons
- Reproducible run instructions
- Quantitative evaluation summary

Recommended experiment metadata (per run):
- Run ID
- Date/time
- Cohort version
- Feature set
- Model version
- Training config hash
- Evaluation metrics
- Notes on failures/edge cases

---

## 7. Engineering Standards

- Keep modules small, explicit, and testable.
- Prefer deterministic pipelines where feasible.
- Document interface contracts before broad integration.
- Add tests for any behavior used in downstream research conclusions.
- Avoid silent changes to core assumptions.

For model/runtime changes, include:
- What changed
- Why it changed
- Expected impact on evaluation outcomes

---

## 8. Collaboration and Branching

Suggested branch naming:
- `research/<topic>`
- `model/<topic>`
- `runtime/<topic>`
- `docs/<topic>`
- `infra/<topic>`

Commit style:
- One coherent change per commit
- Message includes scope and intent

Pull request checklist:
- Problem and scope
- Validation or test evidence
- Risks and open questions
- Follow-up items

---

## 9. Data, Privacy, and Compliance

Principles:
- Least-privilege data handling
- No PHI in repository history
- Keep provenance and governance auditable
- Respect licensing and data-use restrictions for every dataset

Before using a dataset, record:
- Source
- Access terms
- Allowed use
- Retention constraints
- Attribution requirements

---

## 10. Roadmap (Living Document)

Near-term:
- Lock MVP cohort and baseline protocols
- Build first executable NSCLC response model
- Establish benchmark suite for trajectory prediction and resistance forecasting

Mid-term:
- Harden runtime abstraction for reusability
- Expand perturbation simulation workflows
- Introduce partner-facing outputs for trial support

Long-term:
- Generalize LS-OS beyond first oncology wedge
- Build a scalable developer platform on top of executable biology

---

## 11. How to Contribute

If you are adding research:
- Start with a concise hypothesis and scope
- Define evaluation criteria before training
- Include reproducibility notes

If you are adding engineering:
- Start from interface and dependency boundaries
- Add tests and migration notes if behavior changes
- Keep docs aligned with implementation

If you are adding documentation:
- Prioritize clear assumptions, decisions, and tradeoffs
- Link docs to concrete artifacts (runs, configs, modules)

---

## 12. Ownership

**Company:** Kalman Labs  
**Platform:** Living Systems OS (LS-OS)

For collaboration inquiries, use the project contact channels maintained by the Kalman Labs team.

