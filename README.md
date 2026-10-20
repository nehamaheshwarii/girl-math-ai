# girl-math-ai

A portfolio of small, runnable AI products — built with product discipline:
every project names its user and problem, ships with evals, and documents
its costs, risks, and limitations. Focus: **finance and product-building**.

## Projects
| # | Project | Capability it proves | Status |
|---|---------|---------------------|--------|
| 01 | [Chat with a 10-K](01_chat_with_10k/) | Agentic RAG with citations; "not found" instead of hallucinating | ✅ v1 built, tests passing |
| 02 | Earnings Call Analyst | Single agent, structured output | Planned — reuses 01's ingestion/retrieval |
| 03 | User Research Synthesizer | RAG over messy qualitative data | Planned |
| 04 | Investment Research Team | Multi-agent team with a risk/critic agent | Planned |
| 05 | PRD Critic / Launch Intelligence | Multi-agent + web research | Planned |
| 06 | Fraud Investigation Agent (synthetic data) | Tools + human-in-the-loop | Planned |
| 07 | Always-on Earnings Radar | Scheduled / always-on agents | Planned |
| 08 | Trust & Evals Kit | Evals, cost/latency, citation checking | Planned |

## Rules for every project in this repo
- Clone and run in under 5 minutes; works without paid API keys where possible
- Public or synthetic data only — never an employer's data
- README is a mini case study: problem, user, architecture, evals, cost, risks
