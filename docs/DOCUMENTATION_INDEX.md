# Documentation and Plans Index

**Purpose**: Master index of all documentation, plans, and runbooks in the repository.
**Last updated**: May 16, 2026

---

## 📚 **Documentation Structure**

### **🎯 Active Plans & Runbooks**
| Document | Status | Location | Purpose |
|----------|--------|----------|---------|
| [Refactor Plan](docs/REFACTOR_PLAN.md) | 📋 Active | docs/ | In-place architecture refactor |
| [Target Architecture](docs/ARCHITECTURE.md) | 📋 Active | docs/ | Refactor "definition of done" |
| [2D Decomposition Plan](docs/AI_interaction/2d_decomposition_plan.md) | 📋 Active | docs/AI_interaction/ | Scientific analysis plan |
| [Concat-Sort Monitoring](docs/AI_interaction/concat-sort/MONITORING_GUIDE.md) | ✅ Complete | docs/AI_interaction/concat-sort/ | Pipeline monitoring |
| [UnitMatch Integration](docs/AI_interaction/concat-sort/plan_unitmatch.md) | 📋 Planned | docs/AI_interaction/concat-sort/ | Cross-session tracking |
| [Subject Data Audit](/.claude/plans/subject_data_audit_and_transfer.md) | 📋 Active | .claude/plans/ | Multi-subject data management |

### **🔬 Analysis Documentation**
| Document | Purpose | Location |
|----------|---------|----------|
| [Analysis Manual](docs/ANALYSIS_MANUAL.md) | Complete workflow guide | docs/ |
| [Running Guide](RUNNING.md) | Step-by-step execution | project root |
| [Normalization Audit & Fixes](docs/NORMALIZATION.md) | March 2026 audit + fix history | docs/ |
| [Figure Notes](analysis_suite/figures/07_advanced/) | Advanced analysis notes | analysis_suite/figures/ |

### **🤖 AI Assistant Configuration**
| Document | Purpose | Location |
|----------|---------|----------|
| [Agent Instructions](docs/AI_interaction/PROMPT.md) | Main assistant prompt | docs/AI_interaction/ |
| [Agent Roles](docs/AI_interaction/AGENTS.md) | Specialized agent definitions | docs/AI_interaction/ |
| [Copilot Instructions](docs/AI_interaction/copilot-instructions.md) | **Canonical** copilot config | docs/AI_interaction/ |
| [Skills](/.claude/skills/) | Agent skill definitions | .claude/skills/ |

### **📖 User Documentation**
| Document | Purpose | Location |
|----------|---------|----------|
| [Main README](README.md) | Project overview & quick start | project root |
| [Analysis Suite Guide](analysis_suite/README.md) | Figure generation workflow | analysis_suite/ |
| [Package Documentation](src/visdetect/README.md) | Library API reference | src/visdetect/ |

---

## 🗂️ **Documentation Categories**

### **Plans & Engineering**
- **Location**: `docs/AI_interaction/` (AI-driven plans) and `.claude/plans/` (operational)
- **Purpose**: Implementation plans, audits, feasibility studies
- **Examples**: Codebase reorganization, concat-sort planning, 2D decomposition

### **Runbooks & Monitoring**
- **Location**: `docs/AI_interaction/concat-sort/` and `docs/manuals/`
- **Purpose**: Operational procedures, monitoring guides, troubleshooting
- **Examples**: Pipeline monitoring, data QC procedures, manual workflows

### **Analysis Documentation**
- **Location**: `analysis_suite/` and `docs/`
- **Purpose**: Scientific methodology, bug fixes, analysis notes
- **Examples**: Normalization audits, figure explanations, method validation

### **Developer Documentation**
- **Location**: `project root`, `src/`, and subdirectories
- **Purpose**: Setup instructions, API docs, package documentation
- **Examples**: README files, installation guides, code documentation

---

## 🔧 **Cleanup Actions Taken**

### **✅ Consolidated**
- Copilot instructions: `docs/AI_interaction/copilot-instructions.md` is **canonical**
- Reorganization docs: superseded by `docs/REFACTOR_PLAN.md`; the 3 March 2026 docs archived to `archive/reorganization_docs_2026/`
- Normalization docs: 4 March 2026 reports consolidated into `docs/NORMALIZATION.md`
- Concat-sort plans: All in `docs/AI_interaction/concat-sort/`

### **✅ Organized**
- Analysis-specific docs remain in `analysis_suite/`
- User docs remain in appropriate locations (`README.md`, `RUNNING.md`)
- AI assistant configs centralized in `docs/AI_interaction/`

### **📋 Recommendations**

1. **Move operational plans**: Consider moving `.claude/plans/` → `docs/operations/`
2. **Link cross-references**: Add links between related documents
3. **Archive completed plans**: Move completed reorganization docs to `archive/`
4. **Create topic indexes**: Add indexes for concat-sort, normalization, etc.

---

## 🔍 **Finding Documents**

### **By Purpose**
- **Setting up the project**: README.md → RUNNING.md → analysis_suite/README.md
- **Running analysis**: CLAUDE.md → docs/ANALYSIS_MANUAL.md → analysis_suite/
- **Monitoring pipelines**: docs/AI_interaction/concat-sort/MONITORING_GUIDE.md
- **Understanding fixes**: docs/NORMALIZATION.md

### **By Status**
- **Active plans**: `docs/REFACTOR_PLAN.md`, `docs/AI_interaction/2d_decomposition_plan.md`
- **Completed projects**: `archive/reorganization_docs_2026/` (2026 reorganization, superseded)
- **Operational guides**: `docs/AI_interaction/concat-sort/MONITORING_GUIDE.md`

---

*This index helps navigate the comprehensive documentation across the repository while maintaining logical organization by purpose and audience.*