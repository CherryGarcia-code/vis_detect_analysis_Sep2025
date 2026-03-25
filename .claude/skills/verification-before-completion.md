---
name: verification-before-completion
description: Use when about to claim work is complete, fixed, or passing - requires running verification commands and confirming output before making any success claims; evidence before assertions always
---

# Verification Before Completion

## Overview

Claiming work is complete without verification is dishonesty, not efficiency.

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

## The Gate Function

```
BEFORE claiming any status or expressing satisfaction:

1. IDENTIFY: What command/check proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, inspect results
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim

Skip any step = lying, not verifying
```

## Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Script runs | Script output: exit 0 | "should work now" |
| Figure is correct | Inspect output figure, check axes/labels/data | Script ran without errors |
| Stats are valid | Check stats CSV, verify n/p/effect sizes | "computed correctly" |
| Bug fixed | Reproduce original symptom: resolved | Code changed, assumed fixed |
| Analysis complete | All outputs exist and look correct | "no errors in log" |
| Cache regenerated | Load cache, verify row counts and columns | Script finished |

## Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification ("Great!", "Perfect!", "Done!", etc.)
- About to commit without verification
- Relying on partial verification
- Thinking "just this once"
- **ANY wording implying success without having run verification**

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence != evidence |
| "Just this once" | No exceptions |
| "Script ran without errors" | No errors != correct output |
| "I changed the right line" | Verify the OUTPUT, not the code |
| "Partial check is enough" | Partial proves nothing |

## Key Patterns

**Analysis scripts:**
```
OK: [Run script] [See: figure saved, N rows cached] "Script complete, figure at path/fig.png"
BAD: "Should produce the correct figure now"
```

**Bug fixes:**
```
OK: [Run script that was failing] [See: no error, correct output] "Bug fixed, verified"
BAD: "Fixed the typo, should work now"
```

**Cache regeneration:**
```
OK: [Run producer] [Check: cache exists, row count matches] "Cache regenerated: N rows"
BAD: "Regenerated the cache"
```

**Downstream effects:**
```
OK: [Run downstream scripts] [See: all complete] "Downstream scripts verified"
BAD: "Upstream cache updated, downstream should be fine"
```

## When To Apply

**ALWAYS before:**
- ANY variation of success/completion claims
- ANY expression of satisfaction
- ANY positive statement about work state
- Committing changes
- Moving to next task
- Reporting results to user

## The Bottom Line

**No shortcuts for verification.**

Run the command. Read the output. Inspect the result. THEN claim the result.

This is non-negotiable.
