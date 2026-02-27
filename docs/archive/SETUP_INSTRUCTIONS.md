# Spektron — Setup Instructions

## Step 1: Extract the Project
Download `Spektron-project.tar.gz` from the Claude chat, then:
```bash
cd /Users/admin/Documents/GitHub
tar xzf ~/Downloads/Spektron-project.tar.gz
# This creates Spektron/ with all code, data, and docs
```

## Step 2: Set Up Python Environment
```bash
cd /Users/admin/Documents/GitHub/Spektron
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 3: Verify Setup
```bash
python tests/smoke_test.py
```
This runs 15 tests. Expect some failures initially — that's what Claude Code will fix.

## Step 4: Open in VS Code + Claude Code
```bash
code /Users/admin/Documents/GitHub/Spektron
```
Then in VS Code terminal:
```bash
claude
```
Claude Code will automatically read CLAUDE.md and understand the project.

## Step 5: First Prompt to Claude Code
```
Read CLAUDE.md, PROJECT_STATUS.md, and IMPLEMENTATION_PLAN.md. Then run 
tests/smoke_test.py and fix all failing tests. Work through the issues 
module by module.
```

## Key Files for Claude Code
- `CLAUDE.md` — Master project instructions (auto-read by Claude Code)
- `PROJECT_STATUS.md` — What works, what's broken, known bugs
- `IMPLEMENTATION_PLAN.md` — Detailed task list with phases
- `TECHNICAL_REFERENCE.md` — Quick reference for all architecture components
- `tests/smoke_test.py` — Test suite (fix until all pass)

## Key Files for Research Context
- `paper/ADVANCED_TECHNIQUES_SYNTHESIS.md` — All 10 ML techniques with citations
- `paper/BRAINSTORM_V2.md` — Brainstorm innovations and paper framing
- `paper/RESEARCH_FULL_REFERENCE.md` — Complete literature review
- `paper/RESEARCH_WORKING_DOC.md` — Running working document
