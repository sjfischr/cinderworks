---
inclusion: auto
---

# Git Workflow

After every completed code change (task completion, bugfix, feature implementation), automatically commit and push:

1. Stage all changed files related to the completed work
2. Write a concise, descriptive commit message summarizing what was done
3. Push to a feature branch (never directly to main/master)
4. Use `git push -u origin <branch>` when pushing a new branch for the first time

This applies to spec task execution, ad-hoc code changes, and any other modifications to the codebase.
