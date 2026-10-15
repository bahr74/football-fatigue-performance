# Claude Code instructions

Claude Code is the primary implementation assistant. [PROJECT_RULES.md](PROJECT_RULES.md) is the authority for scientific integrity, engineering, data handling, Git, and milestones; read and follow it. Use [README.md](README.md), [docs/architecture.md](docs/architecture.md), and [docs/data_policy.md](docs/data_policy.md) for context. Do not copy shared rules into this file.

## Working agreement

- Before editing, inspect the relevant files and `git status`. Do not assume datasets exist or that their schemas are known.
- Respect the active authorized milestone; do not start later roadmap steps. For complex tasks, give a short plan first.
- Implement incrementally with the smallest correct change, following existing conventions. No unrequested refactors, dependencies, or empty modules.
- Never modify `data/raw/`. Treat dataset statements as unverified until the file was actually inspected.
- Run focused tests after changes; report only checks that actually ran, plus changes and remaining issues.
- Git: use read-only commands only. The user performs all state-changing Git operations (add, commit, push, pull, merge, branch create/switch). No remote or GitHub actions.
- Ask before destructive operations or installing packages.

## Environment

Python 3.12, virtual environment at `.venv` in the repository:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```
