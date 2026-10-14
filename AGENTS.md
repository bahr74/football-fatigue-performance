# Codex instructions

Read and follow [PROJECT_RULES.md](PROJECT_RULES.md), the shared authority for scientific integrity, engineering, data handling, and authorization boundaries. Consult [README.md](README.md), [docs/architecture.md](docs/architecture.md), and [docs/data_policy.md](docs/data_policy.md) for context. Do not duplicate or silently weaken the shared rules.

Inspect the workspace and Git status before edits. Work only on the user's authorized milestone. Preserve unrelated work and report the checks actually executed. Do not assume data exists.

Use Python 3.12. On this workstation:

```powershell
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -m pip install --no-cache-dir -e ".[dev]"
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -m pytest
```

The environment is outside the repository; filesystem approval may be required for changes to it. No remote operations are authorized by these instructions. Read any more specific instructions encountered before editing their scope.
