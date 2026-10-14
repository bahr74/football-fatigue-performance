# Claude Code instructions

Read and follow [PROJECT_RULES.md](PROJECT_RULES.md), the shared authority for scientific integrity, engineering, data handling, and authorization boundaries. Use [README.md](README.md), [docs/architecture.md](docs/architecture.md), and [docs/data_policy.md](docs/data_policy.md) for project context. Keep shared requirements in PROJECT_RULES.md rather than maintaining a separate copy here.

Inspect the workspace and Git status before edits, respect the active authorized milestone, and report only checks that actually ran. Do not assume datasets are present.

Use Python 3.12. On this workstation:

```powershell
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -m pip install --no-cache-dir -e ".[dev]"
& "F:\Football-Fatigue-Project.venv\Scripts\python.exe" -m pytest
```

The environment is outside the repository; filesystem approval may be required for changes to it. These instructions do not authorize remote operations.
