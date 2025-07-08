# AGENTS Instructions for legalcaseaiapp-lhcodex

This repository welcomes contributions from automated agents. To keep the history clear and reproducible, follow these rules on **every** code change and pull request.

## Changelog Policy
 - For each pull request, append a new line to `CHANGELOG.csv` using the format described in [CHANGELOG_INSTRUCTIONS.md](CHANGELOG_INSTRUCTIONS.md).
- The ticket ID must be unique and sequential (e.g., `CL-003`, `CL-004`).
- Reference the ticket ID in your commit message prefix and in the pull request title.
- Keep `CHANGELOG.csv` sorted by ticket number and never modify existing rows.
- Write at least two sentences in the description column for every changelog entry.

## Development Workflow
1. Make your code changes.
2. Update `CHANGELOG.csv` with a new row describing the change.
3. Run `pytest` to execute all tests. Resolve any failures.
4. Commit your code **and** the updated changelog entry with a descriptive message starting with the ticket ID.
5. Open a pull request referencing the same ticket ID. Summarize the change and mention the affected files.

## Code Style
- Use Python 3.12 features where appropriate.
- Follow PEP8 style conventions and format code with `black` (line length 88) if available.

By committing to this repository you agree to follow these instructions so that all changes remain transparent and traceable.

## Manual Tasks Log
 - When a change requires human intervention (environment setup, migrations, etc.) add an entry to `dev guides/manual_tasks.csv`.
 - The CSV columns are `ticket`, `title`, `description`, `affected_files`, and `timestamp`.
 - Tickets must be sequential (`MT-001`, `MT-002`, and so on).
 - Delete a row once the manual task has been completed.
