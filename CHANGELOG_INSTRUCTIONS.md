# Change Log Update Guide

This repository tracks updates in `CHANGELOG.csv`. Each row is a JIRA-style entry with detailed context.

## CSV Columns
- **Ticket**: Unique identifier, e.g., `CL-###`.
- **Date**: Date of change in `YYYY-MM-DD` format.
- **Time**: Time of change (24h) in `HH:MM` format.
- **Author**: Person making the change.
- **Description**: Rich text summary of the change. Use at least two sentences
  to fully explain each update.
- **Files Affected**: Key files or paths modified.
- **Notes**: Any additional comments or context.

## How to Add an Entry
1. Append a new line to `CHANGELOG.csv`.
2. Use the next sequential ticket ID (`CL-002`, `CL-003`, etc.).
3. Fill out each column. Use quotes if your text contains commas.
4. Commit the updated `CHANGELOG.csv` with a descriptive message.

Example entry:
```
CL-002,2025-06-18,09:30,LinDon Harris,"Updated README with setup steps. It now includes environment variable configuration.","README.md",""
```

Keep the log sorted by ticket number to maintain history.
