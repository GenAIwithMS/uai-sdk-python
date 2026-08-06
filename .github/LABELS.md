# Issue Labels

This document describes the label system used to triage issues and PRs.

## Label Set

| Label               | Emoji | Description                                              |
|---------------------|-------|----------------------------------------------------------|
| `roadmap-aligned`   | 🗺️    | Features we want to implement. On the planned roadmap.   |
| `needs-discussion`  | 💬    | Open for debate. Requires community/contributor input.   |
| `won't-implement`   | 🚫    | Clearly not aligned with the project's direction.        |
| `help-wanted`       | 🆘    | We want contributions here. Good for beginners.          |
| `good-first-issue`  | 👶    | Beginner-friendly tasks. Implies help-wanted.           |
| `bug`               | 🐛    | Confirmed or suspected bugs.                            |
| `enhancement`       | ✨    | New feature requests.                                   |
| `breaking`          | 💥    | Requires changes to major version.                      |
| `question`          | ❓    | Usage questions, support requests.                      |
| `documentation`     | 📝    | Docs updates needed.                                    |
| `performance`       | ⚡    | Performance-related issues.                             |
| `security`          | 🔒    | Security-related issues.                               |
| `testing`           | 🧪    | Test coverage or test infrastructure.                   |

## When to Use Each Label

- **`roadmap-aligned`** — The feature aligns with our planned phases (MVP → Beta → v1.0 → v2.0). We will implement it ourselves on schedule.

- **`needs-discussion`** — The value or approach is unclear. We need to discuss with maintainers and the community before deciding.

- **`won't-implement`** — The request does not fit the project scope (e.g. TypeScript SDK features in this Python repo, or features explicitly excluded by the SRS).

- **`help-wanted`** — We have decided to implement this but have limited bandwidth. External contributions are explicitly welcome.

- **`good-first-issue`** — Subset of `help-wanted`. Ideal for first-time contributors. Minimal domain knowledge needed.

## Creating Labels on GitHub

Labels can be created or applied manually in GitHub, or via the GitHub CLI:

```bash
# Create labels (run once per repo)
gh label create "roadmap-aligned" --description "Features we want to implement on the planned roadmap" --color "5E81AC"
gh label create "needs-discussion" --description "Open for debate, requires discussion" --color "B48EAD"
gh label create "won't-implement" --description "Clearly not aligned with project direction" --color "EBCBEB"
gh label create "help-wanted" --description="We want contributions here" --color "50C878"
gh label create "good-first-issue" --description "Beginner-friendly tasks" --color "7057DE"
```
