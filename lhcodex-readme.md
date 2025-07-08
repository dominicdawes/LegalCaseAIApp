The AT command has been deprecated. Please use schtasks.exe instead.

The request is not supported.

---

## 🌳 Two-branch workflow (main mirrors upstream, my-features carries your work)

**Goal**  
* `main` — always a pristine mirror of `dominicdawes/LegalCaseAIApp` (`upstream/main`).  
* `my-features` — your features + periodic **merge commits** from `main` (no history rewrites).

### 🔄 Daily refresh

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main     # main = exact mirror
git push origin main

git checkout my-features
git merge main                        # bring latest upstream into your branch
git push origin my-features
Resolve conflicts → git add <fixed> → git commit to finish the merge.

🚀 Opening a Pull Request
GitHub → New PR: base main ← compare my-features (inside your fork).

Merge (or squash) after review.

Run the Daily refresh again so main fast-forwards with the new work.

🧭 Quick reference
bash
Copy code
# abort an in-progress merge
git merge --abort

# see commits unique to your branch
git log main..my-features --oneline
