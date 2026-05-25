# Git remotes — team repo (M2 + M3)

You work as a **collaborator** on the milestone-2 repo (not your personal fork):

| Remote | URL | Role |
|--------|-----|------|
| **origin** | https://github.com/Sooush/CS349D-miniengine-sk-nc-ms2 | Team repo — **push M3 here** ([repo](https://github.com/Sooush/CS349D-miniengine-sk-nc-ms2)) |
| **upstream** | https://github.com/stanford-mast/CS349D-miniengine | Official course starter |

Personal fork (optional, only M1 there): [naochien/CS349D-miniengine-nc-sk](https://github.com/naochien/CS349D-miniengine-nc-sk) — you cannot create a second fork of the course repo on GitHub; use the Sooush repo for M2/M3.

Local history already includes a merge from `CS349D-miniengine-sk-nc-ms2` (see `git log`).

---

## Push milestone 3

```bash
cd /path/to/CS349D-miniengine
git remote -v

git add -A
git commit -m "Milestone 3: chunked prefill + radix prefix cache"
git push -u origin main
```

You need **write access** on Sooush’s repo (collaborator invite accepted). If push fails with 403, ask Sooush to confirm your collaborator role or use a PR from your fork.

---

## If you must use a PR (no direct push)

```bash
git remote add naochien https://github.com/naochien/CS349D-miniengine-nc-sk.git 2>/dev/null || true
git push naochien main
# Open PR: naochien/CS349D-miniengine-nc-sk → Sooush/CS349D-miniengine-sk-nc-ms2
```

---

## Pull course updates

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

---

## GCP VM

```bash
git clone https://github.com/Sooush/CS349D-miniengine-sk-nc-ms2.git
cd CS349D-miniengine-sk-nc-ms2
pip install -e ".[bench]"
```
