# Branch Management Strategy

## Purpose

`hongy_main` is the personal integration trunk. It records the complete
history of personal feature work and upstream synchronization without
rewriting its published history.

## Branch roles

### `master`

`master` is an upstream-only mirror of the contributor upstream trunk.

- Update it only from the upstream remote.
- Keep updates fast-forward-only.
- Do not develop features on it.
- Do not merge personal branches into it.

Typical update:

```bash
git fetch upstream
git switch master
git merge --ff-only upstream/master
```

### `hongy_main`

`hongy_main` is the long-lived personal integration trunk.

- Never rebase, reset, or amend its published history.
- Merge completed personal branches with `--no-ff`.
- Merge upstream `master` into it with `--no-ff`.
- Do not develop incomplete work directly on this branch.
- Do not force-push it.

Its first-parent history is the personal integration and release history.
The full DAG retains both feature and upstream history.

### `feat/*`

Feature branches contain one personal feature or fix.

- Create them from the current `hongy_main`.
- Keep feature commits on the feature branch until the work is complete.
- Merge completed features into `hongy_main` with `--no-ff`.
- A private, unshared feature branch may rebase onto `hongy_main`.
- A shared feature branch must not be rebased; merge `hongy_main` into it instead.
- Do not merge `master` separately into every feature branch.

## Required topology

The desired history contains explicit merge commits:

```text
feature:       F1---F2
                    \
hongy_main: ...---M1---M2
                   /       \
master:     ...---U1---U2---U3
```

`M1` merges a completed feature. `M2` merges upstream `master`.
The first parent of `M2` is the previous `hongy_main`; the second parent is
`master`.

Inspect the views as follows:

```bash
git log --first-parent --oneline hongy_main
git log --graph --oneline --decorate --all
```

## Feature integration workflow

Use a temporary integration branch so conflict resolution does not dirty the
long-lived trunk:

```bash
git switch hongy_main
git switch -c integrate/feat-<name>-YYYYMMDD

git merge --no-ff --no-commit feat/<name>

# Resolve conflicts, build, and run focused tests.
git commit

git switch hongy_main
git merge --ff-only integrate/feat-<name>-YYYYMMDD
```

The final fast-forward moves `hongy_main` to an existing non-fast-forward
merge commit. It does not flatten the feature history.

## Upstream synchronization workflow

First update the upstream mirror:

```bash
git fetch upstream
git switch master
git merge --ff-only upstream/master
```

Then integrate `master` through a temporary branch:

```bash
git switch hongy_main
git switch -c integrate/master-YYYYMMDD
git merge --no-ff --no-commit master

# Resolve conflicts, build, and run focused tests.
git commit

git switch hongy_main
git merge --ff-only integrate/master-YYYYMMDD
```

This creates a merge commit whose second parent is the current `master` HEAD.
Do not use `git rebase master` while on `hongy_main`.

## Verification and safety

Before every integration:

```bash
git status --short --branch
git log --graph --oneline --decorate --all
```

The worktree must be clean before switching or merging. After resolving
conflicts, run the smallest relevant build and regression checks first, then
run the required broader checks before advancing `hongy_main`.

Use short-lived integration branch names and keep rollback points when an
integration is risky:

```text
integrate/master-YYYYMMDD
integrate/feat-<name>-YYYYMMDD
safety/hongy_main-pre-master-YYYYMMDD
```

Keep the latest three to five safety points when practical. Delete old
integration branches only after the target merge has been verified.

## Remote policy

Use separate remote roles when possible:

- `upstream/master`: contributor upstream trunk.
- `origin/hongy_main`: personal remote trunk.

Push `master` only after a fast-forward update from upstream. Push
`hongy_main` only as a normal fast-forward update after local verification.
Never force-push `hongy_main`.

## Synchronization cadence

Check upstream every one to two weeks, before large feature work, and before
important releases. Synchronize sooner when upstream contains a required fix,
security update, or dependency change.

A difference such as `hongy_main` being many commits ahead and a few commits
behind `master` is expected. It is not a reason to rebase or flatten the DAG.
