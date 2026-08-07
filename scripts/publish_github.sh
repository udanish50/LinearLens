#!/usr/bin/env bash
set -euo pipefail

OWNER="${GITHUB_OWNER:-udanish50}"
REPO="${GITHUB_REPO:-LinearLens}"
VISIBILITY="${GITHUB_VISIBILITY:-public}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required" >&2
  exit 1
fi

LOGIN="$(gh api user --jq '.login')"
if [ "$LOGIN" != "$OWNER" ]; then
  echo "Authenticated GitHub account is '$LOGIN', expected '$OWNER'." >&2
  exit 1
fi

python scripts/validate_release.py

if [ ! -d .git ]; then
  git init
fi

git branch -M main
git add -A

if ! git diff --cached --quiet; then
  git commit -m "Initial open-source release of Linear Lens"
fi

if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  URL="https://github.com/$OWNER/$REPO.git"
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$URL"
  else
    git remote add origin "$URL"
  fi
  git push -u origin main
else
  gh repo create "$OWNER/$REPO" "--$VISIBILITY" --source=. --remote=origin --push
fi

gh repo edit "$OWNER/$REPO" --default-branch main >/dev/null
printf '\nUploaded: https://github.com/%s/%s\n' "$OWNER" "$REPO"
