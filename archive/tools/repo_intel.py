#!/usr/bin/env python3
"""
repo_intel.py — Expert repo intelligence for the repo you're *currently in*.

What it does:
- Local git: repo root, branch, HEAD, dirty counts, upstream ahead/behind
- GitHub (if origin points to github.com): visibility, default branch, topics,
  languages (by bytes), stars/forks/watchers, open issues/PRs, latest release,
  and latest Actions workflow runs.

Setup:
- Recommended: set GITHUB_TOKEN env var (fine-grained token with read access is enough).
  PowerShell:
    setx GITHUB_TOKEN "github_pat_...."
  Restart terminal after setx.

Usage:
  python tools/repo_intel.py
  python tools/repo_intel.py --json
  python tools/repo_intel.py --no-remote
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from typing import Any, Dict, Optional, Tuple


# ---------------------------
# Local git helpers
# ---------------------------

def run_git(args: list[str], cwd: str) -> Tuple[int, str, str]:
    p = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def repo_root(start: str) -> str:
    code, out, err = run_git(["rev-parse", "--show-toplevel"], start)
    if code != 0 or not out:
        raise RuntimeError(err or "Not a git repo (or git not installed).")
    return out


def current_branch(root: str) -> str:
    code, out, _ = run_git(["branch", "--show-current"], root)
    return out if code == 0 and out else "(detached HEAD)"


def head_sha(root: str) -> str:
    code, out, _ = run_git(["rev-parse", "HEAD"], root)
    return out if code == 0 else ""


def status_counts(root: str) -> Dict[str, int]:
    """
    Parse porcelain status to count:
    - staged: index changes
    - modified: working tree changes
    - untracked: new files
    """
    code, out, _ = run_git(["status", "--porcelain"], root)
    if code != 0 or not out:
        return {"staged": 0, "modified": 0, "untracked": 0}

    staged = modified = untracked = 0
    for line in out.splitlines():
        if not line:
            continue
        if line.startswith("??"):
            untracked += 1
            continue
        x, y = line[0], line[1]
        if x != " ":
            staged += 1
        if y != " ":
            modified += 1
    return {"staged": staged, "modified": modified, "untracked": untracked}


def upstream_divergence(root: str) -> Dict[str, Any]:
    code, upstream, _ = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], root)
    if code != 0 or not upstream:
        return {"upstream": None, "ahead": None, "behind": None}

    code2, counts, _ = run_git(["rev-list", "--left-right", "--count", "HEAD...@{u}"], root)
    if code2 != 0 or not counts:
        return {"upstream": upstream, "ahead": None, "behind": None}

    try:
        ahead_s, behind_s = counts.split()
        return {"upstream": upstream, "ahead": int(ahead_s), "behind": int(behind_s)}
    except Exception:
        return {"upstream": upstream, "ahead": None, "behind": None}


def remotes(root: str) -> Dict[str, str]:
    code, out, _ = run_git(["remote", "-v"], root)
    if code != 0 or not out:
        return {}
    r: Dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "(fetch)":
            r[parts[0]] = parts[1]
    return r


# ---------------------------
# GitHub parsing + API
# ---------------------------

RE_GH_SSH = re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")
RE_GH_HTTPS = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")


def parse_github_slug(url: str) -> Optional[Tuple[str, str]]:
    u = url.strip()
    m = RE_GH_SSH.match(u)
    if m:
        return m.group("owner"), m.group("repo")
    m = RE_GH_HTTPS.match(u)
    if m:
        return m.group("owner"), m.group("repo")
    return None


def http_json(url: str, token: Optional[str], method: str = "GET") -> Dict[str, Any]:
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "repo-intel-py")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def graphql(query: str, variables: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
    if not token:
        # GraphQL generally expects auth; keep behavior explicit.
        raise RuntimeError("GraphQL requires GITHUB_TOKEN (recommended).")

    url = "https://api.github.com/graphql"
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "repo-intel-py")
    req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


REPO_QUERY = """
query RepoIntel($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    url
    isPrivate
    visibility
    description
    homepageUrl
    defaultBranchRef { name }
    primaryLanguage { name }
    stargazerCount
    forkCount
    watchers { totalCount }
    issues(states: OPEN) { totalCount }
    pullRequests(states: OPEN) { totalCount }
    licenseInfo { spdxId }
    isArchived
    pushedAt
    repositoryTopics(first: 20) {
      nodes { topic { name } }
    }
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      edges { size node { name } }
    }
    releases(first: 1, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes { name tagName publishedAt url }
    }
  }
  rateLimit {
    limit
    used
    remaining
    resetAt
  }
}
"""


def to_compact_languages(edges: list[dict]) -> list[dict]:
    return [{"lang": e["node"]["name"], "bytes": e["size"]} for e in edges]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=".", help="Path inside the repo (default: .)")
    ap.add_argument("--remote", default="origin", help="Remote name to use (default: origin)")
    ap.add_argument("--no-remote", action="store_true", help="Skip GitHub API calls")
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    start = os.path.abspath(args.path)

    try:
        root = repo_root(start)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    local = {
        "repo_root": root,
        "branch": current_branch(root),
        "head": head_sha(root),
        "status": status_counts(root),
        "upstream": upstream_divergence(root),
        "remotes": remotes(root),
    }

    origin_url = local["remotes"].get(args.remote) or next(iter(local["remotes"].values()), None)
    gh_slug = parse_github_slug(origin_url) if origin_url else None

    report: Dict[str, Any] = {"local": local, "github": None, "actions": None}

    if (not args.no_remote) and gh_slug:
        owner, name = gh_slug

        # GraphQL gives the best single-shot “repo intelligence”
        try:
            data = graphql(REPO_QUERY, {"owner": owner, "name": name}, token)
            repo = data["repository"]

            report["github"] = {
                "repo": repo["nameWithOwner"],
                "url": repo["url"],
                "visibility": repo["visibility"],
                "private": repo["isPrivate"],
                "default_branch": (repo["defaultBranchRef"] or {}).get("name"),
                "primary_language": (repo["primaryLanguage"] or {}).get("name"),
                "stars": repo["stargazerCount"],
                "forks": repo["forkCount"],
                "watchers": repo["watchers"]["totalCount"],
                "open_issues": repo["issues"]["totalCount"],
                "open_prs": repo["pullRequests"]["totalCount"],
                "license": (repo["licenseInfo"] or {}).get("spdxId"),
                "archived": repo["isArchived"],
                "pushed_at": repo["pushedAt"],
                "homepage": repo["homepageUrl"],
                "description": repo["description"],
                "topics": [n["topic"]["name"] for n in repo["repositoryTopics"]["nodes"]],
                "languages": to_compact_languages(repo["languages"]["edges"]),
                "latest_release": (repo["releases"]["nodes"][0] if repo["releases"]["nodes"] else None),
                "rate_limit": data["rateLimit"],
            }
        except Exception as e:
            report["github"] = {
                "repo": f"{owner}/{name}",
                "error": str(e),
                "hint": "Set GITHUB_TOKEN for GraphQL. Without it, run with --no-remote or use gh CLI.",
            }

        # Actions runs via REST (nice operational signal)
        # NOTE: this works without token for public repos, but token helps rate limits.
        try:
            runs_url = f"https://api.github.com/repos/{owner}/{name}/actions/runs?per_page=5"
            runs = http_json(runs_url, token)
            report["actions"] = [
                {
                    "name": r.get("name"),
                    "status": r.get("status"),
                    "conclusion": r.get("conclusion"),
                    "event": r.get("event"),
                    "branch": r.get("head_branch"),
                    "created_at": r.get("created_at"),
                    "url": r.get("html_url"),
                }
                for r in runs.get("workflow_runs", [])
            ]
        except Exception as e:
            report["actions"] = {"error": str(e)}

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    # Pretty print (expert summary)
    print("\nLocal")
    print("-----")
    print(f"Repo root     : {local['repo_root']}")
    print(f"Branch        : {local['branch']}")
    print(f"HEAD          : {(local['head'] or '')[:12]}")
    up = local["upstream"]
    print(f"Upstream      : {up['upstream']}")
    print(f"Ahead/Behind  : {up['ahead']}/{up['behind']}" if up["ahead"] is not None else "Ahead/Behind  : n/a")
    st = local["status"]
    print(f"Dirty         : staged={st['staged']} modified={st['modified']} untracked={st['untracked']}")
    if origin_url:
        print(f"Remote        : {origin_url}")

    if report["github"]:
        gh = report["github"]
        print("\nGitHub")
        print("------")
        if "error" in gh:
            print(f"Repo          : {gh.get('repo')}")
            print(f"Error         : {gh.get('error')}")
            print(f"Hint          : {gh.get('hint')}")
        else:
            print(f"Repo          : {gh['repo']}")
            print(f"Visibility    : {gh['visibility']}")
            print(f"Default       : {gh['default_branch']}")
            print(f"Stars/Forks   : {gh['stars']}/{gh['forks']}")
            print(f"Open Issues   : {gh['open_issues']}")
            print(f"Open PRs      : {gh['open_prs']}")
            print(f"Topics        : {', '.join(gh['topics'][:12])}" if gh["topics"] else "Topics        : (none)")
            if gh["languages"]:
                langs = ", ".join([f"{x['lang']}({x['bytes']})" for x in gh["languages"]])
                print(f"Languages     : {langs}")
            if gh["latest_release"]:
                lr = gh["latest_release"]
                print(f"Latest release: {lr.get('tagName')} ({lr.get('publishedAt')})")
            rl = gh["rate_limit"]
            print(f"RateLimit     : remaining={rl['remaining']} resetAt={rl['resetAt']}")

    if report["actions"]:
        print("\nActions (latest 5 runs)")
        print("----------------------")
        if isinstance(report["actions"], dict) and "error" in report["actions"]:
            print(f"Error: {report['actions']['error']}")
        else:
            for r in report["actions"]:
                print(f"- {r['name']} | {r['status']}/{r['conclusion']} | {r['branch']} | {r['created_at']}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
