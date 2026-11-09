"""Collect the real tool definitions of popular MCP servers by starting each one and calling tools/list.

The servers are installed in /opt/mcp-harvest (npm --ignore-scripts, go install, pip) and started as the
unprivileged `agentbox` user with an empty environment plus dummy credentials, so nothing here can reach a
real account. Only the tool list is used; the servers are never asked to run a tool.

    .venv/bin/python harness/harvest_real_tools.py      # writes harness/real_catalog/<server>.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

H = "/opt/mcp-harvest"
NB = f"{H}/node_modules/.bin"
OUT = Path(__file__).resolve().parent / "real_catalog"

SERVERS = {
    "github": ([f"{H}/bin/github-mcp-server", "stdio", "--toolsets", "all"], {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_dummy"}),
    "grafana": ([f"{H}/bin/mcp-grafana"], {"GRAFANA_URL": "http://127.0.0.1:9", "GRAFANA_SERVICE_ACCOUNT_TOKEN": "dummy"}),
    "sentry": ([f"{NB}/sentry-mcp", "--access-token=dummy"], {}),
    "notion": ([f"{NB}/notion-mcp-server"], {"NOTION_TOKEN": "dummy"}),
    "kubernetes": ([f"{NB}/mcp-server-kubernetes"], {}),
    "playwright": ([f"{NB}/playwright-mcp"], {}),
    "stripe": (["node", f"{H}/node_modules/@stripe/mcp/dist/index.js", "--tools=all", "--api-key=sk_test_dummy"], {}),
    "supabase": ([f"{NB}/mcp-server-supabase", "--access-token=dummy"], {}),
    "mongodb": ([f"{NB}/mongodb-mcp-server"], {}),
    "slack": ([f"{NB}/mcp-server-slack"], {"SLACK_BOT_TOKEN": "xoxb-dummy", "SLACK_TEAM_ID": "T000"}),
    "linear": ([f"{NB}/mcp-linear"], {"LINEAR_API_KEY": "lin_dummy"}),
    "postgres": ([f"{NB}/mcp-server-postgres", "postgresql://127.0.0.1:9/app"], {}),
    "git": ([f"{NB}/git-mcp-server"], {}),
    "atlassian": ([f"{H}/py/bin/mcp-atlassian"], {"JIRA_URL": "https://example.atlassian.net", "JIRA_USERNAME": "a@example.com",
                                                   "JIRA_API_TOKEN": "dummy", "CONFLUENCE_URL": "https://example.atlassian.net/wiki",
                                                   "CONFLUENCE_USERNAME": "a@example.com", "CONFLUENCE_API_TOKEN": "dummy"}),
}

# Extra servers for the catalog-size test (harness/real_catalog_extra/); not part of the main 613-tool catalog.
EXTRA = {
    "gitlab": ([f"{NB}/mcp-gitlab"], {"GITLAB_PERSONAL_ACCESS_TOKEN": "glpat-dummy", "USE_PIPELINE": "true", "USE_MILESTONE": "true", "USE_GITLAB_WIKI": "true"}),
    "circleci": ([f"{NB}/mcp-server-circleci"], {"CIRCLECI_TOKEN": "dummy"}),
    "hubspot": ([f"{NB}/mcp-hubspot"], {"PRIVATE_APP_ACCESS_TOKEN": "dummy"}),
    "heroku": ([f"{NB}/heroku-mcp-server"], {"HEROKU_API_KEY": "dummy"}),
    "airtable": ([f"{NB}/airtable-mcp-server"], {"AIRTABLE_API_KEY": "dummy"}),
    "elasticsearch": ([f"{NB}/mcp-server-elasticsearch"], {"ES_URL": "http://127.0.0.1:9", "ES_API_KEY": "dummy"}),
    "netlify": ([f"{NB}/netlify-mcp"], {}),
    "figma": ([f"{NB}/figma-developer-mcp", "--stdio"], {"FIGMA_API_KEY": "dummy"}),
    "browserbase": ([f"{NB}/mcp-server-browserbase"], {"BROWSERBASE_API_KEY": "dummy", "BROWSERBASE_PROJECT_ID": "dummy"}),
    "filesystem": ([f"{NB}/mcp-server-filesystem", "/srv/agentbox/home"], {}),
    "memory": ([f"{NB}/mcp-server-memory"], {}),
    "twilio": ([f"{NB}/twilio-mcp-server", "ACdummy/SKdummy:dummy"], {}),
    "salesforce": ([f"{NB}/sf-mcp-server", "--orgs", "DEFAULT_TARGET_ORG", "--toolsets", "all"], {}),
    "monday": ([f"{NB}/mcp-server-monday-api", "-t", "dummy"], {}),
    "docker": ([f"{NB}/mcp-server-docker"], {}),
    "terraform": ([f"{H}/bin/terraform-mcp-server", "stdio"], {}),
    "aws_ecs": ([f"{H}/py/bin/ecs-mcp-server"], {"AWS_REGION": "eu-west-1"}),
    "aws_eks": ([f"{H}/py/bin/awslabs.eks-mcp-server", "--allow-write"], {"AWS_REGION": "eu-west-1"}),
    "aws_cloudwatch": ([f"{H}/py/bin/awslabs.cloudwatch-mcp-server"], {"AWS_REGION": "eu-west-1"}),
    "clickhouse": ([f"{H}/py/bin/mcp-clickhouse"], {"CLICKHOUSE_HOST": "127.0.0.1"}),
}


async def harvest(name: str, cmd: list[str], env: dict) -> list[dict]:
    base = {"PATH": "/opt/node22/bin:/usr/bin:/bin", "HOME": "/srv/agentbox/home", **env}
    wrapped = ["-u", "agentbox", "--", "env", "-i", *[f"{k}={v}" for k, v in base.items()], *cmd]
    params = StdioServerParameters(command="runuser", args=wrapped, cwd="/srv/agentbox/home")
    async with stdio_client(params, errlog=open(OUT / f"{name}.stderr.txt", "w")) as (r, w):
        async with ClientSession(r, w) as s:
            await asyncio.wait_for(s.initialize(), 60)
            tools, cursor = [], None
            while True:
                res = await asyncio.wait_for(s.list_tools(cursor=cursor) if cursor else s.list_tools(), 60)
                tools += [t.model_dump(mode="json", exclude_none=True) for t in res.tools]
                cursor = res.next_cursor
                if not cursor:
                    return tools


async def main() -> None:
    global OUT
    extra = "--extra" in sys.argv
    only = set(a for a in sys.argv[1:] if not a.startswith("--"))
    if extra:
        OUT = OUT.parent / "real_catalog_extra"
    OUT.mkdir(exist_ok=True)
    for name, (cmd, env) in (EXTRA if extra else SERVERS).items():
        if only and name not in only:
            continue
        try:
            tools = await asyncio.wait_for(harvest(name, cmd, env), 120)
            (OUT / f"{name}.json").write_text(json.dumps(tools, indent=1))
            print(f"{name:11s} {len(tools):4d} tools")
        except BaseException as e:  # noqa: BLE001
            print(f"{name:11s} FAILED {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    asyncio.run(main())
