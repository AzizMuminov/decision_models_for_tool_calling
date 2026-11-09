# batch-importer

Data team batch job. Runs on AWS EKS cluster `data-prod`, namespace `batch` (not on the prod-eu-1 Kubernetes cluster).

## Where things live (engineering handbook)
- Code and reviews: GitHub, org `acme`. A few legacy services are still on GitLab (group `acme`) with GitLab CI; their README says so.
- CI: GitHub Actions (CircleCI was retired in 2025).
- Errors: Sentry, org `acme` (one project per service).
- Metrics, logs, alerts, incidents, on-call: Grafana (Prometheus, Loki, Grafana Incident, Grafana OnCall). Legacy AWS services log to CloudWatch (region eu-west-1).
- Tickets: Payments uses Jira (project PAY); Engineering uses Linear (team ENG). Design docs are in Confluence; runbooks in Notion.
- Chat: Slack. Incident updates go to #incidents.
- Production: Kubernetes (cluster prod-eu-1). The data team's batch jobs run on AWS EKS (cluster `data-prod`).
- Data: MongoDB (database `shop`), Postgres (payments, read-only).
