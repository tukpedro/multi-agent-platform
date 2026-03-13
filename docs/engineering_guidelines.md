# Engineering Guidelines

## Code Standards

### Python

- Use Python 3.11+ for all new projects
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use `ruff` for linting and formatting

### Git Workflow

- Use feature branches: `feature/TICKET-123-description`
- Write clear commit messages following conventional commits format
- All PRs require at least 2 approvals before merging
- Squash-merge to main branch
- Never push directly to main

### Testing

- Minimum 80% code coverage for all new code
- Write unit tests for all business logic
- Write integration tests for API endpoints
- Use `pytest` as the test framework
- Run tests before pushing: `pytest tests/ -v`

## Architecture Principles

### API Design

- Use REST conventions for all APIs
- Version APIs with URL prefix: `/api/v1/`
- Return consistent error response format
- Use pagination for list endpoints (max 100 items per page)

### Database

- Use migrations for all schema changes (Alembic)
- Add indexes for frequently queried columns
- Use connection pooling in production
- Never store secrets in the database

### Monitoring

- All services must expose a `/health` endpoint
- Log in structured JSON format
- Set up alerts for P99 latency > 500ms
- Use Datadog for metrics and APM

## Code Review Guidelines

- Review within 24 hours of PR submission
- Focus on correctness, readability, and maintainability
- Leave constructive, specific comments
- Approve when ready, request changes when blocking issues exist

## Deployment

- All deployments go through CI/CD pipeline
- Staging environment must pass all tests before production
- Use blue-green deployments for zero-downtime releases
- Rollback plan must be documented for each deployment
