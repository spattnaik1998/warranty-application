# Security policy

## Supported version

Security fixes are applied to the latest Warrant release.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting feature for this repository.
Do not open a public issue containing exploit details, credentials, or sensitive
repository data.

Include the affected version, reproduction steps, impact, and any suggested
mitigation. Maintainers should acknowledge a report within seven days.

## Trust boundary

Static validation is the default. `warrant validate --allow-exec` runs commands
declared by the target repository and therefore must be used only on trusted
source. Process timeouts and environment filtering are safety controls, not an
operating-system sandbox. The FastAPI endpoint never executes uploaded code.
