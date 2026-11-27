# Security Policy — EgoSight XR

## Supported Versions

Security patches and dependency updates are actively applied to the following release lines:

| Version / Branch | Supported | Notes |
| :--- | :--- | :--- |
| `feature/full-dataset-intelligence` | Yes | Active development and benchmark pipeline |
| `main` | Yes | Production and release builds |
| `< 1.0.0` | No | Historical checkpoints and prototypes |

## Reporting a Vulnerability

The EgoSight XR development team takes software security, data integrity, and privacy seriously. If you discover a vulnerability or security flaw, please report it privately:

1. **GitHub Security Advisory:** Submit a private disclosure through GitHub Security Advisories under the repository's "Security" tab.
2. **Direct Contact:** Email the maintainer at `arhamchoudhary.dev@gmail.com` with the subject tag `[SECURITY] EgoSight XR Vulnerability Report`.

Please do not open public GitHub issues or discussions for undisclosed security vulnerabilities.

### Report Details

To help us investigate and triage effectively, please include:
- A clear description of the vulnerability and its potential impact.
- Affected components (backend API, frontend dashboard, model inference pipeline, dataset acquisition, or packaging scripts).
- Minimal reproducible steps or proof-of-concept (PoC).
- Any proposed remediation or mitigation steps.

## Response SLA and Coordination

- **Initial Acknowledgment:** Within 48 hours of receipt.
- **Triage & Assessment:** Within 5 business days, confirming severity, scope, and impact.
- **Remediation & Patch:** Security patches will be verified against our test suite and benchmark pipeline before release.
- **Coordinated Disclosure:** We adhere to responsible disclosure principles and will coordinate publication dates with reporters.

## Security Practices in EgoSight XR

- **Automated Dependency Updates:** Managed via GitHub Dependabot for Python packages (`pip`), frontend modules (`npm`), and CI/CD actions (`github-actions`).
- **Static Analysis & Security Scanning:** Automated CodeQL and Snyk security pipelines on every push and pull request.
- **Strict Data Validation:** All prediction payloads and manifest entries are validated against Track 4 schemas before acceptance.
- **Local Storage Isolation:** Machine learning checkpoints, downloaded media buffers, and staging artifacts run under restricted local file permissions.
