# NeuralBlitz Security Policy

## Our Philosophy

The security of the NeuralBlitz framework is paramount. Our commitment to safety is a direct extension of the core axioms defined in our **[Charter](/scriptorium/governance/CharterLayer.md)**, particularly Axiom I (The Flourishing Objective) and Axiom II (Radical Transparency). We aim to build a system that is not only intelligent but also resilient, secure, and trustworthy.

We deeply value the work of security researchers and believe that a strong, collaborative relationship with the community is essential to achieving our security goals.

## Supported Versions

Only the latest major version of NeuralBlitz receives active security support. Please ensure you are using the most up-to-date release.

| Version | Supported          |
| ------- | ------------------ |
| 14.x    | :white_check_mark: |
| < 14.0  | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you believe you have discovered a security vulnerability, please report it to us privately. This gives us the opportunity to investigate and address the issue without putting the project or its users at risk.

**To report a vulnerability, please email us at:**

`security-veritas@neuralblitz.ai`

### What to Include in Your Report

To help us resolve the issue as quickly as possible, please include the following in your report:

*   **A clear description** of the vulnerability and its potential impact.
*   **The component or file** where the vulnerability exists.
*   **Step-by-step instructions** to reproduce the issue.
*   **Any proof-of-concept code,** scripts, or screenshots that can help us understand the vulnerability.

We will do our best to acknowledge your report within 48 hours.

## Disclosure Process

1.  **Acknowledgment:** Once we receive your report, we will send an acknowledgment confirming that we have received it and have begun our investigation.
2.  **Investigation:** Our internal "Guardian Team" will work to verify the vulnerability. We may contact you for additional information during this stage.
3.  **Remediation:** We will develop a patch to address the vulnerability. This will be done in a private fork.
4.  **Coordinated Disclosure:** Once the patch is ready, we will coordinate a release date with you. We will issue a public security advisory, which will be cryptographically sealed with a `GoldenDAG` to ensure its authenticity.
5.  **Recognition:** With your permission, we will publicly credit you for your discovery in the security advisory. We believe in recognizing the community's efforts to keep NeuralBlitz secure.

## Scope

This policy applies to the code and infrastructure of the official NeuralBlitz repository. The following are considered out of scope:

*   Vulnerabilities in third-party dependencies (please report those to the respective projects).
*   Social engineering or phishing attacks.
*   Denial of Service (DoS) attacks.
*   Physical security of data centers.

Thank you for helping us keep NeuralBlitz safe and secure.
