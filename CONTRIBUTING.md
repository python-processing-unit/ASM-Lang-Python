# Contributing to ASM-Lang

We welcome bug reports, feature requests (within reason), documentation improvements, tests, and code patches. This document explains how to contribute and the licensing terms that apply to contributions.

## TL;DR

- Open issues for bugs or feature requests.
- For code changes, open a pull request (PR) from a branch or fork.
- By contributing (issues, PRs, commits, patches, or other submissions) you agree to release your contributions under the Unlicense, dedicating them to the public domain. See "License for contributions" below.

## File of record

This repository is licensed under the Unlicense. See the repository `UNLICENSE` file for the full text:
https://unlicense.org/

## License for contributions (IMPORTANT)

By submitting any contribution to this repository (including commits, pull requests, issues with code/text, or other content), you confirm and agree that:

- You are the copyright owner of that contribution (or are authorized to submit it).
- You irrevocably dedicate and release your contribution under the Unlicense, placing it into the public domain worldwide to the fullest extent permitted by law.
- You grant the project maintainers and the public the right to use, copy, modify, publish, distribute, sublicense, and/or sell copies of your contribution without restriction and without requiring attribution.
- If you cannot agree to these terms, do not submit contributions to this repository.

## How to contribute

1. Search Issues
   - Check existing issues to see whether your bug or feature request already exists. If it does, add relevant information or a comment.

2. Open an Issue
   - If no issue exists, create one. Provide a clear title, a concise description of the problem or proposal, steps to reproduce (for bugs), and any relevant environment information.

3. Make a Change
   - Fork the repository (or create a branch in a branch-per-feature workflow if you have push rights).
   - Create a branch named meaningfully, e.g., `fix/typo-xyz`, `feat/asm-parser`, or `test/add-instruction-tests`.
   - Implement your change. Keep commits focused and atomic.

4. Tests and Quality
   - Add tests for bug fixes and features where applicable.
   - Make sure existing tests pass.
   - Run linters and formatters used by the project (if none are mandated, follow existing code style).

5. Commit Messages
   - Use clear, descriptive commit messages. Include the rationale if the change is non-obvious.

6. Open a Pull Request
   - Push your branch to your fork and open a pull request against this repository's main branch (or other branch named in the repo workflow).
   - Describe what you changed and why. Link related issues with `#issue-number`.
   - If your PR includes substantial changes, list the testing steps you used.

7. Review Process
   - Maintainers will review PRs and may request changes. Please respond to feedback in a timely manner.
   - Once approved, a maintainer will merge. You may be asked to rebase or squash commits before merge.

## Non-code contributions

- Documentation fixes, examples, and small editorial changes can often be merged quickly.
- For larger documentation efforts, open an issue first to discuss scope.

## Code style

- For Python code, follow PEP 8 (https://peps.python.org/pep-0008/)
- For ASM-Lang code, NO USING TABS FOR INDENTS!

## Security

- Do not submit secrets, credentials, or other sensitive data.
- If you discover a major security vulnerability, please report it privately to the maintainers rather than opening a public issue.
- Minor security vulnerabilities (e.g. errors in the standard library) should be reported through a public issue.

## Contact / Questions

- If you’re unsure where to start, open an issue with a brief outline and ask for guidance.
- For legal or licensing questions, open an issue and tag the maintainers.

---