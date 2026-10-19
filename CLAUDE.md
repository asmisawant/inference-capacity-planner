# Notes for Claude

## Git commits

- Author every commit as the repo owner, not as Claude. Before committing, run:

  ```bash
  git config user.name "asmisawant"
  git config user.email "107591325+asmisawant@users.noreply.github.com"
  ```

- Don't add `Co-Authored-By: Claude …` or `Claude-Session: …` trailers to commit messages.
- Commits from the bench workflow (`github-actions[bot]`, "bench: results from run …") stay as the bot.

## Numbers

Every number in the README must come from something we ran or derived. README tables live
between `GENERATED` markers and are written by `python -m capacity_planner.report --update-readme`;
`tests/test_readme.py` fails if they drift from the code or from `results/`.

## GPU runs

GPU benchmarks run on Modal via `.github/workflows/bench.yml`, never from the dev sandbox
(Modal's CLI can't connect from cloud sessions). Add runs to `bench/requests.json` with a new
`id`; the workflow's budget guard caps total spend at `budget_usd`.
