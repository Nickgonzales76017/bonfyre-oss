# Bonfyre Pages Runtime

`bonfyre-oss` provides a reusable GitHub Actions workflow for Bonfyre Pages apps:

- workflow: `.github/workflows/pages-runtime.yml`
- purpose: build Bonfyre on the runner, detect changed `input/**` files, run an app-local processing script, upload artifacts, commit outputs, and deploy Pages

## Intended split

- `bonfyre-oss`
  - reusable workflow
  - shared runtime conventions
  - future shared frontend helpers
- app repo
  - domain-specific pipeline script
  - branding and UI
  - app-local templates and artifacts

## Inputs

- `app_name`
- `input_glob`
- `process_script`
- `deploy_path`
- `artifact_name`
- `bonfyre_repo`
- `bonfyre_ref`

## Required app-local script contract

The called repo should provide a script such as:

- `scripts/process-inputs.sh`

Environment variables provided by the runtime:

- `BONFYRE_APP_NAME`
- `CHANGED_FILES_FILE`
- `DEPLOY_PATH`

The script should:

1. read file paths from `$CHANGED_FILES_FILE`
2. write generated outputs into `artifacts/`
3. write Pages-ready output into `$DEPLOY_PATH`

The runtime then adds:

- `site/job.json`
- `site/summary.json`

and uploads/deploys the result.
