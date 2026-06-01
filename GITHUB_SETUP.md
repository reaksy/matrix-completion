# GitHub setup

This project already has a local git repository on branch `main`.

## 1. Create an empty GitHub repository

Create a new repository on GitHub.

Recommended settings:

- Repository name: `matrix-completion-research`
- Visibility: private while the course project is still in progress
- Do not add README
- Do not add `.gitignore`
- Do not add license yet

## 2. Add the remote

Use SSH if your GitHub SSH key is configured:

```bash
cd "/Users/karimau/Projects_C++/Course Project"
git remote add origin git@github.com:<username>/matrix-completion-research.git
```

Or use HTTPS:

```bash
cd "/Users/karimau/Projects_C++/Course Project"
git remote add origin https://github.com/<username>/matrix-completion-research.git
```

## 3. Make the first commit

```bash
cd "/Users/karimau/Projects_C++/Course Project"
git add .
git commit -m "Initial research project setup"
```

## 4. Push

```bash
git push -u origin main
```

## 5. Check CI

After the push, open the repository on GitHub and check the `Actions` tab.
The CI workflow should run:

- `ruff check`
- `pytest`

