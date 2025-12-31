# PyPI Trusted Publishing Setup

This project uses GitHub Actions to automatically publish releases to PyPI when version tags are pushed.

## Trusted Publishing (Recommended)

Trusted publishing uses OpenID Connect (OIDC) to authenticate with PyPI without needing API tokens.

### Setup Steps

1. **Go to PyPI Publishing Settings**:
   - Visit: https://pypi.org/manage/account/publishing/
   - Log in to your PyPI account

2. **Add a new publisher**:
   - PyPI Project Name: `um-agent-coder`
   - Owner: `UMwai`
   - Repository name: `um-agent-coder`
   - Workflow name: `release.yml`
   - Environment name: `release`

3. **Configure GitHub Environment** (if not already exists):
   - Go to: https://github.com/UMwai/um-agent-coder/settings/environments
   - Create environment named `release`
   - Add protection rules (optional but recommended):
     - Required reviewers: Add team members who should approve releases
     - Wait timer: 0 minutes (or add delay if needed)
     - Deployment branches: Only allow tags matching `v*`

4. **Test the workflow**:
   ```bash
   # Create and push a test tag
   git tag v0.2.1-test
   git push origin v0.2.1-test

   # Watch the workflow: https://github.com/UMwai/um-agent-coder/actions

   # Delete test tag after verification
   git tag -d v0.2.1-test
   git push origin :refs/tags/v0.2.1-test
   ```

## Fallback: API Token Method

If trusted publishing is not configured, you can use a PyPI API token:

1. **Generate API token**:
   - Visit: https://pypi.org/manage/account/token/
   - Create a new token with scope limited to `um-agent-coder` project
   - Copy the token (starts with `pypi-`)

2. **Add secret to GitHub**:
   - Go to: https://github.com/UMwai/um-agent-coder/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste the PyPI token

3. **Update workflow**:
   - Edit `.github/workflows/release.yml`
   - Comment out the "Publish to PyPI (Trusted Publishing)" step
   - Uncomment the "Publish to PyPI (API Token)" step

## Creating a Release

1. **Update version**:
   ```bash
   # Edit pyproject.toml and update version
   vim pyproject.toml  # Change version = "0.2.0" to "0.3.0"
   ```

2. **Commit changes**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.3.0"
   git push
   ```

3. **Create and push tag**:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

4. **Monitor workflow**:
   - GitHub Actions: https://github.com/UMwai/um-agent-coder/actions
   - PyPI: https://pypi.org/project/um-agent-coder/

## Workflow Features

- **Automatic builds**: Creates wheel and source distributions
- **Package validation**: Runs `twine check` before publishing
- **GitHub Releases**: Automatically creates release with:
  - Release notes
  - Distribution files attached
  - Links to PyPI
  - Auto-generated changelog
- **Environment protection**: Requires approval if configured
- **Security**: Uses OIDC tokens instead of long-lived credentials

## Troubleshooting

### Trusted Publishing Fails

If you see authentication errors:
1. Verify PyPI publisher configuration matches exactly
2. Check environment name is `release`
3. Ensure repository owner and name are correct
4. Make sure workflow file path is `.github/workflows/release.yml`

### Package Already Exists

PyPI doesn't allow overwriting existing versions:
- Increment version in `pyproject.toml`
- Delete the tag: `git tag -d v0.x.y && git push origin :refs/tags/v0.x.y`
- Create new tag with updated version

### Build Fails

Check the build logs in GitHub Actions:
- Ensure all dependencies are listed in `pyproject.toml`
- Verify package structure is correct
- Run local build test: `python -m build`

## References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions PyPI Publish](https://github.com/marketplace/actions/pypi-publish)
- [Python Packaging Guide](https://packaging.python.org/)
