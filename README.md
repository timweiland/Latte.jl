# Latte

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timweiland.github.io/Latte.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timweiland.github.io/Latte.jl/dev/)
[![Build Status](https://github.com/timweiland/Latte.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/Latte.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/Latte.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/Latte.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Development

For developers contributing to this package, see [DEVELOPMENT.md](DEVELOPMENT.md) for the complete development workflow including:

- **Quick setup**: `make setup && make generate-reference && make test`
- **Fast testing**: Pre-computed MCMC reference data for rapid CI
- **Reference generation**: High-quality MCMC validation data
- **Code formatting**: Automated with Runic

```bash
# Quick start for developers
make setup              # Install dependencies and TestEnv
make generate-reference # Generate MCMC reference data (10-15 min)
make test              # Run fast tests (30 seconds)
```
