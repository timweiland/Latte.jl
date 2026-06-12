# Contributing to Latte.jl

Thanks for your interest in contributing. Latte.jl is a Julia package for
inference in latent Gaussian models, exposing INLA, TMB, and HMC-Laplace behind
one `@latte` model. Contributions of all kinds are welcome: bug reports, feature
requests, documentation, and code.

## Getting set up

```bash
git clone https://github.com/timweiland/Latte.jl
cd Latte.jl
make setup          # instantiate the package and dev environments
```

## Development workflow

The `Makefile` wraps the common tasks:

```bash
make test           # run the full test suite
make format         # format the whole repo (runic)
make docs           # build the documentation
make reference-data # regenerate MCMC reference data used by some tests
```

To run a single test file interactively:

```julia
julia --project
julia> using TestEnv; TestEnv.activate()
julia> include("test/path/to/file.jl")
```

## Pull requests

1. Open an issue first for anything beyond a small fix, so we can agree on the
   approach before you invest time.
2. Branch off `main`, make your change, and add tests. New behaviour should come
   with tests that fail before the change and pass after.
3. Run `make format` and `make test` locally. A `runic` formatting hook runs on
   commit, and CI runs the suite across the supported Julia versions.
4. Keep public functions documented with docstrings.
5. Open the PR against `main` with a short description of what changed and why.

## Reporting bugs

Open an issue on the [tracker](https://github.com/timweiland/Latte.jl/issues)
with a minimal reproducible example: the model, the data (or a synthetic
generator), and what you expected versus what happened. Please include your
Julia and Latte versions.

## License

By contributing you agree that your contributions will be licensed under the MIT
License that covers the project.
