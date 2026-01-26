# Contributing to UTIO

Thank you for your interest in contributing to UTIO!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Beyond-Early-Exit.git
cd Beyond-Early-Exit
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to public functions
- Keep functions focused and testable

## Testing

- Add tests for new features in `tests/`
- Ensure all tests pass: `pytest`
- Aim for >80% code coverage

## Benchmarks

- Add benchmark cases to `utio/benchmark.py`
- Document expected performance characteristics
- Include hardware specs in benchmark results

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation if needed
6. Submit a PR with a clear description

## Questions?

Open an issue for discussion or questions.
