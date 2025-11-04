# Contributing to Options Trading Bot

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Trading_bot.git
   cd Trading_bot
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-indicator` - for new features
- `fix/reward-calculation-bug` - for bug fixes
- `docs/update-readme` - for documentation
- `refactor/cleanup-env-code` - for refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style and conventions
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes

Before submitting:
```bash
# Test that imports work
python -c "from src.options_clstm_ppo import OptionsClstmPPO; print('✅ Imports OK')"

# Run a quick training test (3 episodes)
python train_enhanced_clstm_ppo.py --fresh-start --episodes 3
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: implement new technical indicator"
```

Good commit message format:
```
<type>: <short description>

<longer description if needed>

Fixes #<issue-number>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference to any related issues
- Screenshots/logs if applicable

## 📋 Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use type hints where appropriate

Example:
```python
def calculate_reward(
    self,
    portfolio_value_old: float,
    portfolio_value_new: float
) -> float:
    """Calculate portfolio-based reward.
    
    Args:
        portfolio_value_old: Previous portfolio value
        portfolio_value_new: Current portfolio value
        
    Returns:
        Scaled reward value
    """
    return (portfolio_value_new - portfolio_value_old) / portfolio_value_old * 1e-4
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

### File Organization

- Core algorithms go in `src/`
- Utility scripts go in `scripts/`
- Documentation goes in `docs/`
- Tests go in `tests/` (if you create this directory)

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Exact steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS (Linux/Windows/Mac)
   - Python version
   - PyTorch version
   - GPU (if applicable)
6. **Logs**: Relevant error messages or logs
7. **Code**: Minimal code to reproduce (if applicable)

## 💡 Suggesting Features

When suggesting features:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your idea for solving it
3. **Alternatives**: Other solutions you've considered
4. **Additional Context**: Any other relevant information

## 🧪 Testing Guidelines

### Before Submitting PR

1. **Import Test**: Verify all imports work
2. **Quick Training Test**: Run 3-5 episodes to ensure no crashes
3. **Check Logs**: Verify no new errors or warnings
4. **GPU Test** (if applicable): Test on GPU if making GPU-related changes

### Writing Tests

If adding new functionality, consider adding tests:

```python
# tests/test_reward_function.py
def test_reward_calculation():
    """Test that reward function calculates correctly."""
    old_value = 100000
    new_value = 101000
    expected_reward = 0.1  # 1% gain * 1e-4 scaling
    
    reward = calculate_reward(old_value, new_value)
    assert abs(reward - expected_reward) < 1e-6
```

## 📝 Documentation Guidelines

### Code Comments

- Explain **why**, not **what** (code shows what)
- Comment complex algorithms
- Add references to research papers if implementing paper concepts

### README Updates

When adding features, update:
- Feature list
- Quick start guide (if needed)
- Command line arguments
- Examples

### Documentation Files

For major features, add documentation in `docs/`:
- `docs/YOUR_FEATURE.md`
- Include: overview, usage, examples, troubleshooting

## 🎯 Areas for Contribution

### High Priority

- **Performance Optimization**: Speed up training or inference
- **New Indicators**: Add technical indicators or features
- **Testing**: Add unit tests and integration tests
- **Documentation**: Improve docs, add examples
- **Bug Fixes**: Fix reported issues

### Medium Priority

- **Visualization**: Add training visualization tools
- **Backtesting**: Improve backtesting capabilities
- **Risk Management**: Enhance risk controls
- **Data Sources**: Add new data providers

### Low Priority

- **UI/Dashboard**: Web dashboard for monitoring
- **Alerts**: Add notification system
- **Multi-Asset**: Support for other asset classes

## ❓ Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Comments**: For code review questions

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards others

By participating, you agree to abide by these principles.

