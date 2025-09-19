# Contributing to SimuNet

Thank you for your interest in contributing to SimuNet! This document provides guidelines and information for contributors.

## 🎯 Project Vision

SimuNet aims to be the leading platform for social network simulation and research. We welcome contributions that:

- Improve simulation realism and accuracy
- Enhance research capabilities and reproducibility
- Add new agent behaviors or content types
- Optimize performance and scalability
- Improve documentation and usability

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- MongoDB (for local development)
- Redis (for local development)
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/your-username/simu-net.git
cd simu-net
```

2. **Set up development environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

3. **Configure environment**
```bash
cp .env.example .env.dev
# Edit .env.dev with your local configuration
```

4. **Start local services**
```bash
# Using Docker Compose (recommended)
docker-compose -f docker-compose.dev.yml up -d

# Or start services manually
# MongoDB: mongod --dbpath ./data/db
# Redis: redis-server
```

5. **Run tests to verify setup**
```bash
pytest
```

## 📋 Development Workflow

### Branch Strategy

We use a feature branch workflow:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/feature-name`: Individual feature development
- `hotfix/issue-description`: Critical bug fixes

### Making Changes

1. **Create a feature branch**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
```bash
# Run full test suite
pytest

# Run specific test categories
pytest tests/test_agents/
pytest tests/test_vector/

# Run with coverage
pytest --cov=simu_net --cov-report=html
```

4. **Commit your changes**
```bash
git add .
git commit -m "feat: add new user persona type"
```

5. **Push and create pull request**
```bash
git push origin feature/your-feature-name
# Create PR through GitHub interface
```

## 🎨 Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

**Example:**
```python
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from .base import SimuNetAgent
from ..data.models import ContentAgent


class UserAgent(SimuNetAgent):
    """Simulates human user behavior in social networks.
    
    This agent implements realistic posting, engagement, and network
    growth patterns based on configurable persona types.
    
    Args:
        persona_type: Type of user persona to simulate
        behavior_params: Custom behavior parameters
        network_connections: Initial network connections
        
    Example:
        >>> agent = UserAgent(
        ...     persona_type=PersonaType.INFLUENCER,
        ...     behavior_params={"posting_frequency": 0.3}
        ... )
        >>> await agent.start()
    """
    
    def __init__(
        self,
        persona_type: PersonaType,
        behavior_params: Optional[Dict[str, float]] = None,
        network_connections: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.persona_type = persona_type
        self.behavior_params = behavior_params or {}
        self.network_connections = network_connections or []
        
    async def create_content(self, topic: str) -> Optional[str]:
        """Create new content based on user interests.
        
        Args:
            topic: Content topic to focus on
            
        Returns:
            Content ID if created, None if skipped
            
        Raises:
            ContentCreationError: If content generation fails
        """
        # Implementation here
        pass
```

### Code Organization

- **Modules**: Keep modules focused and cohesive
- **Classes**: Single responsibility principle
- **Functions**: Pure functions when possible, clear side effects
- **Constants**: Use UPPER_CASE for module-level constants
- **Configuration**: Use Pydantic models for configuration

### Testing Guidelines

We use pytest with the following conventions:

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

from simu_net.agents.user_agent import UserAgent
from simu_net.data.models import PersonaType


class TestUserAgent:
    """Test suite for UserAgent class."""
    
    @pytest.fixture
    def user_agent(self):
        """Create a test user agent."""
        return UserAgent(
            agent_id="test_user_001",
            persona_type=PersonaType.CASUAL
        )
    
    @pytest.mark.asyncio
    async def test_content_creation(self, user_agent):
        """Test content creation functionality."""
        # Arrange
        topic = "technology"
        
        # Act
        content_id = await user_agent.create_content(topic)
        
        # Assert
        assert content_id is not None
        assert isinstance(content_id, str)
    
    @pytest.mark.parametrize("persona_type,expected_frequency", [
        (PersonaType.CASUAL, 0.1),
        (PersonaType.INFLUENCER, 0.5),
        (PersonaType.BOT, 1.0),
    ])
    def test_posting_frequency_by_persona(self, persona_type, expected_frequency):
        """Test posting frequency varies by persona type."""
        agent = UserAgent(persona_type=persona_type)
        assert agent.get_posting_frequency() == expected_frequency
```

**Test Categories:**
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test scalability and performance
- **Property Tests**: Test with generated inputs using Hypothesis

## 📚 Documentation Guidelines

### Code Documentation

- **Docstrings**: All public APIs must have comprehensive docstrings
- **Type Hints**: Use type hints for better IDE support and documentation
- **Comments**: Explain complex logic, not obvious code
- **Examples**: Include usage examples in docstrings

### User Documentation

- **API Documentation**: Keep API docs up to date with code changes
- **Tutorials**: Add tutorials for new features
- **Architecture Docs**: Update architecture documentation for significant changes
- **Research Guides**: Document research methodologies and best practices

### Documentation Format

We use Markdown for most documentation:

```markdown
# Feature Name

Brief description of the feature and its purpose.

## Usage

Basic usage example:

```python
from simu_net import FeatureName

feature = FeatureName(config="value")
result = feature.process()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| config | str | "default" | Configuration parameter |

## Examples

### Basic Example

Detailed example with explanation...

### Advanced Example

More complex usage scenario...
```

## 🧪 Testing Requirements

### Test Coverage

- **Minimum Coverage**: 80% overall, 90% for core components
- **Critical Paths**: 100% coverage for agent decision-making logic
- **Edge Cases**: Test error conditions and boundary cases
- **Performance**: Include performance regression tests

### Test Types

1. **Unit Tests**
   - Test individual functions and methods
   - Mock external dependencies
   - Fast execution (< 1 second per test)

2. **Integration Tests**
   - Test component interactions
   - Use real databases (test instances)
   - Moderate execution time (< 10 seconds per test)

3. **End-to-End Tests**
   - Test complete simulation workflows
   - Use production-like environment
   - Longer execution time acceptable

4. **Property-Based Tests**
   - Use Hypothesis for generated test inputs
   - Test invariants and properties
   - Especially useful for agent behaviors

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simu_net --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_agents/test_user_agent.py

# Run tests matching pattern
pytest -k "test_content_creation"

# Run tests with specific markers
pytest -m "integration"

# Run tests in parallel
pytest -n auto
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Test with latest version** to ensure bug still exists
3. **Reproduce consistently** with minimal example
4. **Check logs** for error messages and stack traces

### Bug Report Template

```markdown
**Bug Description**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Configure simulation with...
2. Run agent with...
3. Observe error...

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS 12.0]
- Python version: [e.g., 3.9.7]
- SimuNet version: [e.g., 0.2.1]
- Dependencies: [relevant package versions]

**Logs**
```
Paste relevant log output here
```

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Use Case**
Describe the research or simulation scenario where this would be useful.

**Proposed Implementation**
If you have ideas about how this could be implemented.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

## 🏷️ Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(agents): add influencer persona type

Add new influencer persona with higher posting frequency and 
engagement rates. Includes configuration options for follower 
count and content amplification.

Closes #123

fix(vector): handle empty embedding vectors

Previously crashed when content had no text. Now returns
default embedding vector for empty content.

docs(api): update authentication examples

Add examples for JWT token usage and refresh token flow.

test(moderator): add property-based tests for policy enforcement

Use Hypothesis to generate test cases for different content
types and policy configurations.
```

## 🔄 Pull Request Process

### PR Checklist

Before submitting a pull request:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR description explains changes
- [ ] Breaking changes documented

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Architecture documentation updated (if applicable)

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information that reviewers should know.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Reviewer tests the changes in their environment
4. **Approval**: Maintainer approves and merges the PR

## 🏆 Recognition

We appreciate all contributions! Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **GitHub**: Contributor statistics and badges
- **Academic Papers**: Co-authorship for significant research contributions

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat with maintainers and community
- **Email**: Direct contact for sensitive issues

### Maintainer Response Times

- **Bug Reports**: Within 48 hours
- **Feature Requests**: Within 1 week
- **Pull Requests**: Within 1 week
- **Security Issues**: Within 24 hours

## 📄 License

By contributing to SimuNet, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to SimuNet! 🎉