# SimuNet: Multi-Agent Social Network Simulator

A research platform for studying social media dynamics through multi-agent simulation, focusing on engagement patterns, content moderation, and policy effectiveness.

## 🎯 Overview

SimuNet simulates realistic social network interactions using autonomous agents to study:
- **Engagement Dynamics**: How content spreads and goes viral
- **Content Moderation**: Effectiveness of different moderation policies  
- **Systemic Effects**: Impact of algorithmic changes on user behavior
- **Policy Research**: A/B testing of platform policies and their outcomes

## 🏗️ Architecture

- **User Agents**: Simulate diverse user behaviors and personas
- **Content Agents**: Represent posts with rich metadata and embeddings
- **Moderator Agents**: Enforce content policies using ML classifiers
- **Platform Agent**: Manages feed algorithms and recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MongoDB
- Redis
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simu-net

# Install dependencies
make install-dev

# Setup environment
cp .env.example .env

# Setup databases (using Docker)
make setup-db
```

### Running

```bash
# Start the API server
make run-dev

# Run tests
make test

# View coverage
make test-cov
```

## 📊 Features

- **Multi-Agent Simulation**: LangGraph-orchestrated agent interactions
- **Real-time Visualization**: Live social graph and content flow monitoring
- **ML-Powered Moderation**: Toxicity, hate speech, and misinformation detection
- **Policy Experimentation**: A/B testing framework for platform policies
- **Research Integration**: Jupyter notebooks and statistical analysis tools
- **Scalable Infrastructure**: Kubernetes-ready with monitoring and metrics

## 🔬 Research Applications

- Study misinformation spread patterns
- Compare moderation policy effectiveness
- Analyze engagement vs. safety trade-offs
- Model network effects and viral dynamics
- Test algorithmic bias and fairness

## 📈 Development Status

This project is under active development. See the [implementation plan](.kiro/specs/simu-net/tasks.md) for current progress.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [Requirements Document](.kiro/specs/simu-net/requirements.md)
- [Design Document](.kiro/specs/simu-net/design.md)
- [Implementation Plan](.kiro/specs/simu-net/tasks.md)