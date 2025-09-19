# SimuNet: Multi-Agent Social Network Simulator

SimuNet is a sophisticated multi-agent simulation platform designed to study social network dynamics, content moderation effectiveness, and the impact of platform policies on user engagement and information spread. Built for researchers and platform safety teams, SimuNet provides a controlled environment to experiment with different algorithmic approaches and policy configurations.

## 🎯 Purpose

SimuNet addresses the critical need for understanding complex social media dynamics by simulating:
- **Engagement Patterns**: How content spreads and what drives virality
- **Moderation Effectiveness**: Impact of different content moderation strategies
- **Policy Experimentation**: A/B testing of platform algorithms and policies
- **Emergent Behaviors**: Network effects and community formation dynamics

## 🏗️ Architecture

### Multi-Agent System
- **User Agents**: Simulate diverse user personas with realistic behavior patterns
- **Content Agents**: Represent posts with rich metadata and semantic understanding
- **Moderator Agents**: Implement configurable content policy enforcement
- **Platform Agent**: Manage algorithmic feed ranking and recommendations

### Technology Stack
- **Agent Orchestration**: LangGraph for multi-agent coordination
- **Data Storage**: MongoDB for persistent data, Redis for caching
- **Vector Search**: FAISS for semantic content similarity
- **ML/NLP**: Hugging Face transformers for content analysis
- **API**: FastAPI with WebSocket support for real-time updates
- **Frontend**: React dashboard with D3.js visualizations

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- MongoDB
- Redis
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/simu-net.git
cd simu-net
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start services**
```bash
# Start MongoDB and Redis (if not using Docker)
# Or use Docker Compose
docker-compose up -d mongodb redis
```

6. **Run the simulation**
```bash
python -m simu_net.main
```

## 📊 Features

### ✅ Completed Features
- **Core Infrastructure**: Multi-agent framework with event-driven communication
- **Data Models**: Comprehensive Pydantic models for all entities
- **User Simulation**: Configurable personas with realistic behavior patterns
- **Content Processing**: Semantic analysis with embeddings and metadata
- **Vector Search**: FAISS-based similarity search for recommendations
- **Content Moderation**: NLP-powered toxicity and misinformation detection
- **Platform Algorithms**: Feed ranking with configurable engagement/safety weights

### 🚧 In Development
- **LangGraph Orchestration**: Multi-agent workflow coordination
- **A/B Testing Framework**: Policy experimentation infrastructure
- **Real-time Dashboard**: React-based visualization interface
- **Advanced Analytics**: Research tools and statistical analysis

### 🔮 Planned Features
- **Scalability**: Kubernetes deployment and horizontal scaling
- **Advanced ML**: Multi-language support and ensemble models
- **Research Integration**: Jupyter notebook integration and data export
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 🧪 Usage Examples

### Basic Simulation
```python
from simu_net import SimulationManager

# Create simulation with default configuration
sim = SimulationManager()

# Add agents
sim.add_user_agents(count=100, persona_mix="balanced")
sim.add_moderator_agents(count=5, policy="moderate")

# Run simulation
await sim.run(duration_hours=24)

# Analyze results
results = sim.get_results()
print(f"Total interactions: {results['total_interactions']}")
print(f"Content moderated: {results['moderated_content']}")
```

### Policy Experimentation
```python
# Create A/B experiment
experiment = sim.create_experiment(
    name="Safety vs Engagement",
    control_config={"engagement_weight": 0.8, "safety_weight": 0.2},
    treatment_config={"engagement_weight": 0.4, "safety_weight": 0.6},
    duration_hours=168  # 1 week
)

# Run experiment
await sim.run_experiment(experiment.id)

# Get results
results = experiment.get_results()
print(f"Control engagement: {results['control']['engagement_rate']}")
print(f"Treatment engagement: {results['treatment']['engagement_rate']}")
```

## 📁 Project Structure

```
simu_net/
├── agents/                 # Agent implementations
│   ├── base.py            # Base agent class
│   ├── user_agent.py      # User behavior simulation
│   ├── content_agent.py   # Content processing
│   ├── moderator_agent.py # Content moderation
│   └── platform_agent.py  # Feed algorithms
├── data/                  # Data models and storage
│   ├── models.py          # Pydantic data models
│   ├── storage.py         # Database connections
│   └── repositories.py    # Data access layer
├── events/                # Event system
│   ├── manager.py         # Event coordination
│   └── models.py          # Event data models
├── vector/                # Vector search
│   ├── faiss_manager.py   # FAISS operations
│   ├── similarity_search.py # Search algorithms
│   └── vector_store.py    # Vector storage
└── config.py              # Configuration management
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simu_net

# Run specific test categories
pytest tests/test_agents/
pytest tests/test_vector/
pytest tests/test_data/
```

## 📈 Monitoring

SimuNet provides comprehensive monitoring capabilities:

- **Agent Metrics**: Performance and behavior statistics
- **Content Metrics**: Engagement and moderation effectiveness
- **System Metrics**: Resource usage and performance
- **Experiment Metrics**: A/B test results and statistical significance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Full API Documentation](docs/)
- **Research Papers**: [Academic Publications](docs/research/)
- **Examples**: [Usage Examples](examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/simu-net/issues)

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Uses [Hugging Face](https://huggingface.co/) models for NLP capabilities
- Inspired by research in computational social science and platform governance

---

**SimuNet** - Understanding social networks through simulation 🌐