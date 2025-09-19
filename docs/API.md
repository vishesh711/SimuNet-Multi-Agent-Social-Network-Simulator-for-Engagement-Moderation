# SimuNet API Documentation

## Overview

SimuNet provides a comprehensive API for controlling simulations, managing agents, and accessing results. The API is built with FastAPI and provides both REST endpoints and WebSocket connections for real-time updates.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, SimuNet uses API key authentication:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/simulation/status
```

## Core Endpoints

### Simulation Management

#### GET /simulation/status
Get current simulation status and metrics.

**Response:**
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "agents": {
    "user_agents": 100,
    "content_agents": 1250,
    "moderator_agents": 5,
    "platform_agents": 1
  },
  "metrics": {
    "total_interactions": 5420,
    "content_created": 1250,
    "content_moderated": 45,
    "viral_content": 12
  }
}
```

#### POST /simulation/start
Start a new simulation.

**Request Body:**
```json
{
  "config": {
    "duration_hours": 24,
    "user_count": 100,
    "moderator_count": 5,
    "content_generation_rate": 0.1
  }
}
```

#### POST /simulation/stop
Stop the current simulation.

#### POST /simulation/reset
Reset simulation state.

### Agent Management

#### GET /agents
List all agents with their current status.

**Query Parameters:**
- `agent_type`: Filter by agent type (user, content, moderator, platform)
- `status`: Filter by status (active, inactive, error)
- `limit`: Number of results (default: 100)
- `offset`: Pagination offset

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "user_001",
      "agent_type": "user",
      "status": "active",
      "persona_type": "casual",
      "created_at": "2024-01-01T00:00:00Z",
      "last_active": "2024-01-01T12:30:00Z",
      "metrics": {
        "posts_created": 15,
        "interactions": 45,
        "engagement_rate": 0.12
      }
    }
  ],
  "total": 105,
  "limit": 100,
  "offset": 0
}
```

#### GET /agents/{agent_id}
Get detailed information about a specific agent.

#### POST /agents/user
Create a new user agent.

**Request Body:**
```json
{
  "persona_type": "influencer",
  "behavior_params": {
    "posting_frequency": 0.2,
    "engagement_likelihood": 0.8,
    "misinformation_susceptibility": 0.1
  },
  "network_connections": ["user_002", "user_003"]
}
```

#### DELETE /agents/{agent_id}
Remove an agent from the simulation.

### Content Management

#### GET /content
List content with filtering and pagination.

**Query Parameters:**
- `created_by`: Filter by creator agent ID
- `content_type`: Filter by content type
- `is_flagged`: Filter by moderation status
- `min_engagement`: Minimum engagement threshold
- `limit`: Number of results
- `offset`: Pagination offset

#### GET /content/{content_id}
Get detailed content information including metadata and engagement metrics.

**Response:**
```json
{
  "content_id": "content_001",
  "text_content": "This is a sample post about AI safety...",
  "created_by": "user_001",
  "created_at": "2024-01-01T10:00:00Z",
  "metadata": {
    "topic_classification": {
      "technology": 0.8,
      "politics": 0.1,
      "entertainment": 0.1
    },
    "sentiment_score": 0.6,
    "misinformation_probability": 0.05,
    "virality_potential": 0.7
  },
  "engagement_metrics": {
    "likes": 45,
    "shares": 12,
    "comments": 8,
    "views": 234,
    "engagement_rate": 0.28
  },
  "moderation_status": {
    "is_flagged": false,
    "confidence_scores": {
      "toxicity": 0.02,
      "hate_speech": 0.01
    },
    "action_taken": "none"
  }
}
```

#### POST /content/{content_id}/moderate
Manually trigger moderation for specific content.

### Feed Management

#### GET /feed/{user_id}
Get personalized feed for a user.

**Query Parameters:**
- `limit`: Number of items (default: 20)
- `ranking_mode`: Feed ranking algorithm (engagement_focused, safety_focused, balanced, chronological)

**Response:**
```json
{
  "user_id": "user_001",
  "feed": [
    {
      "content_id": "content_123",
      "score": 0.85,
      "reason": "high_engagement",
      "created_at": "2024-01-01T11:00:00Z"
    }
  ],
  "ranking_mode": "balanced",
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### Experiments

#### GET /experiments
List all A/B experiments.

#### POST /experiments
Create a new A/B experiment.

**Request Body:**
```json
{
  "name": "Safety vs Engagement Test",
  "description": "Compare safety-focused vs engagement-focused ranking",
  "control_config": {
    "ranking_mode": "engagement_focused",
    "engagement_weight": 0.8,
    "safety_weight": 0.2
  },
  "treatment_config": {
    "ranking_mode": "safety_focused",
    "engagement_weight": 0.3,
    "safety_weight": 0.7
  },
  "traffic_split": 0.5,
  "duration_hours": 168
}
```

#### GET /experiments/{experiment_id}
Get experiment details and current results.

#### POST /experiments/{experiment_id}/start
Start an experiment.

#### POST /experiments/{experiment_id}/stop
Stop an experiment early.

### Analytics

#### GET /analytics/engagement
Get engagement analytics and trends.

**Query Parameters:**
- `time_range`: Time range for analysis (1h, 24h, 7d, 30d)
- `group_by`: Grouping dimension (hour, day, persona_type, content_type)

#### GET /analytics/moderation
Get moderation effectiveness metrics.

#### GET /analytics/network
Get network analysis metrics (clustering, centrality, etc.).

#### GET /analytics/viral
Get viral content analysis.

### Data Export

#### GET /export/simulation
Export complete simulation data.

**Query Parameters:**
- `format`: Export format (json, csv, parquet)
- `start_time`: Start time for data range
- `end_time`: End time for data range

#### GET /export/experiment/{experiment_id}
Export experiment results.

## WebSocket Endpoints

### Real-time Updates

Connect to WebSocket for real-time simulation updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Simulation update:', data);
};
```

**Message Types:**
- `agent_created`: New agent added
- `content_created`: New content posted
- `interaction`: User interaction with content
- `moderation_action`: Content moderated
- `viral_detected`: Content went viral
- `experiment_update`: A/B test metrics update

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid agent configuration",
    "details": {
      "field": "persona_type",
      "issue": "Must be one of: casual, influencer, bot, activist"
    }
  }
}
```

## Rate Limiting

API endpoints are rate limited:
- **Standard endpoints**: 100 requests per minute
- **Data export endpoints**: 10 requests per minute
- **WebSocket connections**: 5 concurrent connections per API key

## SDK Examples

### Python SDK

```python
from simu_net_client import SimuNetClient

client = SimuNetClient(api_key="your-api-key")

# Start simulation
simulation = client.start_simulation({
    "duration_hours": 24,
    "user_count": 100
})

# Create experiment
experiment = client.create_experiment({
    "name": "Test Experiment",
    "control_config": {"engagement_weight": 0.8},
    "treatment_config": {"engagement_weight": 0.4}
})

# Get results
results = client.get_experiment_results(experiment.id)
```

### JavaScript SDK

```javascript
import { SimuNetClient } from 'simu-net-js';

const client = new SimuNetClient({ apiKey: 'your-api-key' });

// Real-time updates
client.onSimulationUpdate((data) => {
    console.log('Update:', data);
});

// Get analytics
const analytics = await client.getEngagementAnalytics({
    timeRange: '24h',
    groupBy: 'hour'
});
```

## Webhooks

Configure webhooks to receive notifications about simulation events:

```json
{
  "webhook_url": "https://your-app.com/webhooks/simu-net",
  "events": ["viral_detected", "experiment_completed"],
  "secret": "your-webhook-secret"
}
```

## OpenAPI Specification

The complete OpenAPI specification is available at:
```
http://localhost:8000/docs
```

Interactive API documentation (Swagger UI) is available at:
```
http://localhost:8000/redoc
```