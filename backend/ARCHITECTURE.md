# ML Playground Backend Architecture

## 🎯 **Design Principles**
- **Single Source of Truth**: Core algorithms live in `/core` only
- **Clean Separation**: Backend adapts core algorithms, doesn't implement them
- **Auto-Discovery**: Dynamic algorithm registration without manual configuration
- **Plugin Architecture**: Easy to add new algorithms by just adding to `/core`
- **Type Safety**: Full typing and validation throughout

## 📁 **Directory Structure**

```
backend/
├── api/                    # FastAPI routes and endpoints
│   ├── __init__.py
│   ├── algorithms.py       # Algorithm CRUD operations
│   ├── training.py         # Training endpoints
│   ├── predictions.py      # Prediction endpoints
│   └── code.py            # Code snippet endpoints
├── adapters/              # Bridge between core algorithms and API
│   ├── __init__.py
│   ├── base.py            # Base adapter interface
│   └── algorithm_adapter.py # Generic algorithm adapter
├── services/              # Business logic services
│   ├── __init__.py
│   ├── algorithm_service.py # Algorithm management
│   ├── training_service.py  # Training orchestration
│   ├── snippet_service.py   # Code extraction (existing)
│   └── comparison_service.py # Performance comparisons
├── core_integration/      # Integration with /core algorithms
│   ├── __init__.py
│   ├── discovery.py       # Auto-discover core algorithms
│   ├── loader.py          # Load and instantiate algorithms
│   └── validator.py       # Validate algorithm interfaces
├── models/                # Pydantic models for API
│   ├── __init__.py
│   ├── requests.py        # Request models
│   ├── responses.py       # Response models
│   └── schemas.py         # Data schemas
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── cache.py          # Caching utilities
│   ├── metrics.py        # Performance metrics
│   └── validation.py     # Input validation
├── config.py             # Configuration management
├── main.py              # FastAPI application
└── requirements.txt
```

## 🔄 **Data Flow**

1. **API Request** → FastAPI endpoint in `/api`
2. **Service Layer** → Business logic in `/services`
3. **Adapter Layer** → Unified interface in `/adapters`
4. **Core Integration** → Load from `/core` algorithms
5. **Core Algorithm** → Actual ML implementation
6. **Response** → Back through the layers

## 🚀 **Benefits**

- **Zero Duplication**: Core algorithms in `/core` only
- **Auto-Discovery**: New algorithms automatically available
- **Clean Testing**: Each layer can be tested independently
- **Easy Scaling**: Add algorithms without backend changes
- **Type Safety**: Full typing throughout the stack
- **Performance**: Efficient caching and async operations
