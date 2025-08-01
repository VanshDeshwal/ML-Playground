# 🧠 ML Playground

A modern, interactive machine learning playground built with FastAPI and vanilla JavaScript. Explore, train, and experiment with ML algorithms through a clean, responsive web interface.

## ✨ Features

- **🔄 Auto-Discovery**: Automatically discovers ML algorithms from the `/core` directory
- **🎨 Interactive UI**: Clean, modern interface with syntax-highlighted code viewing
- **📊 Real-time Training**: Live training progress and metrics visualization
- **🚀 Fast API**: High-performance backend with automatic API documentation
- **📱 Responsive**: Works seamlessly on desktop and mobile devices
- **🎯 Type Safety**: Full TypeScript-style validation with Pydantic models

## 🏗️ Architecture

```
ML Playground/
├── frontend/          # Static web interface
│   ├── index.html     # Homepage
│   ├── algorithm.html # Algorithm training page
│   ├── styles.css     # Modern CSS with variables
│   ├── app.js         # Homepage controller
│   ├── algorithm.js   # Training page controller
│   └── api.js         # API service with smart caching
├── backend/           # FastAPI backend
│   ├── main.py        # Application entry point
│   ├── api/           # API route handlers
│   ├── services/      # Business logic
│   ├── models/        # Pydantic data models
│   ├── adapters/      # Algorithm adapters
│   └── core_integration/ # Auto-discovery system
├── core/              # ML algorithm implementations
└── .github/workflows/ # CI/CD pipelines
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VanshDeshwal/ML-Playground.git
   cd ML-Playground
   ```

2. **Setup Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Run the backend**
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open the frontend**
   - Open `frontend/index.html` in your browser
   - Or serve it locally: `python -m http.server 3000` from the frontend directory

## 📚 Adding New Algorithms

1. **Create your algorithm file** in the `/core` directory:
   ```python
   # core/my_algorithm.py
   
   # === SNIPPET-START: MyAlgorithmClass ===
   class MyAlgorithmClass:
       def __init__(self, param1=1.0, param2=10):
           self.param1 = param1
           self.param2 = param2
       
       def fit(self, X, y):
           # Training logic here
           return self
       
       def predict(self, X):
           # Prediction logic here
           return predictions
   # === SNIPPET-END: MyAlgorithmClass ===
   ```

2. **Restart the backend** - The auto-discovery system will automatically detect and register your new algorithm!

## 🔧 API Endpoints

- `GET /api/algorithms` - List all available algorithms
- `GET /api/algorithms/{id}` - Get algorithm details
- `POST /api/algorithms/{id}/train` - Train an algorithm
- `GET /api/algorithms/{id}/code` - Get algorithm source code
- `GET /health` - Health check

Full API documentation available at `http://localhost:8000/docs` when running the backend.

## 🌐 Deployment

### Frontend (GitHub Pages)
The frontend is automatically deployed to GitHub Pages on every push to main.

### Backend (Azure Container Apps)
The backend is automatically deployed to Azure on every push to main.

**Live Demo**: [ML Playground](https://vanshdeshwal.github.io/ML-Playground/)

## 🛠️ Technologies

### Frontend
- **Vanilla JavaScript** - No frameworks, maximum performance
- **CSS Grid & Flexbox** - Modern responsive layouts
- **Prism.js** - Syntax highlighting for code viewing
- **Font Awesome** - Beautiful icons
- **Inter Font** - Clean, modern typography

### Backend
- **FastAPI** - High-performance Python web framework
- **Pydantic** - Data validation and serialization
- **NumPy & Scikit-learn** - Core ML libraries
- **Uvicorn** - ASGI server

### DevOps
- **GitHub Actions** - CI/CD pipelines
- **Docker** - Containerization
- **Azure Container Apps** - Cloud deployment

## 📈 Performance Features

- **Smart Caching** - Frontend caches API responses with TTL
- **Async Processing** - Non-blocking algorithm training
- **Code Splitting** - Modular architecture for fast loading
- **Auto-Discovery** - Dynamic algorithm loading without restarts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add your algorithm to the `/core` directory
4. Test locally
5. Commit changes: `git commit -am 'Add my feature'`
6. Push to branch: `git push origin feature/my-feature`
7. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for the ML community
- Inspired by modern web development practices
- Designed for educational and research purposes

---

**Happy Learning! 🚀**
