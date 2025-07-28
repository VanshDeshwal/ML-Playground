# 🤖 ML Playground

An interactive platform to explore and experiment with machine learning algorithms implemented from scratch using FastAPI and Streamlit.

## 🌟 Features

- **📝 From-Scratch Implementations**: All ML algorithms implemented using pure NumPy for educational transparency
- **🎛️ Interactive Web Interface**: Real-time hyperparameter tuning with immediate visual feedback
- **📊 Rich Visualizations**: Training dynamics, model coefficients, and prediction visualizations
- **⚡ Fast API Backend**: RESTful API for model training and data generation
- **🔬 Educational Focus**: Understand how algorithms work under the hood
- **🎯 Extensible Architecture**: Easy to add new algorithms and features

## 🚀 Current Algorithms

### Regression
- **Linear Regression** - Gradient descent implementation with customizable learning rate and iterations

### Classification  
- **Logistic Regression** - Sigmoid-based binary classification with gradient descent
- **k-Nearest Neighbors (Classification)** - Lazy learning with multiple distance metrics
- **Decision Tree (Classification)** - Information gain-based splitting

### Clustering
- **k-Means Clustering** - Lloyd's algorithm with centroid tracking

### Regression
- **k-Nearest Neighbors (Regression)** - Distance-weighted prediction
- **Decision Tree (Regression)** - Variance-based splitting for continuous targets

## 🌟 New Features

- **🔄 Scikit-learn Comparison**: Compare your custom implementations with scikit-learn
- **📊 Multiple Datasets**: Choose from built-in datasets (Iris, Wine, Diabetes, Breast Cancer) or upload your own
- **📈 Rich Visualizations**: Interactive plots showing training dynamics and model performance
- **⚙️ Flexible Configuration**: Extensive hyperparameter tuning options

## 🛠️ Local Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/VanshDeshwal/ML-Playground.git
   cd ML-Playground
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend API**
   ```bash
   cd backend
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Start the frontend (in a new terminal)**
   ```bash
   cd frontend
   streamlit run app.py --server.port 8501
   ```

5. **Access the application**
   - 🌐 **Frontend**: http://localhost:8501
   - 🔗 **API**: http://localhost:8000

5. **Access the application**
   - Frontend (Streamlit): http://localhost:8501
   - Backend API (FastAPI): http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
ML-Playground/
│
├── backend/                   # FastAPI backend
│   ├── main.py               # FastAPI app with API endpoints
│   └── models/               # From-scratch ML implementations
│       ├── __init__.py
│       ├── linear_regression.py
│       ├── logistic_regression.py
│       ├── kmeans.py
│       ├── knn.py
│       └── decision_tree.py
│
├── frontend/                 # Streamlit frontend
│   └── app.py               # Main Streamlit application
│
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🎯 How to Use

1. **Select Algorithm**: Choose from available ML algorithms on the home page
2. **Choose Dataset**: 
   - **Generated**: Synthetic data with configurable parameters
   - **Built-in**: Real datasets (Iris, Wine, Diabetes, Breast Cancer)
   - **Upload**: Your own CSV file
3. **Configure Dataset**: Adjust parameters like sample size, features, noise level
4. **Set Hyperparameters**: Tune algorithm-specific parameters like learning rate
5. **Enable Comparison**: Toggle scikit-learn comparison to see performance differences
6. **Train Model**: Click "Train Model" to run the algorithm
7. **Analyze Results**: View performance metrics, comparisons, and visualizations
8. **Experiment**: Try different configurations to understand algorithm behavior

## 🔧 API Endpoints

### Core Endpoints
- `GET /`: Welcome message
- `GET /algorithms`: List all available algorithms
- `GET /algorithms/{id}`: Get specific algorithm details
- `POST /train`: Train a model with specified parameters
- `GET /datasets`: List available built-in datasets
- `POST /upload_dataset`: Upload a custom CSV dataset
- `GET /generate_data`: Generate sample datasets for visualization

### Example API Usage

```python
import requests

# Get available algorithms
response = requests.get("http://localhost:8000/algorithms")
algorithms = response.json()

# Train a linear regression model with sklearn comparison
payload = {
    "algorithm_id": "linear_regression",
    "hyperparameters": {"alpha": 0.01, "n_iters": 1000},
    "dataset_config": {"n_samples": 500, "n_features": 10, "noise": 0.1},
    "compare_sklearn": True,
    "dataset_source": "builtin",
    "builtin_dataset": "diabetes"
}
response = requests.post("http://localhost:8000/train", json=payload)
result = response.json()

# Access both custom and sklearn results
custom_metrics = result['metrics']
sklearn_metrics = result['sklearn_comparison']
```

## 🎨 Screenshots

### Home Page
- Algorithm selection and overview
- Server status and quick start guide

### Algorithm Explorer  
- Interactive hyperparameter tuning
- Real-time training and results
- Performance metrics and visualizations

## 🧪 Development

### Adding New Algorithms

1. **Create Algorithm Class** in `backend/models/`
   ```python
   class MyNewAlgorithm:
       def __init__(self, param1=default1, param2=default2):
           # Initialize algorithm
           
       def fit(self, X, y):
           # Training logic
           
       def predict(self, X):
           # Prediction logic
   ```

2. **Update API** in `backend/main.py`
   - Add algorithm to `ALGORITHMS` dictionary
   - Implement training logic in `/train` endpoint

3. **Update Frontend** in `frontend/app.py`
   - Add visualization logic for new algorithm type

### Running Tests

```powershell
# Test the linear regression implementation
cd backend/models
python linear_regression.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Add New Algorithms**: Implement new ML algorithms from scratch
2. **Improve Visualizations**: Create better charts and interactive plots
3. **Enhance UI/UX**: Improve the Streamlit interface
4. **Add Features**: Dataset upload, model comparison, etc.
5. **Documentation**: Improve docs and add tutorials

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

## 📋 Technologies Used

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **ML Libraries**: NumPy, scikit-learn (for datasets/metrics only)
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **Development**: Python 3.8+, Virtual Environment

## 🎓 Educational Goals

This project aims to:
- Demystify machine learning algorithms
- Provide hands-on experience with algorithm implementation
- Show the impact of hyperparameters on model performance
- Visualize training dynamics and model behavior
- Bridge the gap between theory and practice

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Thanks to the scikit-learn team for dataset generation utilities
- Plotly for amazing interactive visualizations
- FastAPI and Streamlit communities for excellent documentation

---

Made with ❤️ for learning and understanding ML algorithms
