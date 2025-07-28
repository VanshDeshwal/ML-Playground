# ML Playground - HTML/CSS/JS Frontend

A modern, responsive web interface for exploring machine learning algorithms. This frontend is designed to be deployed on GitHub Pages with a backend hosted on Azure.

## 🚀 Architecture

```
Frontend (GitHub Pages) ←→ Backend (Azure)
    HTML/CSS/JS           FastAPI + ML Models
```

## 📁 Project Structure

```
frontend/
├── index.html          # Main HTML page
├── styles.css          # Modern CSS with responsive design
├── app.js             # Main application logic
├── api.js             # Backend API communication
└── config.js          # Environment configuration
```

## 🔧 Features

- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Real-time Backend Status**: Shows connection status to your API
- **Algorithm Cards**: Interactive cards for each ML algorithm
- **Modal Interface**: Clean modal dialogs for algorithm details
- **Training Interface**: Start training with visual feedback
- **Error Handling**: Graceful fallbacks for offline scenarios
- **Environment Detection**: Automatically switches between local and production APIs

## 🌐 Deployment Options

### Option 1: GitHub Pages (Recommended)

1. **Enable GitHub Pages**:
   - Go to your repository settings
   - Navigate to "Pages" section
   - Select "GitHub Actions" as source
   - The workflow in `.github/workflows/deploy.yml` will handle deployment

2. **Update Backend URL**:
   ```javascript
   // In config.js, update the production URL
   production: {
       API_BASE_URL: 'https://your-azure-app.azurewebsites.net'
   }
   ```

3. **Access Your Site**:
   - Your site will be available at: `https://yourusername.github.io/repository-name`

### Option 2: Local Development

1. **Simple HTTP Server**:
   ```bash
   # Python 3
   python -m http.server 8080
   
   # Node.js (if you have it)
   npx serve .
   ```

2. **Access Locally**:
   - Open browser to `http://localhost:8080`

## ⚙️ Configuration

### Environment Setup

The app automatically detects the environment:

- **Local Development**: Uses `http://localhost:8000` (your local FastAPI)
- **Production**: Uses your Azure backend URL

### Backend Requirements

Your backend must support CORS for the frontend domain. The backend is already configured for:

- GitHub Pages (`*.github.io`)
- Local development (`localhost`)

## 🎨 Customization

### Styling
- Edit `styles.css` for visual customization
- CSS variables at the top make color theming easy
- Responsive breakpoints are predefined

### Functionality
- Add new features in `app.js`
- Extend API calls in `api.js`
- Configure endpoints in `config.js`

## 🔍 Browser Compatibility

- **Modern Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **Mobile**: iOS Safari, Chrome Mobile, Samsung Internet
- **Features Used**: ES6+, Fetch API, CSS Grid, CSS Variables

## 📱 Responsive Design

The interface adapts to different screen sizes:

- **Desktop**: Multi-column card grid with full modal
- **Tablet**: Responsive grid with touch-friendly interactions
- **Mobile**: Single column layout with mobile-optimized modals

## 🚀 Performance Features

- **Caching**: API responses are cached to reduce server load
- **Lazy Loading**: Content loads progressively
- **Optimized Assets**: Minimal external dependencies
- **Fast Loading**: Lightweight vanilla JavaScript

## 🔧 Development

### Local Development Setup

1. **Start Backend**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start Frontend**:
   ```bash
   cd frontend
   python -m http.server 8080
   ```

3. **Open Browser**: `http://localhost:8080`

### Debugging

- Open browser developer tools (F12)
- Check console for API errors
- Network tab shows backend communication
- Use `refreshApp()` in console to reload data

## 📦 Deployment Checklist

- [ ] Update backend URL in `config.js`
- [ ] Test local development setup
- [ ] Push to GitHub (triggers auto-deployment)
- [ ] Verify GitHub Pages is enabled
- [ ] Test production deployment
- [ ] Check backend CORS configuration
- [ ] Verify all algorithms load correctly

## 🎯 Next Steps

1. **Deploy Backend to Azure**:
   - Create Azure App Service
   - Deploy your FastAPI backend
   - Update the URL in `config.js`

2. **Enable GitHub Pages**:
   - Commit and push your changes
   - Enable GitHub Pages in repository settings
   - Your site will be live automatically!

3. **Custom Domain (Optional)**:
   - Add a custom domain in GitHub Pages settings
   - Update CORS settings in backend if needed

## 💡 Tips

- **Fast Development**: Use browser dev tools for real-time CSS editing
- **API Testing**: Use the browser console to call `checkBackend()` or `refreshApp()`
- **Mobile Testing**: Use browser responsive design mode
- **Performance**: Monitor Network tab for optimization opportunities

---

**Ready to deploy? Just push to GitHub and your ML Playground will be live on GitHub Pages!** 🎉
