# ML Playground Deployment Guide

## Frontend Deployment (GitHub Pages)

The frontend is automatically deployed when you push to the main branch. However, it requires a deployed backend to function properly.

## Backend Deployment Options

### Option 1: Heroku (Recommended)

1. **Install Heroku CLI**
   ```bash
   # Install from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   cd backend
   heroku create ml-playground-backend-[your-username]
   ```

3. **Add Procfile**
   ```bash
   echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Add Procfile for Heroku"
   git push heroku main
   ```

5. **Update Frontend Config**
   - Edit `frontend/config.js`
   - Replace `https://your-backend-deployment.herokuapp.com` with your Heroku app URL

### Option 2: Render

1. **Connect GitHub Repo**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Select the `backend` folder as root directory

2. **Configure Build**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Update Frontend Config**
   - Edit `frontend/config.js` 
   - Update the production API_BASE_URL with your Render URL

### Option 3: Vercel

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy Backend**
   ```bash
   cd backend
   vercel --prod
   ```

3. **Update Frontend Config**
   - Edit `frontend/config.js`
   - Update with your Vercel URL

## Environment Configuration

The frontend automatically detects the environment:
- **Localhost**: Uses `http://localhost:8000` (development)
- **Deployed**: Uses the production API URL you specify

## CORS Configuration

Make sure your backend includes CORS headers for your frontend domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing Deployment

1. Deploy your backend using one of the options above
2. Update `frontend/config.js` with the correct backend URL
3. Push changes to GitHub
4. Your frontend will automatically deploy to GitHub Pages
5. Visit your GitHub Pages URL to test

## Troubleshooting

- **CORS Errors**: Ensure backend allows your frontend domain
- **API Not Found**: Check backend URL in `config.js`
- **Local Development**: Use `npm run dev` or `python -m http.server 8080` in frontend folder
