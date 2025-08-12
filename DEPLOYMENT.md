# 🚀 Deployment Guide - AudioNorm Backend

This guide covers deploying the AudioNorm Backend to various platforms, with a focus on Render.com.

## 🌐 Render.com Deployment (Recommended)

### Quick Deploy with render.yaml

1. **Fork/Clone the Repository**
   ```bash
   git clone https://github.com/hoangphat25092005/AudioNorm_Backend.git
   cd AudioNorm_Backend
   ```

2. **Connect to Render**
   - Go to [Render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New" → "Web Service"
   - Connect your repository

3. **Auto-Deploy with render.yaml**
   - Render will automatically detect the `render.yaml` file
   - Click "Apply" to use the configuration
   - The service will be created with all the correct settings

### Manual Render Deployment

If you prefer manual setup:

1. **Create Web Service**
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

2. **Environment Variables**
   Set these in the Render dashboard:
   ```
   # Required
   MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
   SECRET_KEY=your-super-secret-key-here
   
   # Optional - Google OAuth
   client_id=your-google-client-id
   client_secret=your-google-client-secret
   authorised_origins=https://your-app.onrender.com
   authorised_redirect=https://your-app.onrender.com/auth/google/callback
   
   # Optional - Email
   MAIL_USERNAME=your-email@gmail.com
   MAIL_PASSWORD=your-app-password
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_FROM_NAME=AudioNorm Platform
   
   # Production
   ENVIRONMENT=production
   ```

3. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Your API will be available at `https://your-app.onrender.com`

## 🐳 Docker Deployment

### Build and Run Locally
```bash
# Build the image
docker build -t audionorm-backend .

# Run the container
docker run -p 8000:8000 \
  -e MONGO_URI="your-mongo-uri" \
  -e SECRET_KEY="your-secret-key" \
  audionorm-backend
```

### Deploy to any Docker platform
- Railway
- Fly.io
- Digital Ocean App Platform
- AWS ECS
- Google Cloud Run

## ☁️ Other Platforms

### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set MONGO_URI="your-mongo-uri"
heroku config:set SECRET_KEY="your-secret-key"
git push heroku main
```

### Railway
1. Connect GitHub repository
2. Set environment variables
3. Deploy automatically

### Vercel (Serverless)
```bash
npm i -g vercel
vercel --prod
```

## 🔧 Production Checklist

### Before Deployment
- [ ] Update CORS origins in `app/main.py`
- [ ] Set strong `SECRET_KEY`
- [ ] Configure MongoDB Atlas (not local MongoDB)
- [ ] Set up email credentials (optional)
- [ ] Update OAuth redirect URLs
- [ ] Test all endpoints locally

### After Deployment
- [ ] Test API endpoints at `/docs`
- [ ] Verify database connection
- [ ] Test authentication flow
- [ ] Test email notifications (if configured)
- [ ] Monitor logs for errors

## 🌍 Environment Variables Reference

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.net/` |
| `SECRET_KEY` | JWT secret key | `your-super-secret-key` |

### Optional
| Variable | Description | Default |
|----------|-------------|---------|
| `ALGORITHM` | JWT algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry | `30` |
| `ENVIRONMENT` | Environment type | `development` |

### Google OAuth (Optional)
| Variable | Description |
|----------|-------------|
| `client_id` | Google OAuth client ID |
| `client_secret` | Google OAuth client secret |
| `authorised_origins` | Allowed origins |
| `authorised_redirect` | OAuth callback URL |

### Email (Optional)
| Variable | Description |
|----------|-------------|
| `MAIL_USERNAME` | SMTP username |
| `MAIL_PASSWORD` | SMTP password/app password |
| `MAIL_SERVER` | SMTP server |
| `MAIL_PORT` | SMTP port |
| `MAIL_FROM_NAME` | Email sender name |

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure `PYTHONPATH` includes app directory
   - Check all dependencies in requirements.txt

2. **Database Connection**
   - Verify MongoDB URI format
   - Check network access in MongoDB Atlas

3. **CORS Issues**
   - Update allowed origins in production
   - Check frontend domain configuration

4. **Email Not Working**
   - Verify SMTP credentials
   - Check if email service is configured

### Health Check
Your API includes a health endpoint:
```
GET https://your-app.onrender.com/
```

### Logs
Monitor deployment logs in your platform dashboard for troubleshooting.

## 📊 Performance Tips

1. **Use uvicorn with workers in production**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4
   ```

2. **Enable connection pooling for MongoDB**

3. **Use Redis for caching (optional)**

4. **Monitor with APM tools**

## 🔒 Security Notes

- Never commit `.env` files
- Use strong, unique secret keys
- Enable HTTPS in production
- Restrict CORS origins
- Use environment variables for secrets
- Regular security updates

---

Happy Deploying! 🚀
