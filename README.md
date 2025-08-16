## 🧠 Deep Learning Model (CNN)

AudioNorm leverages a custom Convolutional Neural Network (CNN) for audio normalization. The model predicts the optimal gain to achieve a target LUFS (loudness) level for uploaded audio files.

- **Model location:** [`app/services/audio_service.py`](app/services/audio_service.py) and [`AudioNorm_DL/models/norm_cnn.pth`](AudioNorm_DL/models/norm_cnn.pth)
- **Architecture:**
  - Multiple convolutional layers for feature extraction from mel-spectrograms
  - Fully connected layers for regression (gain prediction)
  - Trained on music/audio datasets for robust normalization
- **Usage:**
  - The backend loads the CNN model at startup and uses it for all normalization requests
  - You can retrain or update the model using scripts in `AudioNorm_DL/`

**Example (Python):**
```python
# See app/services/audio_service.py for full details
class AudioNormCNN(nn.Module):
    def __init__(self, n_mels=64, additional_features_dim=9):
        ...
```


# AudioNorm Backend

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green?logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-async-green?logo=mongodb)
![License](https://img.shields.io/github/license/hoangphat25092005/AudioNorm_Backend)

> **AudioNorm Backend** is a modern FastAPI backend for audio normalization, user authentication (local & Google OAuth), feedback with email notifications, and MongoDB persistence. Built for speed, security, and extensibility.

## 🚀 Features

- 🔒 JWT authentication (local & Google OAuth)
- 📧 Feedback system with email notifications
- 🎧 Audio normalization API (LUFS, preview, export)
- 🗄️ MongoDB async storage (Motor)
- 📄 Interactive API docs (Swagger/OpenAPI)
- 🐳 Docker & Render deployment ready

## 📂 Project Structure

```text
AudioNorm_Backend/
├── app/
│   ├── main.py                # FastAPI app entrypoint
│   ├── config/                # DB, email, JWT, OAuth config
│   ├── controllers/           # API route controllers
│   ├── models/                # Pydantic & DB models
│   └── services/              # Business logic
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build
├── render.yaml                # Render deployment config
├── start.sh                   # Entrypoint script
└── DEPLOYMENT.md              # Deployment guide
```

## 🛠️ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/hoangphat25092005/AudioNorm_Backend.git
cd AudioNorm_Backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your .env (see .env.example)

# 4. Run the server
uvicorn app.main:app --reload
```

## 📑 API Documentation

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## 🧩 Main Components

- **Authentication:** JWT, Google OAuth, password reset
- **Audio Normalization:** Upload, normalize, preview, export (LUFS)
- **Feedback:** User feedback, threaded responses, email notifications
- **User Management:** Registration, profile, password reset

## 📝 Example Request

```bash
# Normalize audio (POST)
curl -X POST "http://localhost:8000/audio/normalize/-14" \
  -H "Authorization: Bearer <token>" \
  -F "file=@your_audio.wav"
```

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
- MongoDB (Atlas or local)
- SMTP credentials (optional, for email notifications)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment

Create a .env file (see [.env.example](.env.example)). Important notes about variable names:

- Mongo connection in code reads MONGODB_URL in [app/config/database.py](app/config/database.py)
  - Set MONGODB_URL=mongodb+srv://... (recommended)
  - If you already use MONGO_URI from .env.example/render, set both to the same value
- JWT in [app/services/auth_service.py](app/services/auth_service.py):
  - SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
- Google OAuth in [app/config/oauth.py](app/config/oauth.py):
  - Uses GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (different from .env.example’s client_id/client_secret)
- Email in [app/config/email.py](app/config/email.py):
  - MAIL_USERNAME, MAIL_PASSWORD, MAIL_SERVER, MAIL_PORT, MAIL_FROM_NAME

Minimal example:
```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# Email (optional)
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_FROM_NAME=AudioNorm Platform

ENVIRONMENT=development
```

## Run locally

Development:
```bash
uvicorn app.main:app --reload
```

Production-like:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health:
- GET /           → basic info
- GET /health     → health check

## Docker

Build and run:
```bash
docker build -t audionorm-backend .
docker run -p 8000:8000 \
  -e MONGODB_URL="mongodb+srv://..." \
  -e SECRET_KEY="your-secret-key" \
  audionorm-backend
```

The container uses [Dockerfile](Dockerfile) and starts uvicorn with [app/main.py](app/main.py).

## Render deployment

Use [render.yaml](render.yaml) or follow [DEPLOYMENT.md](DEPLOYMENT.md).

## API overview

Auth: [app/controllers/auth_controller.py](app/controllers/auth_controller.py), [`app.services.auth_service.AuthService`](app/services/auth_service.py)
- POST /auth/register → Register new user (body: username, email, password, confirm_password)
- POST /auth/login → Login with username/password, returns JWT
- POST /auth/token → OAuth2PasswordRequestForm-compatible token endpoint
- GET /auth/google/login → Start Google OAuth
- GET /auth/google/callback → Complete Google OAuth

Users: [app/controllers/user_controller.py](app/controllers/user_controller.py), [`app.services.user_service.UserService`](app/services/user_service.py)
- GET /users/{user_id} → Get user by id

Feedback: [app/controllers/feedback_controller.py](app/controllers/feedback_controller.py), [`app.services.feedback_service.FeedbackService`](app/services/feedback_service.py)
- POST /feedback/submit → Submit feedback (auth required)
- POST /feedback/respond → Respond to feedback (auth required)
- GET /feedback/ → List all feedback with response counts
- GET /feedback/{feedback_id} → Get one feedback with responses
- GET /feedback/user/my-feedback → List current user’s feedback (auth required)

## Usage examples

Register:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo",
    "email": "demo@example.com",
    "password": "P@ssw0rd!",
    "confirm_password": "P@ssw0rd!"
  }'
```

Login:
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"P@ssw0rd!"}'
```

Submit feedback (requires Bearer token):
```bash
TOKEN="paste-access-token-here"
curl -X POST http://localhost:8000/feedback/submit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"feedback_text":"Loving the platform so far","rating":5}'
```

Respond to feedback:
```bash
curl -X POST http://localhost:8000/feedback/respond \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"feedback_id":"<feedbackId>","response_text":"Thanks for the feedback!"}'
```

List feedback:
```bash
curl http://localhost:8000/feedback/
```

## Email notifications

- Confirmation sent on submit (user receives email) via [`app.services.email_service.EmailService.send_feedback_confirmation`](app/services/email_service.py)
- Notification sent to feedback author on new response via [`app.services.email_service.EmailService.send_feedback_notification`](app/services/email_service.py)
- Email sending is a no-op if credentials are not configured in [app/config/email.py](app/config/email.py)

## CORS

CORS is configured in [app/main.py](app/main.py). In development, all origins are allowed. In production, update allowed_origins accordingly.

## Notes and known issues

- JWT payload mismatch:
  - [`app.config.jwt_dependency.get_current_user`](app/config/jwt_dependency.py) expects a claim user_id
  - [`app.services.auth_service.AuthService.create_access_token`](app/services/auth_service.py) sets sub
  - Align these (e.g., encode {"user_id": "..."} or read sub in the dependency)
- Mongo env var naming is inconsistent:
  - Code reads MONGODB_URL ([app/config/database.py](app/config/database.py)), while .env.example/render use MONGO_URI
  - Set MONGODB_URL to avoid connection issues (or update the code to read MONGO_URI)
- Google OAuth env naming:
  - [app/config/oauth.py](app/config/oauth.py) uses GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET, while .env.example uses client_id/client_secret
- User controller/model mismatch:
  - [app/controllers/user_controller.py](app/controllers/user_controller.py) builds a response with _id but [`app.models.user.UserResponse`](app/models/user.py) has no id field
- Feedback user_id type:
  - Feedback stores user_id as string on submit, but [get_user_feedback](app/services/feedback_service.py) matches ObjectId(user_id); make types