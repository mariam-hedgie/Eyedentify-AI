# Gunicorn configuration. Gunicorn auto-loads this file from the working
# directory, so these settings apply whether the app is started via the
# Dockerfile CMD or a start command like `gunicorn app:app`.
#
# timeout: model inference on the HF Space can take well over gunicorn's
# 30s default, which would otherwise kill the worker and return a 500/502.
timeout = 240

# Keep memory low on Render's free tier (512 MB); this app just proxies to
# the HF Space, so a couple of workers is plenty.
workers = 2
bind = "0.0.0.0:5000"
