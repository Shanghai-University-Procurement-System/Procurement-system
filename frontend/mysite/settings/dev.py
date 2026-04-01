from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-ny)k7ub@qrq7we#(#4_=6o=75(1&29ypet(nf)d=+u!%$+iqub"

# SECURITY WARNING: define the correct hosts in production!
ALLOWED_HOSTS = ["*"]

# EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"


try:
    from .local import *
except ImportError:
    pass

# Allow intranet penetration domain (CSRF protection)
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://*.vicp.fun", # Peanut Hull (花生壳)
    "http://*.vicp.fun",
]
