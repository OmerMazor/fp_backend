FROM python:3.11-slim

# מערכת + כרום
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates fonts-liberation libasound2 libatk1.0-0 libatk-bridge2.0-0 \
    libx11-6 libxcb1 libxcomposite1 libxcursor1 libxi6 libxdamage1 libxext6 libxfixes3 \
    libxrandr2 libgbm1 libgtk-3-0 libnss3 libnspr4 libdrm2 libxshmfence1 unzip \
 && rm -rf /var/lib/apt/lists/*

# התקנת Google Chrome יציב
RUN mkdir -p /usr/share/keyrings && \
    wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
      | tee /usr/share/keyrings/google-linux-signing-keyring.gpg >/dev/null && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
      > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROME_BIN=/usr/bin/google-chrome

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render מספק PORT כ־ENV. נקשיב עליו.
# חשוב: timeout גבוה כי סקרייפר לוקח זמן.
CMD ["bash", "-lc", "exec gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT} --timeout 180 --workers 2"]
