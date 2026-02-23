FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import werkzeug; print(werkzeug.__version__)"
RUN pip install --no-cache-dir --force-reinstall Flask==2.2.3 Werkzeug==2.2.2
RUN python -c "import werkzeug; print(werkzeug.__version__)"
COPY . .
CMD ["python", "app.py"]