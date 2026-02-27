FROM python:3.10-slim
WORKDIR /workspace
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -c "import werkzeug; print(werkzeug.__version__)"
RUN pip install --no-cache-dir --force-reinstall Flask==2.2.3 Werkzeug==2.2.2
RUN python -c "import werkzeug; print(werkzeug.__version__)"
COPY . .
CMD ["python", "app/app.py"]