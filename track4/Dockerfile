FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY code/ code/
COPY scripts/ scripts/
COPY reference/student_package/ reference/student_package/
COPY weights/ weights/
COPY run.sh .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["bash", "./run.sh"]
