FROM python:3.9-slim

# Prepare Python Environment
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip cache purge

# Copy Files & Preparing Package
WORKDIR /usr/src/image-cap
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/image-cap"
COPY . .

# Run Celery Worker
# CMD python3.10 worker/tasks.py
CMD bash