FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy everything needed for runtime
COPY app.py .
#COPY model.pkl .
COPY checkpoints/ ./checkpoints/
COPY templates/ ./templates/
COPY static/ ./static/

#optional
#COPY create_table.py .

#EXPOSE 8000

#CMD ["python", "create_table.py"]

CMD ["uvicorn", "app:app"]
