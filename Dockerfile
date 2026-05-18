FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install py dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY . .

#Expose port 8000
EXPOSE 8000

# MLflow and AWS config
ENV MLFLOW_TRACKING_URI=http://3.19.56.68:5000
ENV AWS_DEFAULT_REGION=us-east-2
ARG GROQ_API_KEYs
ENV GROQ_API_KEY=$GROQ_API_KEY

#Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
