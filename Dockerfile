FROM python:3.10-bullseye
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install onnxruntime==1.15.1 insightface==0.7.3
RUN pip install -r requirements.txt
COPY . .
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
