FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY stress_test_data/ stress_test_data/

CMD ["python", "./src/main.py", "--train_data_filepath", "stress_test_data/train.csv", "--test_data_filepath", "stress_test_data/test.csv"]
