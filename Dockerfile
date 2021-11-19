FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./data/CMAPSS /app/data/CMAPSS
COPY ./models /app/models
COPY app_FastAPI.py /app
COPY RUL_BiLSTM_CMAPSS.py /app

EXPOSE 5000
CMD ["uvicorn", "app_FastAPI:app", "--host", "0.0.0.0", "--port", "5000"]