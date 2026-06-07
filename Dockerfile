FROM python:3.11-slim
ENV PYTHONIOENCODING=utf-8
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "bot.py"]
