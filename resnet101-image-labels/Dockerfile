FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/morawskim/ai-examples"
WORKDIR /usr/src/app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
     libmagic1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY web.py imagenet_classes.txt ./
COPY ./templates ./templates
EXPOSE 5000
CMD ["python", "web.py"]
