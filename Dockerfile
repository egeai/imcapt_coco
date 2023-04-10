FROM python:3.11

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . /app
# EXPOSE 8000

CMD [ "python3", "./src/data/make_dataset.py" ]
# ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
#CMD python -m uvicorn main:app --host 0.0.0.0 --port 8000

# ENTRYPOINT [ "bash"]