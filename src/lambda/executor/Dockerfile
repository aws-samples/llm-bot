FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "/var/task"

COPY main.py /var/task
COPY ./utils/ /var/task/

CMD [ "main.lambda_handler" ]
