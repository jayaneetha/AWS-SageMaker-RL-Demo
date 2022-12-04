ARG AWS_REGION

FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-tf-cpu-py37

RUN pip install --upgrade pip

RUN pip install gym[box2d]


# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install sagemaker-containers --upgrade

ENV PYTHONUNBUFFERED 1

# Make things a bit easier to debug
WORKDIR /opt/ml/code

RUN pip freeze 

RUN python --version
