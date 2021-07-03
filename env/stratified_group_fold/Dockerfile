FROM python:3.7 AS base

WORKDIR /workspace/deployment
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace/stratified_group_fold
