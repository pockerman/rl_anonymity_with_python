coverage run -m --source=.  unittest discover src/tests/
coverage report -m
addopts = --cov=quant_ds_interview_task --cov-report=term-missing
