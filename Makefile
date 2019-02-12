clean:
	rm -rf ./*/__pycache__
	rm -rf ./*/*.pyc
	rm -rf *.csv
	rm -rf ./*/*.csv

test:
	pytest

install:
	pip install --user .

sim_wer_hard:
	python3 src/mlg/sim_wer.py HARD

sim_wer_soft:
	python3 src/mlg/sim_wer.py
