clean:
	rm -rf ./*/__pycache__
	rm -rf ./*/*.pyc
	rm -rf *.csv
	rm -rf ./*/*.csv
	rm -rf *.log
	rm -rf ./*/*.log

test:
	pytest

install:
	pip install --user .

sim_wer_hard:
	python3 -u src/mlg/sim_wer.py HARD | tee hard_sim_wer.log

sim_wer_soft:
	python3 -u src/mlg/sim_wer.py | tee soft_sim_wer.log

plot_wer:
	python3 -u src/mlg/plot_wer.py | tee plotwer.log
