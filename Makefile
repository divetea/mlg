clean:
	rm -rf ./*/__pycache__
	rm -rf ./*/*.pyc
	rm -rf *.csv
	rm -rf ./*/*.csv

test:
	pytest
