clean:
	rm -rf ./*/__pycache__
	rm -rf ./*/*.pyc

test:
	pytest
