.PHONY : build reqs install clean
NINJA := $(shell command -v ninja 2> /dev/null)


build : reqs
	python setup.py bdist_wheel

reqs :
ifndef NINJA 
	sudo cp ./ninja /usr/bin
endif 
	pip install -r requirements.txt

install :
	pip install --upgrade dist/*.whl

clean :
	-rm -rf build dist/* *.egg-info
