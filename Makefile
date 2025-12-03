install:
	python -m venv .venv
	. .venv/Scripts/activate && pip install -r requirements.txt

run:
	. .venv/Scripts/activate && FLASK_APP=src.app:app flask run
