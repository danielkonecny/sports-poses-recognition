# Self-Supervised Learning for Sports Pose Recognition

Master's Thesis at Brno University of Technology - Faculty of Information Technology.

Author: Daniel Konecny (xkonec75).

## Dependencies

* Python 3.8.
* Libraries (in correct versions) provided in `requirements.txt`.

## How to install

1. Clone this repository: `git clone https://github.com/danielkonecny/sports-poses-recognition.git`.
2. Enter the repository root directory: `cd sports-poses-recognition`.
3. Create a virtual environment: `python3 -m venv .env`.
4. Activate the virtual environment: `source ".env/bin/activate"`.
5. Install the requirements: `pip install -r requirements.txt`.

## How to launch

1. Enter the project root directory `sports-poses-recognition`.
2. Activate the virtual environment: `source ".env/bin/activate"`.
3. Make sure you have the project root directory in `PYTHONPATH` because of relative imports in the scripts: `export PYTHONPATH="${PYTHONPATH}:."`
4. Launch all scripts from the root directory, e.g. `python3 src/model/Encoder.py ...`
5. Use argument `-h` or `--help` for all the necessary information about every script.
