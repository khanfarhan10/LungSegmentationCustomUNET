virtualenv LungUNETCPUEnv


LungUNETCPUEnv\Scripts\activate
deactivate
conda.bat deactivate
pip install -r requirements.txt
pip freeze > requirements.txt
pip list -> full_reqs.txt

python lungunetmodel.py
ipython kernel install --user --name=MonteCarlo
