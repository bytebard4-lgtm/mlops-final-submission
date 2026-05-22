## Branching Strategy

main        → production-ready code  
develop     → integration branch  
feature/*   → experimental features  

Workflow:

feature/new-model → develop → main

This strategy ensures stable production releases while allowing experimentation and testing and its one direction only.

For locally testing purposes here are the insructions and lines of code for simple functionality checks:
Start by making sure you're in the right place and everything is set up:
cd ~/mlops-final-submission

export PYTHONPATH=$(pwd)

pip install -r requirements.txt

Just explain you're checking the code follows Python style rules:
flake8 src app --max-line-length=100

Run the tests:
pytest tests -v --tb=short

Now we train the models, this is where the three candidates get trained and the best one gets saved:
python src/train_models.py

Check the model file actually got created:
ls -lh models/best_model.pkl

Open a second terminal and launch MLflow — leave it running and open the browser at http://127.0.0.1:5000 to show the experiment runs, the accuracy of each model and which one won:
mlflow ui --backend-store-uri file:./mlruns

Back in the first terminal, show the CT pipeline working — you fake a data change by touching the CSV and then the script detects the hash is different and retrains:
echo "" >> data/iris_custom.csv
python src/check_data_and_retrain.py

Start the Flask API:
python app/app.py
Open a third terminal and hit the two endpoints:

curl http://localhost:5002/health

curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.3, 3.3, 6.0, 2.5]}'
