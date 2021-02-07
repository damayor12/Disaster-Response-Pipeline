# Disaster Response Pipeline Project
The Disaster Response Pipeline is an NLP machine learning algorithm that ingests and processes aid messages received by emergency response services. These messages are then categorized by message content and displayed in a convenient to use web application.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing
This project is apart of Udacity's Data Science Nanodegree Program, which provides initial starter code for the project. Additionally, the original datasets are provided in part by FigureEight.