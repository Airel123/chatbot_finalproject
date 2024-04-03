# chatbot_finalproject
## Some NOTES

There are two branches in the repository now. The 'main' branch is what I used to complete the whole process(model training, testing, GUI integrated) . The other branch named 'word2vec' is a new start that follows your suggestion of creating a new file by employing the word2vec for embedding, I am still trying to integrate it into my original code more tests need to be done yet, so you can only see the completed word2vec preprocessing code (wv_preprocessing.py). I will upload the rest of the part code when I finish testing.

## Description

The chatbot is a generative chatbot in mental stress field which utilizes a Seq2Seq with Luong attention mechanism to understand and respond to user input. 

## Dataset Composition

The dataset is in csv format (finaldataset.csv / nocounsel.csv)
It is a custom dataset souced from three parts:
- `general query` :  155 dialogues were sourced. These dialougues are centered around foundational mental health discussions and classical therapy discourse.
- `counseling data` :   1598 dialogues. These dialogues are primarily Q&A pairs derived from online counseling platforms.
- `LLM expanding data` :  81 dialogues were sourced. These dialogues are generated with the help of GPT-4 under the human guidence.

## Structure

Below is the organization and description of the files and directories within this project:

- `app.py`: The Flask server that serves the chatbot's web interface.

- `model.py`: Defines the chatbot model, including the architecture and the logic for processing and responding to user input.

- `parameter_op.py`: Manages the hyperparameter settings used by the chatbot model with the help of optuna framework.
  
- `preprocessing.py`: Prepares the input data for the chatbot, including text cleaning and word representing before it is passed to the model.

- `train.py`: Contains the code to train the chatbot model using training datasets.

- `test.py`: Provides test cases to validate the functionality and performance of the chatbot, ensuring that the chat flows correctly and the model's responses are as expected.

- `readme.txt`: This is a basic package setup guidence.

- `static`: This directory is used to serve static files in a Flask application.
  - `images`: Images used in the chatbot's web interface.
  - `app.js`: The JavaScript file.
  - `chat_style.css`: The stylesheet.

- `templates`: Contains HTML templates for the Flask application.
  - `chat_window.html`: The HTML template for the chatbot's conversation window, where users interact with the bot.




  

