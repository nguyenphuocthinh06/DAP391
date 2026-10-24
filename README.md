# Mental Health Counseling Chatbot

**A context-aware mental health support chatbot combining intent
classification and a local Large Language Model (LLM).**

## Overview

This group project prototypes a chatbot designed to respond to users
seeking mental health support. It combines a machine-learning intent
classifier with an LLM so that the system can identify the likely intent
of a message and generate a concise, empathetic response grounded in
predefined responses associated with that intent.

The notebook is designed to run in **Google Colab**, uses a JSON intent
dataset stored in Google Drive, and serves the `llama3.2:3b` model
locally within the Colab runtime through **Ollama**.

> **Important:** This is an educational prototype, not a substitute for
> a licensed mental health professional. It must not be used to diagnose
> conditions, provide emergency care, or make clinical decisions.
> Generated responses may be inaccurate. If someone may be in immediate
> danger, they should contact local emergency services or a qualified
> professional.

## Key Features

-   **Intent classification:** Predicts the intent associated with a
    user's message.
-   **Text embeddings:** Converts text into vectors using `BAAI/bge-m3`.
-   **Model comparison and tuning:** Uses `GridSearchCV` to tune several
    classifiers.
-   **Stacking ensemble:** Combines SVM, Random Forest, LightGBM, and
    XGBoost predictions using Logistic Regression as the final
    estimator.
-   **Automatic model selection:** Selects the candidate with the
    highest weighted F1 score on the held-out test split.
-   **LLM-generated replies:** Uses Ollama to call `llama3.2:3b` and
    generate a short response based on predefined response context for
    the predicted intent.
-   **Out-of-scope confidence gate:** Rejects a message when the
    classifier's maximum predicted probability is below a configurable
    threshold.
-   **Evaluation and error analysis:** Reports Accuracy, weighted F1,
    weighted Recall, and weighted Precision, and examines rejected
    examples and intent confusions.

## Architecture

``` text
User message
    |
    v
BGE-M3 text embedding
    |
    v
Selected ML classifier / Stacking ensemble
    |
    v
Maximum predicted probability
    |
    +---- Below threshold (0.60) ----> Out-of-scope fallback message
    |
    +---- At or above threshold
                    |
                    v
             Predicted intent
                    |
                    v
       Predefined responses for that intent
                    |
                    v
       Context-grounded prompt sent to Ollama
                    |
                    v
              Llama 3.2 3B
                    |
                    v
             Chatbot response
```

The notebook uses the first two predefined responses for the predicted
intent as context. The LLM prompt instructs the model to answer only
from that context and keep the response natural and concise.

## Technologies

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  Google Colab                        Notebook execution environment

  Google Drive                        Stores and provides access to the
                                      intent dataset

  Sentence Transformers               Loads the BGE-M3 embedding model

  BGE-M3 (`BAAI/bge-m3`)              Converts user messages and training
                                      examples into embeddings

  scikit-learn                        Data splitting, label encoding,
                                      model tuning, stacking, and
                                      evaluation

  SVM (`SVC`)                         Candidate intent classifier

  Random Forest                       Candidate intent classifier

  LightGBM                            Candidate intent classifier

  XGBoost                             Candidate intent classifier

  Logistic Regression                 Meta-classifier for the stacking
                                      ensemble

  Ollama                              Runs and exposes the Llama model
                                      through a local API

  Llama 3.2 3B (`llama3.2:3b`)        Generates the final
                                      natural-language response

  ipywidgets                          Provides the interactive chat
                                      interface in the notebook

  PyTorch                             Selects CUDA when available for
                                      embedding generation
  -----------------------------------------------------------------------

## Dataset

The notebook expects a JSON file at this Google Drive path:

``` text
/content/drive/MyDrive/Project DAP/Gen_dataset_v6.json
```

The code expects the JSON data to contain an `intents` collection. Each
intent entry should provide:

-   `tag`: the intent label.
-   `patterns`: example user messages for training.
-   `responses`: predefined responses used as context for response
    generation.

Example of the expected structure:

``` json
{
  "intents": [
    {
      "tag": "example_intent",
      "patterns": [
        "Example user message",
        "Another way to express the same intent"
      ],
      "responses": [
        "Example contextual response",
        "Another contextual response"
      ]
    }
  ]
}
```

This is an illustrative schema, not a copy of the project's actual
dataset. Make sure your dataset follows the structure expected by the
notebook. The dataset itself is not included in this repository unless
the group adds it separately.

## How It Works

### 1. Load and prepare the data

The notebook reads each intent's `patterns` and associates every pattern
with its corresponding `tag`. It encodes the intent labels into integers
and creates a stratified 80/20 train-test split using
`random_state=123`.

### 2. Generate text embeddings

`BAAI/bge-m3` transforms the training and test messages into vector
representations. The notebook uses CUDA when PyTorch reports that a CUDA
device is available; otherwise, it uses the CPU.

### 3. Tune candidate classifiers

`GridSearchCV` with three-fold cross-validation and weighted F1 scoring
searches the following configurations:

-   **SVM (Linear):** `C` values of `0.1` and `1.0`, with a linear
    kernel.
-   **Random Forest:** `n_estimators` values of `100` and `200`, with
    `max_depth` set to `None` or `10`.
-   **LightGBM:** `n_estimators=100`, with learning rates of `0.05` and
    `0.1`.
-   **XGBoost:** `n_estimators=100`, `learning_rate=0.1`, and
    `max_depth=3`.

The best estimator from each search is then used as a base estimator in
the stacking ensemble. Logistic Regression (`max_iter=1000`) acts as the
final estimator, with three-fold cross-validation.

### 4. Evaluate and select the model

Each candidate, including the stacking classifier, is evaluated on the
held-out test split using:

-   Accuracy
-   Weighted F1 score
-   Weighted Recall
-   Weighted Precision

The model with the highest weighted F1 score is selected for the
chatbot. Actual scores depend on the dataset and runtime and should be
copied from the notebook output after running it; this README does not
claim unobserved results.

### 5. Handle the conversation

When a user submits a message, the system embeds it and obtains class
probabilities from the selected classifier.

-   If the maximum probability is **below `PROB_THRESHOLD = 0.60`**, the
    chatbot returns a fallback message saying the question is outside
    the system's current scope.
-   Otherwise, it maps the predicted class back to its intent tag,
    retrieves up to two predefined responses for that intent, and sends
    the user's message and retrieved context to Llama 3.2 3B through the
    Ollama API.
-   The generated reply is displayed in the notebook chat interface. The
    notebook also displays the predicted intent and confidence for
    debugging.

The threshold is a configurable heuristic, not a guarantee that a
message is safe, relevant, or correctly understood.

### 6. Analyze classification errors

The final notebook cell examines predictions from the selected model and
reports examples rejected by the threshold, common confusions between
intents, and intents with low F1 scores. These diagnostics can help the
group identify weak labels or examples that need improvement.

## Requirements

-   A Google account with access to Google Colab and Google Drive.
-   The intent dataset at the path described above.
-   A Colab runtime with sufficient memory and storage for the embedding
    and language models.
-   Internet access in the runtime to install packages and download
    models.
-   Ollama installed and running in the same runtime as the notebook.
-   A compatible runtime for PyTorch and the selected embedding device.

The notebook installs several dependencies and Ollama in its setup cell.
The exact versions are not pinned in the current notebook, so package or
runtime updates may affect reproducibility.

## Setup and Usage

1.  Upload the notebook to Google Colab.

2.  Place `Gen_dataset_v6.json` in Google Drive at:

    ``` text
    MyDrive/Project DAP/Gen_dataset_v6.json
    ```

3.  Open the notebook and run the setup cell. It mounts Google Drive,
    installs the required Python packages and Ollama, starts the Ollama
    service, downloads `llama3.2:3b`, and sends a warm-up request.

4.  Run the main training cell. It loads the dataset, generates
    embeddings, tunes the classifiers, trains the stacking ensemble,
    evaluates candidates, selects the model, and creates the chat
    interface.

5.  Enter a message in the chat input and click **Send** (or press
    Enter) to test the chatbot.

6.  Run the error-analysis cell after the training cell has completed to
    inspect threshold rejections and intent-classification errors.

**Note:** The training cell must run before the error-analysis cell
because the latter uses variables created during training, including
`best_model_name`, `final_ml_model`, `X_test_vectors`, and `y_test`.

## Configuration

The following settings are defined in the notebook:

  ---------------------------------------------------------------------------------------
  Setting                 Current value                           Description
  ----------------------- --------------------------------------- -----------------------
  `PROB_THRESHOLD`        `0.60`                                  Minimum maximum-class
                                                                  probability required
                                                                  before generating an
                                                                  LLM response

  `OLLAMA_URL`            `http://localhost:11434/api/generate`   Ollama generation
                                                                  endpoint

  `OLLAMA_MODEL_NAME`     `llama3.2:3b`                           LLM used for response
                                                                  generation

  Test size               `0.2`                                   Fraction of examples
                                                                  held out for testing

  Split random state      `123`                                   Reproducible train-test
                                                                  split

  Stacking CV             `3`                                     Number of folds used by
                                                                  the stacking classifier
  ---------------------------------------------------------------------------------------

## Limitations and Future Improvements

The current notebook is a prototype and has several areas that could be
improved:

-   **Mental health safety:** Add carefully designed crisis-response
    handling, clear disclaimers, and escalation guidance reviewed by
    qualified professionals.
-   **Intent coverage:** Expand and validate the dataset, especially for
    ambiguous or underrepresented intents.
-   **Threshold calibration:** Tune and validate the confidence
    threshold on a separate validation set rather than relying on a
    fixed value.
-   **Evaluation:** Add per-intent metrics, a confusion matrix, and
    tests with genuinely out-of-scope messages. The current held-out
    test split appears to contain examples drawn from the known intent
    classes, so it alone does not validate out-of-scope rejection.
-   **Response grounding:** Improve context retrieval and verify that
    generated responses remain faithful to the approved content.
-   **Reproducibility:** Pin dependency versions, document runtime
    requirements, and save model artifacts and evaluation results.
-   **Privacy:** Avoid collecting unnecessary sensitive information, and
    establish clear data retention and access practices before any
    real-world deployment.
-   **Deployment:** Replace the notebook-only widget interface with a
    dedicated application only if the group needs a standalone user
    interface.

## Responsible Use

This project is intended for learning and experimentation. It does not
establish a therapeutic relationship and should not be represented as
professional counseling. Do not rely on it as the sole source of support
in a crisis. Any deployment involving real users should undergo privacy,
security, clinical-safety, and reliability review.

## Project Structure

The uploaded source is a Jupyter notebook. A possible repository layout
is:

``` text
mental-health-chatbot/
├── README.md
├── sourcecode(2).ipynb
└── data/
    └── Gen_dataset_v6.json   # optional; add only if permitted
```

The `data/` folder is illustrative; the notebook currently reads the
dataset from Google Drive rather than from a local repository folder. Do
not publish private or restricted dataset contents without permission.

## Acknowledgements

This project uses open-source machine-learning libraries and pretrained
models, including BGE-M3 and Llama 3.2 3B. Follow the respective model,
software, and dataset licenses and usage terms.
