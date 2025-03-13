# Week 1: Data Collection and Preprocessing

1. Collecting Reviews (200+ examples in TXT, CSV, JSON)

The dataset used for this project consists of Ukrainian-language reviews from various websites. Dataset link: https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews.

We selected 600 Ukrainian reviews from the Rozetka website.

2. Text Cleaning: Stop-word Removal, Lemmatization, Normalization 

Text cleaning was performed using a function that included stop-word removal, lemmatization, and normalization.

3. Entity Recognition and Annotation (Price/Discounts, Issues, Product/Service Names)

Key entities (product, price, complaints) and sentiment were extracted. We used the Mistral API and numerical ratings left by reviewers. Some entities were annotated manually.  


# Week 2: Sentiment Analysis 

1. Sentiment Classification Model (Positive, Neutral, Negative)

We performed sentiment classification using classical machine learning algorithms such as Logistic Regression, Naive Bayes, Random Forest, SVM, and KNN.

![image](https://github.com/user-attachments/assets/ff3502a6-c278-485c-9484-b9d206b2cf33)

However, results across all models were similar.

<img width="518" alt="image" src="https://github.com/user-attachments/assets/f8efda82-6306-4d83-b6f8-3c8bf206ec2c" />

The model incorrectly classified the "Neutral" class. Overall prediction accuracy showed no significant improvement over random guessing.

To improve results, we increased the number of examples for "Positive" and "Neutral" classes in the dataset.

<img width="1115" alt="image" src="https://github.com/user-attachments/assets/e38312e7-2203-45dc-970d-dc9aaaebc325" />

![image](https://github.com/user-attachments/assets/affee725-417a-43cf-8158-f2d35fc9d031)
<img width="530" alt="image" src="https://github.com/user-attachments/assets/9a90cae6-b129-4b0d-bec1-6862f5fbefe2" />

After augmentation, the improvement was minor, but the model no longer classified all examples as "Negative."

We then applied an LSTM model to improve classification accuracy.

![image](https://github.com/user-attachments/assets/eb06236d-1874-45eb-b8d2-eaa0a884fc49)

### Model Evaluation Metrics

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Negative| 0.73      | 0.35   | 0.47     | 63      |
| Neutral | 0.32      | 0.56   | 0.40     | 36      |
| Positive| 0.66      | 0.73   | 0.69     | 52      |
| **Accuracy** |         |        | **0.53**  | 151     |
| **Macro Avg** | 0.57  | 0.55   | 0.52     | 151     |
| **Weighted Avg** | 0.61  | 0.53   | 0.53     | 151     |


To further improve results, we added text representation using TF-IDF and n-grams:

![image](https://github.com/user-attachments/assets/3eab61cc-1376-4de4-9f10-7f929a86296f)

### Model Evaluation Metrics

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Negative| 0.69      | 0.73   | 0.71     | 62      |
| Neutral | 0.52      | 0.47   | 0.49     | 36      |
| Positive| 0.68      | 0.68   | 0.68     | 53      |
| **Accuracy** |         |        | **0.65**  | 151     |
| **Macro Avg** | 0.63  | 0.63   | 0.63     | 151     |
| **Weighted Avg** | 0.65  | 0.65   | 0.65     | 151     |

Compared to the initial results (accuracy: 0.53), using TF-IDF and n-grams improved accuracy to 0.65. The classification performance for "Class 0" and "Class 2" improved significantly, but "Class 1" remains less accurately classified.


2. Sentiment Detection in REST API Response

The FastAPI implementation includes batch data processing and error handling (e.g., empty text input).

Example API output with one empty text:
```json
{
  "results": [
    {
      "text": "Чудовий продукт!",
      "sentiment": "Positive"
    },
    {
      "text": "Цей продукт має нормальні характеристики.",
      "sentiment": "Neutral"
    },
    {
      "text": "Жах! Не купуйте в цього продавця!",
      "sentiment": "Negative"
    }
  ],
  "notification": "1 empty text(s) were skipped."
}
```


# Week 3: Review Topic Classification, Key Phrase, and Question Extraction

1.  Identifying Main Topics (e.g., "Service", "Quality", "Delivery")

Для класифікації відгуків за темами ми використали BERTopic. Ця модель добре працює з неструктурованими текстовими даними, дозволяючи автоматично виявляти тематичні кластери у відгуках.

We used BERTopic for topic modeling. It helps identify meaningful thematic clusters in reviews.

Our approach involves grouping the detected topics into categories that are useful for sellers. For instance, if a review mentions discounts or high prices, we classify it under "Price," regardless of whether it concerns household goods, electronics, or other categories. Additional details, such as product type, are not critical since they can be extracted directly from the website catalog rather than through text analysis.

Identified key topics:

- Price – price, discounts, promotions.

- User Experience – personal impressions after use.

- Recommendations – whether the user recommends the product.

- Delivery/Service – evaluation of the delivery and service process.

- Quality – product quality, defects, reliability.

This approach helps focus on information most valuable for analysis and decision-making by sellers.


2. Extracting Key Complaints and Positive Aspects

We used aspect-based sentiment analysis (ABSA) via PyABSA (https://github.com/yangheng95/PyABSA) since it supports multiple languages, including Ukrainian. This model can extract key entities from text and perform sentiment analysis on them.

Example:
```text
Review: дуже задоволена покупкою. тканина дуже плотна. прошите гарно. наповнювач також всередині плотний. тепле однозначно. і водночас легке. готуємось до зими в умовах війни(((

aspects: ['тканина', 'наповнювач']
sentiments: ['Positive', 'Positive']
```

3. Automatic Question Detection in Text

We used spacy.load("uk_core_news_md") and keyword searches ("що", "чому", "де", "коли", "скільки", "звідки", etc.).

Example:
```text
Review:
Чому розетка не реагує на негативні відгуки та запитання? Дуже швидко перестав працювати основний брелок. Сенс двосторонньої сигналізації зник 

questions: [Чому розетка не реагує на негативні відгуки та запитання?]
```


# Week 4: Deployment in Docker, Testing, and Documentation

1. Deployment in Docker

Dockerfile

- Uses python:3.9-slim as base.

- Installs dependencies (gcc, g++, git, libomp5).

- Optimizes Python package installation.

- Downloads the uk_core_news_md spaCy model.

- Exposes port 8000 and runs FastAPI.

Docker Compose

- Defines api service.

- Maps ./ and ./models directories.

- Sets TZ=UTC.


2. Model Improvement

For more accurate results in sentiment classification, we tested the multilingual-sentiment-analysis model (https://huggingface.co/tabularisai/multilingual-sentiment-analysis). It outperformed the LSTM model:

![image](https://github.com/user-attachments/assets/3eec9055-9647-4435-9c15-7dabf626ffea)

### Model Evaluation Metrics

| Sentiment  | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Negative   | 0.84      | 0.68   | 0.75     | 62      |
| Neutral    | 0.50      | 0.61   | 0.55     | 36      |
| Positive   | 0.75      | 0.81   | 0.78     | 53      |
| **Accuracy** |         |        | **0.71**  | 151     |
| **Macro Avg** | 0.70  | 0.70   | 0.69     | 151     |
| **Weighted Avg** | 0.73  | 0.71   | 0.71     | 151     |


Compared to the previous LSTM model, the multilingual-sentiment-analysis model showed significant improvements in Precision and Recall for the "Negative" and "Positive" classes.

For the "Negative" class:

- Precision increased to 0.84 (compared to 0.69 in LSTM), indicating fewer false positives.
- Recall improved to 0.68 (compared to 0.35 in LSTM), meaning the model better detects negative cases.

For the "Positive" class:

- Precision rose to 0.75 (compared to 0.66 in LSTM), improving the recognition of positive reviews.
- Recall increased to 0.81 (compared to 0.73 in LSTM), showing better detection of positive cases.

For the "Neutral" class:

- Precision dropped to 0.50 (compared to 0.66 in LSTM), meaning more false positives for this class.
- Recall improved to 0.61 (compared to 0.35 in LSTM), indicating better recognition of neutral reviews.


5. Instructions for Running the API and Docker

## Running with Docker
1. Clone the repository

```python
git clone <repository_url>
cd <repository_directory>
```

2 Build the Docker image using docker-compose

```python
docker-compose up --build
```

3 Check if the API is running

Open a browser or use curl:
```python
curl http://localhost:8000/docs
```

4. Example API usage from the command line

To send text for analysis, use curl:
```terminal
curl -X 'POST' \
  'http://127.0.0.1:8000/analyze' \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "Приємна продавець. Жалюзі ідеально підійшли. Згодна що трохи кривий крипіж. Ковпачки з боків так і не натягнула. Там все складно. Але це не заважає функціонуванню. І все інше супер",
      ""
    ]
  }'
```

This sends a request to the FastAPI server at the /analyze endpoint.
The server will process the data, perform topic extraction, aspect-based sentiment analysis, and other operations as defined in the code.
The response will be displayed in the terminal, and results will also be saved in result_full.json and result_simplified.json in the working directory.

5. Running tests with test_docker.py

To test the API, run the test_docker.py script from the repository:
```terminal
python test_docker.py
```

## Running Locally (Without Docker)

1. Install dependencies from requirements.txt

```python
pip install -r requirements.txt
```

2. Start the API

```python
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Check if the API is running

```python
curl http://localhost:8000/docs
```

4. Example API usage from the command line

To send text for analysis, use curl:
```terminal
curl -X 'POST' \
  'http://127.0.0.1:8000/analyze' \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "Приємна продавець. Жалюзі ідеально підійшли. Згодна що трохи кривий крипіж. Ковпачки з боків так і не натягнула. Там все складно. Але це не заважає функціонуванню. І все інше супер",
      ""
    ]
  }'
```

5. Running tests with test_docker.py

To test the API, run the test_docker.py script from the repository:
```terminal
python test_docker.py
```


6. Example Usage

Input - API Request:
```terminal
curl -X 'POST' \
  'http://127.0.0.1:8000/analyze' \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": [
      "Приємна продавець. Жалюзі ідеально підійшли. Згодна що трохи кривий крипіж. Ковпачки з боків так і не натягнула. Там все складно. Але це не заважає функціонуванню. І все інше супер",
      ""
    ]
  }'
```

API Response (Example Output):







