from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import spacy
import re
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import json
from transformers import pipeline

from pyabsa import AspectTermExtraction as ATEPC

# Load Aspect-Based Sentiment Model
aspect_extractor = ATEPC.AspectExtractor('multilingual', auto_device=False, cal_perplexity=True)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load Hugging Face sentiment analysis model
sentiment_pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

# Load ML models
topic_model = BERTopic.load("models/2tips_sentence_BERTopic", embedding_model=embedding_model)

# Group topics by categories
grouped_topics = {
    "Ціна": [0],
    "Рекомендації": [13, 27],
    "Досвід": [2, 12, 28],
    "Доставка/Обслугогування": [8, 11],
    "Якість": [6, 7, 25, 38],
}

# Mapping function for sentiment categories
def map_sentiment(label):
    if label in ["Very Negative", "Negative"]:
        return "Negative"
    elif label == "Neutral":
        return "Neutral"
    elif label in ["Positive", "Very Positive"]:
        return "Positive"

# Load spaCy model for sentence segmentation
nlp = spacy.load("uk_core_news_md")

# Function to map detected topics to grouped categories
def map_topics_to_groups(predicted_topics):
    matched_groups = set()
    for group, topic_ids in grouped_topics.items():
        if any(topic in topic_ids for topic in predicted_topics):
            matched_groups.add(group)
    return list(matched_groups)

# Request model
class TextRequest(BaseModel):
    texts: List[str]

# Custom text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r"[!?]+", ".", text)
    text = text.replace("\n", ". ")
    return text

# Function to split reviews into sentences
def split_into_sentences(texts):
    sentences = []
    sentence_to_review = {}

    for idx, text in enumerate(texts):
        processed_text = preprocess_text(text)
        doc = nlp(processed_text)

        for sent in doc.sents:
            sentence = sent.text.strip()
            if sentence:
                sentences.append(sentence)
                sentence_to_review[sentence] = idx  # Store which review this sentence belongs to
    
    return sentences, sentence_to_review

# Function to identify if a sentence is a question
def is_question(sentence):
    sentence_stripped = sentence.strip()
    
    if sentence_stripped.endswith('?'):
        return True
    
    doc = nlp(sentence_stripped)
    for token in doc:
        if token.is_alpha:
            first_word = token.text.lower()
            if first_word in {"що", "чому", "де", "коли", "скільки", "звідки", "який", "яка", "яке", "які", "чий", "чия", "чие", "чиї"}:
                return True
            break
    return False

def find_questions(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if is_question(sent.text)]

# FastAPI app initialization
app = FastAPI()

@app.post("/analyze")
def analyze_texts(request: TextRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")
    
    non_empty_texts = [text for text in request.texts if text.strip()]
    empty_texts = [text for text in request.texts if not text.strip()]

    notification = None
    if empty_texts:
        notification = f"{len(empty_texts)} empty text(s) were skipped."

    review_topic_groups = {i: [] for i in range(len(non_empty_texts))}
    review_aspects = {i: [] for i in range(len(non_empty_texts))}  # Store aspects per review
    review_questions = {i: [] for i in range(len(non_empty_texts))}  # Store questions per review

    for idx, text in enumerate(non_empty_texts):
        sentences, sentence_to_review = split_into_sentences([text])
        
        if not sentences:
            return {"error": "No valid sentences extracted."}

        # Extract questions from the review text
        review_questions[idx] = find_questions(text)

        # Predict topics for sentences
        predicted_topics = []
        for sentence in sentences:
            topic = topic_model.transform([sentence])[0][0]
            if topic != -1:
                predicted_topics.append(topic)

        unique_topics = np.unique(predicted_topics).tolist() if predicted_topics else []

        # Map topics to categories
        topic_groups = map_topics_to_groups(unique_topics)

        # Assign only topic groups (no raw topics)
        review_topic_groups[idx] = topic_groups

        # **Aspect-Based Sentiment Analysis (ABSA)**
        aspect_results = aspect_extractor.predict([preprocess_text(sentence) for sentence in sentences],
                                                  save_result=False,
                                                  print_result=False,
                                                  ignore_error=True)

        # Set to track unique sentences
        processed_sentences = set()

        # Process extracted aspects
        for aspect_data in aspect_results:
            for aspect, sentiment in zip(aspect_data["aspect"], aspect_data["sentiment"]):
                # Find the sentence containing the aspect
                aspect_sentence = next((s for s in sentences if aspect in s), None)

                if aspect_sentence and aspect_sentence not in processed_sentences:
                    # Mark this sentence as processed
                    processed_sentences.add(aspect_sentence)

                    # Append unique aspect-sentence pairs
                    review_aspects[idx].append({
                        "aspect": aspect,
                        "sentence": aspect_sentence,
                        "sentiment": sentiment,
                    })

    # **Replace NB model with Transformer-based model**
    sentiment_results = sentiment_pipe(non_empty_texts)

    # Extract sentiment labels
    sentiment_predictions = [map_sentiment(res["label"]) for res in sentiment_results]

    # Prepare the original response
    results = []
    for i, text in enumerate(non_empty_texts):
        results.append({
            "text": text,
            "sentiment": sentiment_predictions[i],  # Overall sentiment
            "groups": review_topic_groups[i],  # Only grouped topics
            "complaints": review_aspects[i],  # Aspect-based sentiment analysis with sentences
            "questions": review_questions[i]  # List of questions found in the review
        })

    response = {"results": results}
    if notification:
        response["notification"] = notification

    # Save original response as JSON (optional)
    with open("result_full.json", "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=4)

    # Prepare simplified response
    simplified_results = []
    for i, text in enumerate(non_empty_texts):
        simplified_results.append({
            "review": text,
            "sentiment": sentiment_predictions[i],
            "topics": review_topic_groups[i],
            "complaints": [aspect["sentence"] for aspect in review_aspects[i]],
            "questions": review_questions[i]
        })

    simplified_response = {"results": simplified_results}
    if notification:
        simplified_response["notification"] = notification

    # Save simplified response as JSON
    with open("result_simplified.json", "w", encoding="utf-8") as f:
        json.dump(simplified_response, f, ensure_ascii=False, indent=4)

    return response





# Run with:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
