
# TODO: NOT WORK Error
"""
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like distilbert/distilbert-base-uncased-finetuned-sst-2-english is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
"""
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
results = classifier(["We are very happy to show you the   Transformers library.",
           "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")