# Text Classifier
- In this project I built a machine learning model that automatically classifies text as either being positive or negative 
- The training dataset come from Amazon reviews
```Python
import json

file_path = '/Users/tongzhu/python_projects/ml/text/books_small_10000.json'

reviews = []
with open(file_path) as f:
    for i in f:
        line = json.loads(i)
#       print(line['reviewText'])
#       print(line['overall'])
        reviews.append(Review(line['reviewText'], line['overall']))  # Review(text, score)
```

