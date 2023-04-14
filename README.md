# Text Classifier
In this project I built a machine learning model that automatically classifies text as either being positive or negative 
The training dataset come from Amazon reviews

```python
import random

class Review:
    def __init__(self, text, score):
        self.text = text  # allowed to do review[index].text
        self.score = score  # allowed to do review[index].score
        self.sentiment = self.get_sentiment()
    
    def get_sentiment(self):
        if self.score <= 2:
            return 'Negative'
        elif self.score == 3:
            return 'Neutral'
        else:
            return 'Positive'
        
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
    def get_text(self):
        return [i.text for i in self.reviews]  # method for list comprehension (text)
        
    def get_sentiment(self):  # same method but in different class
        return [i.sentiment for i in self.reviews]  # method for list comprehension (sentiment)
    
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == "Negative", self.reviews))
        positive = list(filter(lambda x: x.sentiment == "Positive", self.reviews))
        positive_shrunk = positive[:len(negative)]  # shrunk the number of positive reviews
        self.reviews = positive_shrunk + negative
        random.shuffle(self.reviews)  # shuffle the order of positive and negative reviews
```

```python
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

