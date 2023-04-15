# Product Review Classifier
- In this project I built a machine learning model that automatically classifies review as either being positive or negative<br>
- The training dataset come from Amazon reviews<br><br>

- To begin, 2 classes are defined to make the code more readable
```python
import random

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
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
        return [i.text for i in self.reviews]
        
    def get_sentiment(self):  # same method but in different class
        return [i.sentiment for i in self.reviews]
    
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == "Negative", self.reviews))
        positive = list(filter(lambda x: x.sentiment == "Positive", self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = positive_shrunk + negative
        random.shuffle(self.reviews)
```

- Import the dataset
```python
import json

file_path = '/Users/tongzhu/python_projects/ml/text/books_small_10000.json'

reviews = []
with open(file_path) as f:
    for i in f:
        line = json.loads(i)
        reviews.append(Review(line['reviewText'], line['overall']))
```

- Split the dataset using `train_test_split`
```python
from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)
```

- Keep the number of positive and negative reviews consistent
```python
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()
```

- Vectorize the reviews
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)
```

- We will be focusing on the Support Vector Machine(SVM) method
```python
from sklearn import svm

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)
```

- Mean accuracy
```python
print(clf_svm.score(test_x_vectors, test_y))
```
Running the above code will produce the following output:
```
0.8076923076923077
```

- F1 score
```python
from sklearn.metrics import f1_score

f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=['Positive', 'Negative'])
```
Running the above code will produce the following output:
```
array([0.80582524, 0.80952381])
```

- Try and improve the model fine tuning the parameters
```python
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'),
              'C': (0.5, 1, 2, 3, 4, 5, 10)}

untuned_clf_svm = svm.SVC()
tuned_clf_svm = GridSearchCV(untuned_clf_svm, parameters, cv=5)
tuned_clf_svm.fit(train_x_vectors, train_y)

print("Best parameters found: ", tuned_clf_svm.best_params_)
print("Best score achieved: ", tuned_clf_svm.best_score_)
```
Running the above code will produce the following output:
```
Best parameters found:  {'C': 3, 'kernel': 'rbf'}
Best score achieved:  0.8313957307060754
```

- Writing some random review samples to test the classifier
```python
random_test = ['Too bad DO NOT buy', 'not fun at all', 'so damn good']
random_test_vectors = vectorizer.transform(random_test)

clf_svm.predict(random_test_vectors)
```
Running the above code will produce the following output:
```
array(['Negative', 'Negative', 'Positive'], dtype='<U8')
```
