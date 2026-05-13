# That what will be the student's result
#Traditional software

marks = int(input("Enter marks: "))

if marks >= 40:
    print("PASS")
else:
    print("FAIL")

# ML based Prediction
from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data
# Hours studied
X = np.array([[1], [2], [3], [4], [5], [6]])

# Pass(1) / Fail(0)
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict
hours = [[4.5]]

prediction = model.predict(hours)

if prediction[0] == 0:
    print("Student may PASS")
else:
    print("Student may FAIL")
