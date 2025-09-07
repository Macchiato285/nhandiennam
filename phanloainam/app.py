from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np

# =========================
# 1. Thuật toán ID3
# =========================
class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def info_gain(self, X, y, feature):
        total_entropy = self.entropy(y)
        values, counts = np.unique(X[feature], return_counts=True)
        weighted_entropy = np.sum([
            (counts[i] / np.sum(counts)) * self.entropy(y[X[feature] == values[i]])
            for i in range(len(values))
        ])
        return total_entropy - weighted_entropy

    def id3(self, X, y, features):
        if len(np.unique(y)) == 1:
            return y.iloc[0]
        if len(features) == 0:
            return y.mode()[0]

        gains = [self.info_gain(X, y, f) for f in features]
        best_feature = features[np.argmax(gains)]
        tree = {best_feature: {}}

        for value in np.unique(X[best_feature]):
            sub_X = X[X[best_feature] == value]
            sub_y = y[X[best_feature] == value]
            if sub_X.shape[0] == 0:
                tree[best_feature][value] = y.mode()[0]
            else:
                subtree = self.id3(sub_X, sub_y, [f for f in features if f != best_feature])
                tree[best_feature][value] = subtree

        return tree

    def fit(self, X, y):
        features = list(X.columns)
        self.tree = self.id3(X, y, features)

    def predict_one(self, x, tree=None):
        if tree is None:
            tree = self.tree
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        value = x.get(feature)
        if value in tree[feature]:
            return self.predict_one(x, tree[feature][value])
        else:
            return None

    def predict(self, X):
        return [self.predict_one(row) for _, row in X.iterrows()]

# =========================
# 2. Load dữ liệu nấm
# =========================
DATA_PATH = "mushrooms.csv"
data = pd.read_csv(DATA_PATH)

# Chỉ chọn 4 đặc trưng thay vì tất cả
selected_features = ["cap-shape", "cap-color", "odor", "stalk-color-above-ring"]

X = data[selected_features]
y = data["class"]

# train cây quyết định
model = DecisionTreeID3()
model.fit(X, y)

# =========================
# 3. Flask app
# =========================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", cols=X.columns)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {col: request.form[col] for col in X.columns}
    prediction = model.predict_one(input_data)
    result = "🍄 Nấm ĂN ĐƯỢC" if prediction == "e" else "☠️ Nấm ĐỘC"
    return render_template("result.html", result=result)


# API JSON
@app.route("/api/predict", methods=["POST"])
def api_predict():
    input_data = request.json
    prediction = model.predict_one(input_data)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
