# Get file paths of cropped images
celebrity_file_names_dict = {}
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list

# Create class dictionary for classification
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1

# Prepare data for training
X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is not None:
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
            X.append(combined_img)
            y.append(class_dict[celebrity_name])

# Convert data to numpy array and reshape
X = np.array(X).reshape(len(X), 4096).astype(float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create and train a basic SVM model
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(X_train, y_train)
basic_score = pipe.score(X_test, y_test)
print(f"Basic model score: {basic_score}")
print(classification_report(y_test, pipe.predict(X_test)))

# Define model and hyperparameters for grid search
model_params = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear'] 
        }  
    }
}

# Perform grid search to find best parameters
scores = []
best_estimators = {}

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
    best_estimators[algo] = clf.best_estimator_

# Convert results to DataFrame
df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

# Use the best classifier
best_clf = best_estimators['svm']

# Create confusion matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('confusion_matrix.png')

# Save the model
joblib.dump(best_clf, 'saved_model.pkl')

# Save the class dictionary
with open("class_dictionary.json", "w") as f:
    f.write(json.dumps(class_dict))

print("Model training complete. Model saved as 'saved_model.pkl'")
print("Class dictionary saved as 'class_dictionary.json'")