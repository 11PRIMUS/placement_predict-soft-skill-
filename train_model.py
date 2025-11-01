import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from scipy import stats
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib



def cefr_num(cefr):
    if pd.isna(cefr) or cefr==0:
        return 3
    cefr = str(cefr).upper()
    cefr_map={
        'A1':1,'A2':2,'B1':3,'B2':4,'C1':5,'C2':6
    }
    return cefr_map.get(cefr, 3)

def outlier_remove(df, columns, n_std=3):
    for col in columns:
        z_score =stats.zscore(df[col])
        abs_z_scores = np.abs(z_score)
        filtered_entries = (abs_z_scores < n_std)
        df = df[filtered_entries]
    return df

def feature_add(X):
    X=X.copy()
    X['HR_GD_interaction'] = X['Mock HR']*X['GD']
    X['English_Total'] = X['English CEFR']*X['English Score']
    X['Overall_Performance'] =X[['Mock HR', 'GD', 'Presentation']].mean(axis=1)
    return X

#data loader
print("Loading and preprocessing data...")
df1 =pd.read_excel('data/Soft Skills Scores 2021-25.xlsx', skiprows=1)
df1.columns = ['Branch', 'Mock HR', 'GD', 'Presentation', 'English CEFR', 'English Score', 'Total', 'Placement']
df1['English CEFR'] = df1['English CEFR'].apply(cefr_num)

df2 =pd.read_excel('data/Soft Skills Scores 2022-26 n.xlsx')
df2_processed = pd.DataFrame({
    'Mock HR': df2['mock_best_score'],
    'GD': df2['gd_best_score'],
    'Presentation': df2['Presentation_Score'],
    'English CEFR': df2['English_Band'].apply(cefr_num),
    'English Score': df2['English_Score'],
    'Placement': df2['placed']
})

#combine datasets
df1_features = df1[['Mock HR', 'GD', 'Presentation', 'English CEFR', 'English Score', 'Placement']]
combined_df = pd.concat([df1_features, df2_processed], ignore_index=True)

#clean data
for col in ['Mock HR', 'GD', 'Presentation', 'English Score']:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    median_val = combined_df[col].median()
    combined_df[col] = combined_df[col].fillna(median_val)
print("\nOutliers removed")
combined_df = outlier_remove(combined_df, ['Mock HR', 'GD', 'Presentation', 'English Score'])

#features and target
X = combined_df[['Mock HR', 'GD', 'Presentation', 'English CEFR', 'English Score']]
X = feature_add(X)
y = combined_df['Placement']

print("\nOriginal class distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#base classfifers
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)

voting_classifier = VotingClassifier(estimators=[('rf',rf), ('gb',gb),('svc',svc)], voting='soft')

pipeline = Pipeline([
    ('sampling', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', voting_classifier)
])
param_grid = {
    'classifier__rf__n_estimators': [100, 200],
    'classifier__rf__max_depth': [10, 20],
    'classifier__rf__min_samples_split': [2, 5],
    'classifier__gb__n_estimators': [100, 200],
    'classifier__gb__max_depth': [3, 5],
    'classifier__gb__learning_rate': [0.01, 0.1],
    'classifier__svc__C': [0.1, 1.0, 10.0],
    'classifier__svc__gamma': ['scale', 'auto']
}

#grid 
print("\nOptimizing hyperparameters ")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.3f}")

best_model=grid_search.best_estimator_

y_train_pred =best_model.predict(X_train)
y_test_pred =best_model.predict(X_test)

#roc-auc score
train_roc_auc =roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
test_roc_auc =roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print("\nTraining Set Metrics:")
print(classification_report(y_train, y_train_pred))
print(f"ROC AUC Score (Train): {train_roc_auc:.3f}")

print("\nTest Set Metrics:")
print(classification_report(y_test, y_test_pred))
print(f"ROC AUC Score (Test): {test_roc_auc:.3f}")

feature_importance=pd.DataFrame({
    'feature':X.columns,
    'importance':best_model.named_steps['classifier'].named_estimators_['rf'].feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='importance', ascending=False))

joblib.dump(best_model, 'p3.pkl')  
