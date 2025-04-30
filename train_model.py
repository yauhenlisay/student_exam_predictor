import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticModel:
    def __init__(self, learning_rate=0.05, max_iter=10000, tolerance=0.0001):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.w = None
        self.b = None
        self.scaler = StandardScaler()
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape
        self.w = np.zeros(n_features)
        self.b = 0
        prev_loss = np.inf
        
        for i in range(self.max_iter):
            z = np.dot(X_scaled, self.w) + self.b
            y_pred = self.sigmoid(z)
            
            dw = (1 / n_samples) * np.dot(X_scaled.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            cur_loss = self.binary_cross_entropy(y, y_pred)
            
            if abs(prev_loss - cur_loss) < self.tolerance:
                print(f"Early stopping at epoch {i}, loss change < tolerance")
                break
                
            prev_loss = cur_loss
            
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {cur_loss:.6f}")
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        z = np.dot(X_scaled, self.w) + self.b
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def load_and_preprocess_data(filepath):
    """Загрузка и предобработка данных"""
    df = pd.read_csv(filepath)
    
    # Преобразование целевой переменной
    df['Сдал'] = df['Сдал'].map({'Нет': 0, 'Да': 1})
    
    # Преобразование категориальных признаков
    df['Сон накануне'] = df['Сон накануне'].map({'Нет': 0, 'Да': 1})
    df['Пил энергетики'] = df['Энергетиков накануне'].map(lambda x: 0 if x == '0' else 1)
    df['Cредняя оценка'] = ((df['Контрольная 1'] + df['Контрольная 2'] + df['Контрольная 3']) / 3).astype(int)
    
    # Дополнительные фичи
    df['Подготовка'] = df['Время подготовки'].map({
        'Последний час': 0,
        'Последняя ночь': 1,
        'За несколько дней': 3,
        'За неделю': 7
    })
    
    marks = df[['Контрольная 1','Контрольная 2','Контрольная 3']]
    df['Вариация оценки'] = marks.max(axis=1) - marks.min(axis=1)
    df['Максимальная оценка'] = marks.max(axis=1)
    df['Минимальная оценка'] = marks.min(axis=1)
    df['Успешность'] = df['Cредняя оценка'].max() -  df['Cредняя оценка']
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Настроение','Энергетиков накануне',
                                    'Посещаемость занятий','Время подготовки'],
                       drop_first=True, dtype=int)
    
    X = df.drop(columns=['Сдал'])
    y = df['Сдал']
    
    return X, y

def main():
    # Загрузка данных
    X, y = load_and_preprocess_data('data/student_exam_data.csv')
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Обучение модели
    model = LogisticModel(learning_rate=0.05, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy на тестовом наборе: {accuracy:.2%}")
    
    # Сохранение модели
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Модель успешно обучена и сохранена в models/model.pkl")

if __name__ == "__main__":
    main()