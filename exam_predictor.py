import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

class LogisticModel:
    def __init__(self):
        self.w = None
        self.b = None
        self.scaler = StandardScaler()
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        if self.w is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        z = np.dot(X_scaled, self.w) + self.b
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def validate_input(data):
    """Проверка валидности входных данных"""
    errors = []
    
    # Проверка оценок
    for test in ['Контрольная 1', 'Контрольная 2', 'Контрольная 3']:
        if not (1 <= data[test] <= 10):
            errors.append(f"{test} должна быть между 1 и 10")
    
    # Проверка категориальных признаков
    valid_values = {
        'Сон накануне': ['Да', 'Нет'],
        'Настроение': ['Плохое', 'Нормальное', 'Хорошее'],
        'Энергетиков накануне': ['0', '1', '2-3', '4+'],
        'Посещаемость занятий': ['Низкая', 'Средняя', 'Высокая'],
        'Время подготовки': ['Последний час', 'Последняя ночь', 'За несколько дней', 'За неделю']
    }
    
    for field, allowed in valid_values.items():
        if data[field] not in allowed:
            errors.append(f"Недопустимое значение для {field}. Допустимые: {', '.join(allowed)}")
    
    if errors:
        raise ValueError("\n".join(errors))

def preprocess_input(data):
    df = pd.DataFrame([data] if isinstance(data, dict) else data)
    
    # Те же преобразования, что и при обучении
    df['Сон накануне'] = df['Сон накануне'].map({'Нет': 0, 'Да': 1})
    df['Пил энергетики'] = df['Энергетиков накануне'].map(lambda x: 0 if x == '0' else 1)
    df['Cредняя оценка'] = ((df['Контрольная 1'] + df['Контрольная 2'] + df['Контрольная 3']) / 3).astype(int)
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
    
    # Фиксированные категории , должны совпадать с обучением!
    categories = {
        'Настроение': ['Нормальное', 'Плохое', 'Хорошее'],
        'Энергетиков накануне': ['0', '1', '2-3', '4+'],
        'Посещаемость занятий': ['Средняя', 'Низкая', 'Высокая'],
        'Время подготовки': ['Последний час', 'Последняя ночь', 'За несколько дней', 'За неделю']
    }
    
    for col, options in categories.items():
        for option in options:
            if option != options[0]:
                df[f"{col}_{option}"] = (df[col] == option).astype(int)
    
    # Все фичи, которые ожидает модель
    expected_features = ['Контрольная 1', 'Контрольная 2', 'Контрольная 3', 'Сон накануне',
       'Пил энергетики', 'Cредняя оценка', 'Подготовка', 'Вариация оценки','Максимальная оценка','Минимальная оценка',
       'Успешность', 'Настроение_Плохое', 'Настроение_Хорошее', 
       'Энергетиков накануне_1', 'Энергетиков накануне_2-3', 'Энергетиков накануне_4+',
       'Посещаемость занятий_Низкая', 'Посещаемость занятий_Средняя',
       'Время подготовки_За несколько дней', 'Время подготовки_Последний час',
       'Время подготовки_Последняя ночь']
    
    # Добавляем недостающие фичи
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[expected_features]

def load_model(model_path='models/model.pkl'):
    """Загрузка обученной модели"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_from_csv(input_file):
    """Предсказание для данных из CSV файла"""
    try:
        input_data = pd.read_csv(input_file)
        # Проверяем наличие всех необходимых колонок
        required_columns = ['Контрольная 1', 'Контрольная 2', 'Контрольная 3', 
                           'Сон накануне', 'Настроение', 'Энергетиков накануне',
                           'Посещаемость занятий', 'Время подготовки']
        
        missing = set(required_columns) - set(input_data.columns)
        if missing:
            raise ValueError(f"В файле отсутствуют обязательные колонки: {missing}")
        
        # Проверяем каждую строку
        for _, row in input_data.iterrows():
            try:
                validate_input(row.to_dict())
            except ValueError as e:
                raise ValueError(f"Ошибка в строке {_+1}: {str(e)}")
        
        # Если все проверки пройдены
        processed_data = preprocess_input(input_data)
        model = load_model()
        
        probabilities = model.predict_proba(processed_data)
        predictions = model.predict(processed_data)
        
        results = pd.DataFrame({
            'Вероятность сдачи': probabilities,
            'Прогноз': ['Да' if p == 1 else 'Нет' for p in predictions]
        })  
        
        print("\nРезультаты прогнозирования:")
        print(results.to_string(index=False))
        
    except Exception as e:
        print(f"\nОшибка при обработке файла: {str(e)}")

def predict_single(student_data):
    """Предсказание для одного студента"""
    try:
        validate_input(student_data)
        df = pd.DataFrame([student_data])
        processed_data = preprocess_input(df)
        model = load_model()
        
        probability = model.predict_proba(processed_data)[0]
        prediction = 'Да' if model.predict(processed_data)[0] == 1 else 'Нет'
        
        print(f"\nРезультат прогнозирования:")
        print(f"Вероятность сдачи: {probability:.2%}")
        print(f"Прогноз: {prediction}")
        
    except ValueError as e:
        print(f"\nОшибка в данных: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Прогнозирование сдачи экзамена')
    parser.add_argument('--input-data', type=str, help='Путь к CSV файлу с данными студентов')
    
    args = parser.parse_args()
    
    if args.input_data:
        predict_from_csv(args.input_data)
    else:
        print("Введите данные студента:")
        try:
            student_data = {
                'Контрольная 1': int(input("Оценка за контрольную 1 (1-10): ")),
                'Контрольная 2': int(input("Оценка за контрольную 2 (1-10): ")),
                'Контрольная 3': int(input("Оценка за контрольную 3 (1-10): ")),
                'Сон накануне': input("Спал накануне (Да/Нет): ").capitalize(),
                'Настроение': input("Настроение (Плохое/Нормальное/Хорошее): ").capitalize(),
                'Энергетиков накануне': input("Энергетиков накануне (0/1/2-3/4+): "),
                'Посещаемость занятий': input("Посещаемость (Низкая/Средняя/Высокая): ").capitalize(),
                'Время подготовки': input("Время подготовки (Последний час/Последняя ночь/За несколько дней/За неделю): ").capitalize()
            }
            
            predict_single(student_data)
        except ValueError as e:
            print(f"\nОшибка ввода: {str(e)}")
        except Exception as e:
            print(f"\nНепредвиденная ошибка: {str(e)}")

if __name__ == "__main__":
    main()