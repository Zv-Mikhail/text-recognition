import easyocr  # Импорт библиотеки EasyOCR
import cv2      # Для чтения изображения (опционально, но полезно для предобработки)
import torch
import os
import pandas as pd 
import re
import pytesseract
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")

from typing import List, Tuple, Dict
# Шаг 1: Инициализация reader с поддержкой русского языка
# 'ru' - для русского, 'en' - для английского. Можно добавить несколько: ['ru', 'en']
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("Используем устройство:", device)

reader = easyocr.Reader(['ru', 'en']) # gpu=True, если есть GPU для ускорения



STOPWORDS = {
    "Инфо", "Встреча", "Персональная", "информация",
    "Работа", "Компания", "Категория", "Должность",
    "Интересы", "Я", "полезен", "Профили", "соцсетей",
    "О", "себе", "Чат",'Телефон','E-mail','Город', 'Я ищу',
    'Я полезен','себе','О себе', '0 себе''Профили соцсетей',
    'Telegram','Вконтакте'
}

def looks_like_name(texts: List[str]) -> Tuple[bool, str]:
    """
    Проверяет, похоже ли содержимое на ФИО (имя и фамилия), поддерживает 
    один или два текстовых блока
    """

    texts = [t.strip() for t in texts if t.strip()]
    # Удаляем слова-маркеры
    filtered = []
    for t in texts:
        words = t.split()
        words = [w for w in words if w not in STOPWORDS]
        if words:
            filtered.append(" ".join(words))

    # Если после фильтрации ничего не осталось
    if not filtered:
        return False, " "

    # одна строка
    if len(filtered) == 1:
        words = filtered[0].split()

        # ФИО в одной строке
        if len(words) == 2 and all(w[0].isupper() for w in words):
            return True, " ".join(words)

        return False, " "

    # две строки
    if len(filtered) == 2:
        w1 = filtered[0].split()
        w2 = filtered[1].split()

        if len(w1) == 1 and len(w2) == 1:
            if w1[0][0].isupper() and w2[0][0].isupper():
                return True, f"{w1[0]} {w2[0]}"

    return False, " "


def sub_strok(text_lines: List[str]) -> Tuple[str, int]:
    """
    Объединяет строки из списка `text_lines` до первого слова-маркера из STOPWORDS
    """
    label_exit = STOPWORDS
    n = -1
    stroka = ''
    for text in text_lines:
        n+=1 
        if text in label_exit:
            return stroka , n 
        elif text == text_lines[-1]:
            stroka += ' ' + text
            return stroka , n 
        else:
            stroka += ' '+ text
    print(f'Похоже найден новый лейбол: {text}')
    return '', n


def extract_text_from_bbox(image_path: str, bbox: List[List[float]]) -> str:
    """
    Вырезает область изображения по bbox и применяет pytesseract для распознавания текста
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Bbox от EasyOCR: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
    x2, y2 = int(bbox[2][0]), int(bbox[2][1])

    # Кроп с padding'ом 5 пикселей
    crop = img[max(0, y1-5):y2+5, max(0, x1-5):x2+5]

    # Предобработка: grayscale + threshold для лучшей точности
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Применяем pytesseract к кропу с config для URL
    refined = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./@')
    return refined.strip()


def extract_fields(image_path: str) -> Dict[str, str]:
    """
    Извлекает структурированные поля из OCR-результатов EasyOCR для одного изображения
    """
    results = reader.readtext(image_path)
    results_sorted = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    bbox, text_lines = zip(*[(bbox, text) for (bbox, text, _) in results_sorted])
    bbox = list(bbox)
    text_lines = list(text_lines)
    
    a = {}
    number = 0
    name_marker = True
    
    while number < len(text_lines):
        text = text_lines[number]

        # ФИО
        if name_marker:
            marker, name_text = looks_like_name(text_lines[number:number+2])
            if marker:
                a['ФИО'] = name_text
                name_marker = False
        
        # Поля, которые могут занимать несколько строк
        elif text in {'Телефон', 'E-mail', 'Город', 'Компания', 'Категория',
                      'Должность', 'Я ищу', 'Я полезен', 'О себе', 'Профили соцсетей'}:
            field, counts = sub_strok(text_lines[number+1:])
            if text == 'E-mail':
                field = '.'.join(field.split())
            a[text] = field
            number += counts
        
        # Соцсети с отдельной обработкой через bbox
        elif text in {'Telegram', 'Вконтакте'}:
            if number + 1 < len(bbox):
                clean_text = extract_text_from_bbox(image_path, bbox[number+1])
                clean_text = clean_text.replace('@', '').strip()
                a[text] = clean_text
                number += 1

        number += 1
    
    return a
        
def merge_rows_keep_all_files(group: pd.DataFrame) -> pd.Series:
    """
    Объединяет несколько строк одного человека в одну, сохраняя все файлы и выбирая
    наиболее информативные значения для остальных колонок
    """
    result = {}

    for col in group.columns:
        if col == "Name file":
            # Берём только имя файла без пути и расширения
            files = group[col].dropna().apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            # Оставляем уникальные, сохраняем порядок
            seen = set()
            files_unique = [f for f in files if not (f in seen or seen.add(f))]
            result[col] = ','.join(files_unique)
        elif col == "ФИО":
            result[col] = group[col].iloc[0]
        else:
            # Выбираем самое длинное значение
            values = group[col].dropna().astype(str)
            result[col] = max(values, key=len) if len(values) > 0 else np.nan

    return pd.Series(result)

folder_path = 'data'

# Список всех файлов 
image_paths = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
image_paths.sort()
# Датафрейм, который будет заполнен
df = pd.DataFrame(columns=['ФИО','Телефон','E-mail','Город',
                            'Компания', 'Категория', 'Должность',
                            'Я ищу','Я полезен', 'О себе',
                            'Профили соцсетей','Telegram','Вконтакте',
                            'Name file'])


for image_path in image_paths:
    print(f'{image_path} + успешно обработан')
    row = extract_fields(image_path)
    row['Name file'] = image_path.replace('data/','')
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Обрезаем ведущие и trailing пробелы во всех строковых колонках
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

df.to_csv('data.csv', index=False, encoding='utf-8')

# Группировка по ФИО без сортировки, чтобы сохранить порядок появления
df_merged = df.groupby("ФИО", sort=False).apply(merge_rows_keep_all_files).reset_index(drop=True)

# Сохранение в CSV
df_merged.to_csv('results.csv', index=False, encoding='utf-8')

# Сохранение в XLSX 
df_merged.to_excel('results.xlsx', index=False)

