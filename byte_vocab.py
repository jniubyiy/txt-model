# byte_vocab.py
# Словарь для байтов (0-255)

BYTE_VOCAB_SIZE = 256

# Отображение байт -> индекс (identity)
BYTE_TO_IDX = {b: b for b in range(BYTE_VOCAB_SIZE)}

# Отображение индекс -> байт
IDX_TO_BYTE = {i: i for i in range(BYTE_VOCAB_SIZE)}

print(f"📚 Загружен байтовый словарь: {BYTE_VOCAB_SIZE} токенов")