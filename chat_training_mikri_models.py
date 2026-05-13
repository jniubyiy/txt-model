# chat_training_mikri_models.py
"""
Веб-интерфейс чата на Flask с интеграцией обученных моделей.
Добавлены кнопки для запуска Stage 1, Stage 2 и полного обучения.
Модели выгружаются из памяти после каждого ответа.
"""

import os
import json
import zmq
from datetime import datetime
from flask import Flask, request, jsonify

# Импорт модуля инференса
from chat_usage_mikri_models import ChatModel

app = Flask(__name__)

# Папка для хранения истории чата
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ZeroMQ контекст (для отправки команды обучения)
context = zmq.Context()
push_socket = context.socket(zmq.PUSH)
push_socket.connect("tcp://127.0.0.1:5555")

# Инициализация модели чата (загружается при старте сервера)
print("Loading chat model...")
chat_model = ChatModel()
print("Chat model loaded.")

# ----------------------------------------------------------------------
# Вспомогательные функции для работы с историей (без изменений)
# ----------------------------------------------------------------------
def get_next_message_index():
    """Возвращает следующий свободный порядковый номер сообщения."""
    files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
    if not files:
        return 1
    indices = []
    for f in files:
        parts = f.split('_')
        if len(parts) >= 2:
            try:
                indices.append(int(parts[1]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else 1

def save_message_to_file(role, text, checked=False):
    """Сохраняет новое сообщение в JSON-файл."""
    index = get_next_message_index()
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{role}_{index}_{timestamp_str}.json"
    filepath = os.path.join(HISTORY_DIR, filename)
    data = {
        "role": role,
        "content": text,
        "edited_content": None,
        "checked": checked,
        "index": index,
        "timestamp": timestamp_str
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return index

def update_message_in_file(index, new_text=None, checked=None):
    """Обновляет текст сообщения и/или состояние checked в файле."""
    for filename in os.listdir(HISTORY_DIR):
        if not filename.endswith('.json'):
            continue
        parts = filename.split('_')
        if len(parts) >= 2 and parts[1] == str(index):
            filepath = os.path.join(HISTORY_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if new_text is not None:
                data['edited_content'] = new_text
            if checked is not None:
                data['checked'] = checked
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
    return False

def load_all_messages():
    """Загружает все сообщения из папки истории, сортирует по индексу."""
    messages = []
    if not os.path.exists(HISTORY_DIR):
        return messages
    for filename in os.listdir(HISTORY_DIR):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(HISTORY_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                msg = json.load(f)
                display_text = msg.get('edited_content') or msg.get('content')
                messages.append({
                    'role': msg['role'],
                    'content': display_text,
                    'original_content': msg.get('content'),
                    'edited_content': msg.get('edited_content'),
                    'checked': msg.get('checked', False),
                    'index': msg.get('index'),
                    'timestamp': msg.get('timestamp')
                })
        except Exception:
            continue
    messages.sort(key=lambda x: x.get('index', 0))
    return messages

# ----------------------------------------------------------------------
# HTML-шаблон с тремя кнопками обучения и кнопкой "Получить ответ"
# ----------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mikri Chat</title>
    <style>
        :root {
            --bg-primary: #1e1e1e;
            --bg-secondary: #2d2d2d;
            --border-color: #444444;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --accent-blue: #2196F3;
            --accent-red: #F44336;
            --input-bg: #333333;
        }
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
            background: var(--bg-primary);
            padding: 12px 20px;
            border-bottom: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            gap: 20px;
            height: 50px;
        }
        .fixed-header h2 {
            margin: 0;
            font-weight: 400;
            color: var(--text-primary);
        }
        .fixed-header .subtitle {
            color: var(--text-secondary);
            font-size: 0.9em;
        }
        .header-spacer {
            flex: 1;
        }
        .train-btn {
            padding: 8px 16px;
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 0;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            white-space: nowrap;
        }
        .train-btn:hover {
            background: #0b7ad1;
        }
        .train-btn:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .status-indicator {
            margin-left: 10px;
            font-size: 0.9rem;
            color: #aaa;
        }
        .fixed-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 100;
            background: var(--bg-primary);
            padding: 16px 20px;
            border-top: 1px solid var(--border-color);
            box-shadow: 0 -2px 4px rgba(0,0,0,0.3);
            display: flex;
            gap: 12px;
            align-items: flex-end;
            transition: height 0.1s ease;
        }
        .input-wrapper {
            flex: 1;
            display: flex;
        }
        .fixed-footer textarea {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 0;
            font-size: 1rem;
            outline: none;
            background: var(--input-bg);
            color: var(--text-primary);
            resize: none;
            overflow-y: auto;
            line-height: 1.5;
            font-family: inherit;
            box-sizing: border-box;
            min-height: 46px;
            max-height: calc(1.5em * 10 + 24px);
        }
        .fixed-footer textarea::placeholder {
            color: #888;
        }
        .fixed-footer button {
            padding: 12px 24px;
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 0;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            white-space: nowrap;
            height: fit-content;
        }
        .fixed-footer button:hover {
            background: #0b7ad1;
        }
        .chat-container {
            position: fixed;
            top: 70px;
            left: 0;
            right: 0;
            bottom: 90px;
            overflow-y: auto;
            padding: 0 20px;
        }
        .message-list {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
            padding: 16px 0;
        }
        .message-row {
            display: flex;
            align-items: flex-start;
            width: 100%;
            box-sizing: border-box;
        }
        .message-content-wrapper {
            display: flex;
            align-items: flex-start;
            width: 100%;
        }
        .edit-box, .toggle-box {
            flex-shrink: 0;
        }
        .edit-box {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 8px;
            background: transparent;
        }
        .edit-btn {
            background: none;
            border: none;
            font-size: 1.4rem;
            line-height: 1;
            cursor: pointer;
            padding: 0 4px;
            color: #aaa;
            opacity: 0.7;
            transition: opacity 0.2s;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .edit-btn:hover {
            opacity: 1;
            color: #fff;
        }
        .message {
            padding: 12px 16px;
            border-radius: 0;
            line-height: 1.5;
            color: white;
            word-wrap: break-word;
            width: 100%;
            box-sizing: border-box;
            white-space: pre-wrap;
        }
        .user .message {
            background: var(--accent-blue);
            border-bottom: 3px solid var(--accent-blue);
        }
        .bot .message {
            background: var(--accent-red);
            border-bottom: 3px solid var(--accent-red);
        }
        .toggle-box {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: 8px;
            background: transparent;
        }
        .toggle-btn {
            background: none;
            border: none;
            font-size: 1.8rem;
            line-height: 1;
            cursor: pointer;
            padding: 0 4px;
            color: white;
            opacity: 0.8;
            transition: opacity 0.2s;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .toggle-btn:hover {
            opacity: 1;
        }
        .toggle-btn.check {
            color: #4CAF50;
        }
        .toggle-btn.cross {
            color: #f44336;
        }
        .message.editing {
            padding: 0;
            background: transparent;
            border-bottom: none;
        }
        .edit-textarea {
            width: 100%;
            min-width: 300px;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 0;
            font-size: 1rem;
            background: var(--input-bg);
            color: var(--text-primary);
            resize: vertical;
            line-height: 1.5;
            font-family: inherit;
            box-sizing: border-box;
        }
        .edit-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        .edit-actions button {
            padding: 6px 12px;
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 0;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .edit-actions button.cancel {
            background: #666;
        }
    </style>
</head>
<body>
    <div class="fixed-header">
        <h2>🧠 Mikri Chat</h2>
        <span class="subtitle">AI Assistant v1.0</span>
        <div class="header-spacer"></div>
        <button class="train-btn" id="trainFullButton">Обучить всё</button>
        <button class="train-btn" id="trainStage1Button">Stage 1</button>
        <button class="train-btn" id="trainStage2Button">Stage 2</button>
        <span class="status-indicator" id="trainingStatus"></span>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="message-list" id="messageList"></div>
    </div>

    <div class="fixed-footer" id="footer">
        <div class="input-wrapper">
            <textarea id="messageInput" placeholder="Введите сообщение..." rows="1"></textarea>
        </div>
        <button id="sendButton">Отправить</button>
        <button id="getResponseButton">Получить ответ</button>
    </div>

    <script>
        const messageList = document.getElementById('messageList');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const getResponseButton = document.getElementById('getResponseButton');
        const trainFullButton = document.getElementById('trainFullButton');
        const trainStage1Button = document.getElementById('trainStage1Button');
        const trainStage2Button = document.getElementById('trainStage2Button');
        const trainingStatus = document.getElementById('trainingStatus');
        const chatContainer = document.getElementById('chatContainer');
        const footer = document.getElementById('footer');

        let isLoadingHistory = false;
        let currentEditRow = null;

        async function sendTrainingCommand(endpoint) {
            trainFullButton.disabled = true;
            trainStage1Button.disabled = true;
            trainStage2Button.disabled = true;
            trainingStatus.textContent = 'Запуск обучения...';
            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const data = await response.json();
                if (data.status === 'ok') {
                    trainingStatus.textContent = 'Обучение запущено.';
                } else {
                    trainingStatus.textContent = 'Ошибка: ' + (data.error || 'неизвестно');
                }
            } catch (error) {
                console.error('Ошибка:', error);
                trainingStatus.textContent = 'Ошибка соединения';
            } finally {
                trainFullButton.disabled = false;
                trainStage1Button.disabled = false;
                trainStage2Button.disabled = false;
            }
        }

        trainFullButton.addEventListener('click', () => sendTrainingCommand('/start_training'));
        trainStage1Button.addEventListener('click', () => sendTrainingCommand('/start_training_stage1'));
        trainStage2Button.addEventListener('click', () => sendTrainingCommand('/start_training_stage2'));

        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 
                parseInt(getComputedStyle(textarea).maxHeight)) + 'px';
            updateChatBottom();
        }

        function updateChatBottom() {
            const footerHeight = footer.offsetHeight;
            chatContainer.style.bottom = footerHeight + 'px';
        }

        const resizeObserver = new ResizeObserver(() => {
            updateChatBottom();
        });
        resizeObserver.observe(footer);

        messageInput.addEventListener('input', function() {
            autoResize(this);
        });

        messageInput.addEventListener('paste', function() {
            setTimeout(() => autoResize(this), 0);
        });

        function createToggleButton(initialChecked = false) {
            const btn = document.createElement('button');
            btn.className = initialChecked ? 'toggle-btn check' : 'toggle-btn cross';
            btn.textContent = initialChecked ? '✔' : '✘';
            btn.setAttribute('data-state', initialChecked ? 'check' : 'cross');
            
            btn.addEventListener('click', async function(e) {
                e.stopPropagation();
                const row = this.closest('.message-row');
                const index = row ? row.dataset.index : null;
                const newChecked = this.getAttribute('data-state') !== 'check';
                
                if (newChecked) {
                    this.textContent = '✔';
                    this.className = 'toggle-btn check';
                    this.setAttribute('data-state', 'check');
                } else {
                    this.textContent = '✘';
                    this.className = 'toggle-btn cross';
                    this.setAttribute('data-state', 'cross');
                }
                
                if (index) {
                    try {
                        await fetch('/edit_message', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ index: parseInt(index), checked: newChecked })
                        });
                    } catch (error) {
                        console.error('Ошибка сохранения состояния кнопки:', error);
                    }
                }
            });
            
            return btn;
        }

        async function saveMessageToServer(role, text) {
            try {
                const response = await fetch('/save_message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ role, text })
                });
                const data = await response.json();
                return data.index;
            } catch (error) {
                console.error('Ошибка сохранения сообщения:', error);
                return null;
            }
        }

        async function updateMessageOnServer(index, newText) {
            try {
                await fetch('/edit_message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index, new_text: newText })
                });
                return true;
            } catch (error) {
                console.error('Ошибка обновления сообщения:', error);
                return false;
            }
        }

        function startEdit(row, messageDiv, currentText, index) {
            if (currentEditRow) {
                cancelEdit(currentEditRow);
            }

            const parent = messageDiv.parentNode;
            
            const editorContainer = document.createElement('div');
            editorContainer.className = 'message editing';
            
            const textarea = document.createElement('textarea');
            textarea.className = 'edit-textarea';
            textarea.value = currentText;
            textarea.rows = 3;
            
            const actions = document.createElement('div');
            actions.className = 'edit-actions';
            
            const saveBtn = document.createElement('button');
            saveBtn.textContent = 'Сохранить';
            saveBtn.onclick = async () => {
                const newText = textarea.value.trim();
                if (newText && newText !== currentText) {
                    const success = await updateMessageOnServer(index, newText);
                    if (success) {
                        messageDiv.textContent = newText;
                        messageDiv.style.display = '';
                        editorContainer.remove();
                        currentEditRow = null;
                    } else {
                        alert('Ошибка сохранения');
                    }
                } else {
                    messageDiv.style.display = '';
                    editorContainer.remove();
                    currentEditRow = null;
                }
            };
            
            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = 'Отмена';
            cancelBtn.className = 'cancel';
            cancelBtn.onclick = () => {
                messageDiv.style.display = '';
                editorContainer.remove();
                currentEditRow = null;
            };
            
            actions.appendChild(saveBtn);
            actions.appendChild(cancelBtn);
            
            editorContainer.appendChild(textarea);
            editorContainer.appendChild(actions);
            
            messageDiv.style.display = 'none';
            parent.insertBefore(editorContainer, messageDiv.nextSibling);
            
            textarea.focus();
            currentEditRow = { row, messageDiv, editorContainer };
        }

        function cancelEdit(editState) {
            if (editState) {
                editState.messageDiv.style.display = '';
                editState.editorContainer.remove();
            }
            currentEditRow = null;
        }

        function addMessage(role, text, index = null, skipSave = false, checked = false) {
            const row = document.createElement('div');
            row.className = `message-row ${role}`;
            if (index !== null) {
                row.dataset.index = index;
            }
            
            const contentWrapper = document.createElement('div');
            contentWrapper.className = 'message-content-wrapper';
            
            const editBox = document.createElement('div');
            editBox.className = 'edit-box';
            const editBtn = document.createElement('button');
            editBtn.className = 'edit-btn';
            editBtn.textContent = '✎';
            editBtn.title = 'Редактировать сообщение';
            editBox.appendChild(editBtn);
            
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message';
            msgDiv.textContent = text;
            
            const toggleBox = document.createElement('div');
            toggleBox.className = 'toggle-box';
            const toggleBtn = createToggleButton(checked);
            toggleBox.appendChild(toggleBtn);
            
            contentWrapper.appendChild(editBox);
            contentWrapper.appendChild(msgDiv);
            contentWrapper.appendChild(toggleBox);
            row.appendChild(contentWrapper);
            messageList.appendChild(row);
            
            editBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const idx = row.dataset.index;
                if (idx) {
                    startEdit(row, msgDiv, msgDiv.textContent, parseInt(idx));
                }
            });
            
            scrollToBottom();

            if (!skipSave && !isLoadingHistory && index === null) {
                saveMessageToServer(role, text).then(newIndex => {
                    if (newIndex) {
                        row.dataset.index = newIndex;
                    }
                });
            }
        }

        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const text = messageInput.value.trim();
            if (!text) return;

            addMessage('user', text);
            messageInput.value = '';
            autoResize(messageInput);
            messageInput.focus();
        }

        async function fetchBotResponse() {
            try {
                const response = await fetch('/get_response', { method: 'POST' });
                const data = await response.json();
                if (data.response) {
                    addMessage('bot', data.response);
                }
            } catch (error) {
                console.error('Ошибка получения ответа:', error);
                addMessage('bot', 'Ошибка соединения с сервером.');
            }
        }

        async function loadHistory() {
            isLoadingHistory = true;
            try {
                const response = await fetch('/load_history');
                const messages = await response.json();
                messageList.innerHTML = '';
                for (const msg of messages) {
                    addMessage(msg.role, msg.content, msg.index, true, msg.checked);
                }
                scrollToBottom();
            } catch (error) {
                console.error('Ошибка загрузки истории:', error);
            } finally {
                isLoadingHistory = false;
            }
        }

        sendButton.addEventListener('click', sendMessage);
        getResponseButton.addEventListener('click', fetchBotResponse);
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        loadHistory();
        updateChatBottom();
        autoResize(messageInput);
    </script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Маршруты Flask
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': ''})
    try:
        response_text = chat_model.generate_response(user_message)
    except Exception as e:
        print(f"Error during generation: {e}")
        response_text = "Извините, произошла ошибка при генерации ответа."
    finally:
        # Выгружаем модели после каждого ответа
        chat_model.unload()
    return jsonify({'response': response_text})

@app.route('/get_response', methods=['POST'])
def get_response():
    """Генерирует ответ бота на последнее сообщение пользователя в истории."""
    try:
        response_text = chat_model.generate_response_from_history()
    except Exception as e:
        print(f"Error during generation: {e}")
        response_text = "Извините, произошла ошибка при генерации ответа."
    finally:
        chat_model.unload()
    return jsonify({'response': response_text})

@app.route('/save_message', methods=['POST'])
def save_message():
    data = request.get_json()
    role = data.get('role')
    text = data.get('text')
    if not role or not text:
        return jsonify({'error': 'Invalid data'}), 400
    index = save_message_to_file(role, text)
    return jsonify({'status': 'ok', 'index': index})

@app.route('/edit_message', methods=['POST'])
def edit_message():
    data = request.get_json()
    index = data.get('index')
    new_text = data.get('new_text')
    checked = data.get('checked')
    if index is None:
        return jsonify({'error': 'Invalid data'}), 400
    success = update_message_in_file(index, new_text, checked)
    if success:
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'error': 'Message not found'}), 404

@app.route('/load_history', methods=['GET'])
def load_history():
    messages = load_all_messages()
    cleaned = [{'role': m['role'], 'content': m['content'], 'index': m['index'], 'checked': m['checked']} for m in messages]
    return jsonify(cleaned)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Отправляет команду 'train' серверу обучения через ZeroMQ PUSH."""
    try:
        push_socket.send_string("train")
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/start_training_stage1', methods=['POST'])
def start_training_stage1():
    """Запускает только Stage 1."""
    try:
        push_socket.send_string("train_stage1")
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/start_training_stage2', methods=['POST'])
def start_training_stage2():
    """Запускает только Stage 2."""
    try:
        push_socket.send_string("train_stage2")
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ----------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Запуск веб-сервера на http://127.0.0.1:7860")
    print(f"История чата сохраняется в папку: {os.path.abspath(HISTORY_DIR)}")
    print("ZeroMQ PUSH сокет подключён к tcp://127.0.0.1:5555")
    app.run(host='127.0.0.1', port=7860, debug=False)