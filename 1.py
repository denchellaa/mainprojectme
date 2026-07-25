import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MEMORY_FILE = "memory.json"

SYSTEM_PROMPT = """
Ты — ИИ-агент по бизнес-идеям и запуску малого бизнеса.

Твоя задача — давать пользователю конкретные бизнес-идеи и планы запуска.

Важно:
1. Запоминай данные, которые пользователь уже сообщил в диалоге.
2. Не задавай повторно вопросы, на которые пользователь уже ответил.
3. Если пользователь отвечает коротко, например "1000000", понимай это как ответ на последний заданный вопрос.
4. Если данных достаточно — не продолжай бесконечно уточнять, а дай конкретный ответ.
5. Если пользователь спрашивает про экономическую ситуацию на дату после твоих достоверных знаний, прямо скажи, что без свежих данных ты можешь дать только сценарный анализ.
6. Отвечай конкретно, без воды.

Если пользователь хочет бизнес по продаже автомобилей, анализируй:
- бюджет
- город/регион
- формат: перекуп, автоподбор, комиссионная продажа, привоз авто, trade-in, выкуп проблемных авто
- риски
- маржинальность
- где искать машины
- где искать клиентов
- юридические нюансы
- план на первые 7 дней
- план на первый месяц

Формат ответа, если данных достаточно:

1. Краткий вывод
2. Подходит ли идея пользователю
3. Лучший формат запуска
4. Почему именно этот формат
5. Стартовый бюджет
6. Что покупать/продавать
7. Где искать автомобили
8. Где искать клиентов
9. Финансовая модель
10. Риски
11. План на 7 дней
12. План на 30 дней
13. Что не делать

ВАЖНО про работу с памятью:
- В конце каждого своего ответа, если ты узнал новые факты о пользователе (бюджет, город, формат, опыт, цель и т.д.),
  добавь СТРОГО в самом конце блок в формате:
  <MEMORY_UPDATE>{"ключ": "значение", ...}</MEMORY_UPDATE>
- Этот блок не показывай как часть ответа пользователю — он будет вырезан автоматически.
- Пиши туда только новые или изменившиеся факты, не дублируй уже известные.
- Ключи на английском snake_case: budget, city, business_type, format, experience, goal, и т.п.
"""


# ---------- РАБОТА С ПАМЯТЬЮ ----------

def load_memory():
    """Загружает память из JSON-файла или создаёт пустую структуру."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "facts": {},          # структурированные факты о пользователе
        "history": []         # история сообщений (user/assistant)
    }


def save_memory(memory):
    """Сохраняет память в JSON-файл."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def extract_memory_update(answer: str):
    """
    Вытаскивает блок <MEMORY_UPDATE>{...}</MEMORY_UPDATE> из ответа модели.
    Возвращает (чистый_ответ, dict_с_новыми_фактами).
    """
    start_tag = "<MEMORY_UPDATE>"
    end_tag = "</MEMORY_UPDATE>"

    if start_tag in answer and end_tag in answer:
        start = answer.index(start_tag)
        end = answer.index(end_tag) + len(end_tag)
        raw = answer[start + len(start_tag):answer.index(end_tag)].strip()

        clean_answer = (answer[:start] + answer[end:]).strip()

        try:
            new_facts = json.loads(raw)
            if not isinstance(new_facts, dict):
                new_facts = {}
        except json.JSONDecodeError:
            new_facts = {}

        return clean_answer, new_facts

    return answer, {}


def build_system_prompt(memory):
    """Подмешивает известные факты о пользователе в системный промпт."""
    facts = memory.get("facts", {})
    if not facts:
        facts_block = "Пока никаких фактов о пользователе не сохранено."
    else:
        facts_block = "\n".join(f"- {k}: {v}" for k, v in facts.items())

    return SYSTEM_PROMPT + f"\n\nИЗВЕСТНЫЕ ФАКТЫ О ПОЛЬЗОВАТЕЛЕ (из памяти):\n{facts_block}\n"


# ---------- ОБЩЕНИЕ С АГЕНТОМ ----------

def ask_agent(user_message: str, memory):
    # Собираем сообщения: system + вся история + новый user
    messages = [{"role": "system", "content": build_system_prompt(memory)}]
    messages.extend(memory["history"])
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    raw_answer = response.choices[0].message.content
    clean_answer, new_facts = extract_memory_update(raw_answer)

    # Обновляем факты
    if new_facts:
        memory["facts"].update(new_facts)

    # Обновляем историю (сохраняем уже очищенный ответ)
    memory["history"].append({
        "role": "user",
        "content": user_message,
        "ts": datetime.now().isoformat()
    })
    memory["history"].append({
        "role": "assistant",
        "content": clean_answer,
        "ts": datetime.now().isoformat()
    })

    save_memory(memory)
    return clean_answer, new_facts


# ---------- ВСПОМОГАТЕЛЬНЫЕ КОМАНДЫ ----------

def print_help():
    print("Команды:")
    print("  /memory  — показать сохранённые факты")
    print("  /history — показать историю диалога")
    print("  /reset   — очистить память")
    print("  выход    — выйти")
    print()


def main():
    memory = load_memory()

    print("ИИ-агент бизнес-идей запущен.")
    if memory["facts"]:
        print(f"Загружена память: {len(memory['facts'])} фактов, "
              f"{len(memory['history'])} сообщений в истории.")
    print("Напиши, какой бизнес ты хочешь, или /help для списка команд.")
    print()

    while True:
        user_input = input("Ты: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["выход", "exit", "quit"]:
            print("Агент: Удачи с бизнесом!")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/memory":
            print("Факты в памяти:")
            print(json.dumps(memory["facts"], ensure_ascii=False, indent=2))
            print()
            continue

        if user_input == "/history":
            for msg in memory["history"]:
                role = msg["role"]
                content = msg["content"][:120].replace("\n", " ")
                print(f"[{role}] {content}...")
            print()
            continue

        if user_input == "/reset":
            memory = {"facts": {}, "history": []}
            save_memory(memory)
            print("Память очищена.")
            print()
            continue

        answer, new_facts = ask_agent(user_input, memory)

        print()
        print("Агент:")
        print(answer)
        if new_facts:
            print()
            print(f"🧠 Запомнил: {new_facts}")
        print("-" * 80)


if __name__ == "__main__":
    main()
