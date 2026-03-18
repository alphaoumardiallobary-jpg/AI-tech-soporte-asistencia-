import os
import json
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory


# CONFIGURACIÓN


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Falta OPENAI_API_KEY en el archivo .env")

# Modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Memoria
chat_history = InMemoryChatMessageHistory()


# PROMPT


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un asistente de soporte técnico profesional.

Debes analizar el problema del usuario y responder SOLO en formato JSON válido.

Categorías permitidas:
- Red
- Hardware
- Software
- Acceso/Credenciales
- Rendimiento
- Otros

Devuelve EXACTAMENTE este formato JSON:

{{
  "category": "...",
  "summary": "...",
  "causes": ["...", "..."],
  "steps": ["...", "..."],
  "escalate": "Sí o No"
}}

Reglas:
- NO escribas texto fuera del JSON
- NO inventes categorías
- SIEMPRE usa las categorías dadas
"""
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)


# PARSER JSON

def parse_json_response(response):
    try:
        text = response.content.strip()

        
        start = text.find("{")
        end = text.rfind("}") + 1

        json_text = text[start:end]

        return json.loads(json_text)

    except Exception:
        print("\n Error parseando JSON. Respuesta original:\n")
        print(response.content)
        return None



# FUNCIONES AUXILIARES


def get_next_ticket_id():
    try:
        with open("tickets.json", "r", encoding="utf-8") as f:
            lines = f.readlines()
            return len(lines) + 1
    except FileNotFoundError:
        return 1


def detect_priority(question):
    question = question.lower()

    if any(word in question for word in ["no funciona", "error", "no puedo", "caído"]):
        return "Alta"
    elif any(word in question for word in ["lento", "problema", "falla"]):
        return "Media"
    else:
        return "Baja"


def save_ticket(question, parsed_response):
    ticket = {
        "id": get_next_ticket_id(),
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "category": parsed_response["category"],
        "priority": detect_priority(question),
        "data": parsed_response
    }

    with open("tickets.json", "a", encoding="utf-8") as f:
        json.dump(ticket, f, ensure_ascii=False)
        f.write("\n")


# CHAIN


chain = prompt | llm


# APP PRINCIPAL


def run_chat():
    print("Asistente de soporte técnico")
    print("Escribe 'salir' para terminar.\n")

    while True:
        user_input = input(" Tú: ").strip()

        if user_input.lower() == "salir":
            print("\nHasta luego.")
            break

        if not user_input:
            print("Por favor, escribe una consulta.\n")
            continue

        try:
            # Llamada al modelo
            raw_response = chain.invoke(
                {
                    "question": user_input,
                    "history": chat_history.messages,
                }
            )

            # Parsear JSON
            parsed = parse_json_response(raw_response)

            if parsed:
                print("\nAsistente:\n")
                print(f"Categoría: {parsed['category']}")
                print(f"Resumen: {parsed['summary']}")
                print(f"Causas: {parsed['causes']}")
                print(f"Pasos: {parsed['steps']}")
                print(f"Escalar: {parsed['escalate']}")
                print("\n" + "-" * 60 + "\n")

                # Guardar memoria
                chat_history.add_user_message(user_input)
                chat_history.add_ai_message(raw_response)

                # Guardar ticket
                save_ticket(user_input, parsed)

        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    run_chat()