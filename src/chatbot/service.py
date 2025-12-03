from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
import os
import base64
from typing import Any

load_dotenv(find_dotenv())

google_api_key = os.environ.get('GOOGLE_API_KEY')
llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=google_api_key)

def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

def chat(message: str):
    """Enviar un mensaje genérico al LLM y devolver la respuesta.

    Este método no usa documentos recuperados; construye un prompt genérico
    con un `SystemMessage` que define el comportamiento (agente HapticVision)
    y el `HumanMessage` con el `message` recibido.
    """
    # Construimos los mensajes
    messages = [
        SystemMessage(
            content=(
                "Eres un asistente inteligente para la aplicación HapticVision. Tu objetivo es ayudar al usuario a navegar por la aplicación y controlar sus funciones mediante comandos de voz. Responde de manera breve y útil."
            )
        ),
        HumanMessage(
            content=message
        )
    ]

    # Ejecuta el modelo con los mensajes
    response = llm.invoke(messages)
    print("LLM response:", response)
    
    return word_wrap(str(response))

def tool_calling(message: str):
    # Declaraciones de herramientas para el control de la aplicación
    
    # Herramienta de navegación
    navigate_fn = {
        "name": "navigate",
        "description": "Navegar a una pantalla específica de la aplicación.",
        "parameters": {
            "type": "object",
            "properties": {
                "screen": {
                    "type": "string",
                    "enum": ["camera", "haptic", "settings", "home"],
                    "description": "La pantalla a la que se desea ir."
                }
            },
            "required": ["screen"]
        },
    }

    # Herramienta de feedback háptico
    haptic_fn = {
        "name": "haptic_feedback",
        "description": "Generar una vibración háptica correspondiente a una emoción.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": ["neutral", "happy", "angry", "sad"],
                    "description": "La emoción para generar la vibración."
                }
            },
            "required": ["emotion"]
        },
    }

    # Herramienta de control de cámara
    camera_fn = {
        "name": "camera_control",
        "description": "Controlar la cámara (ej. cambiar entre frontal y trasera).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["switch_camera"],
                    "description": "La acción a realizar en la cámara."
                }
            },
            "required": ["action"]
        },
    }

    client = genai.Client()
    tool_spec = types.Tool(function_declarations=[navigate_fn, haptic_fn, camera_fn])
    config = types.GenerateContentConfig(tools=[tool_spec])

    # Prompt del sistema para definir el comportamiento del modelo
    system_instruction = "Eres un asistente inteligente para la aplicación HapticVision. Tu objetivo es ayudar al usuario a navegar por la aplicación y controlar sus funciones (cámara, vibración, configuración) mediante comandos de voz. Interpreta la intención del usuario y llama a la herramienta adecuada."

    # Usamos el mensaje del usuario como contenido para el modelo
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text=system_instruction)]),
            types.Content(role="user", parts=[types.Part(text=message)])
        ],
        config=config,
    )

    # Verificar si el modelo solicitó una llamada a función
    if response.candidates and response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]
        if part.function_call:
            return {
                "name": part.function_call.name,
                "args": dict(part.function_call.args)
            }
    
    return None