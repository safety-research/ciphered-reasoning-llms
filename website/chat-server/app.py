from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import ray
from dotenv import load_dotenv
import asyncio
from vllm_actor import VllmActor, GPUInitScheduler

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from prompts import get_translation_prompt
from encoding_schemes import get_inverse_encoding_scheme

import os
import psycopg2
import json
import pandas as pd

conn_string = os.environ["SUPABASE_CONNECTION_URL"]

conn = psycopg2.connect(conn_string)

sel_str = """
-- NuminaMath CoT Rerun
 (
     (data->'experiment_tags'->'numina_math_cot_rerun')::BOOL
     AND (NOT (data->'force_overwrite')::BOOL OR data->'force_overwrite' IS NULL)
     AND (
         (data->'experiment_params'->'sampling_params'->'n')::INT = 4
     ) AND (
        (data->'experiment_params'->'model')::TEXT LIKE '%14B%'
     )
  )
"""

df_data = pd.read_sql(
    f"""
SELECT * FROM public.encoding_schemes 
    WHERE 
{sel_str}
ORDER BY created_at DESC
""",
    conn,
)

L_SHOWN_ENCODINGS = [
    # GPT
    "identity",
    "dot_between_chars",
    "Korean",
    "letter_to_word_with_dot",
    "rot13_cipher",
    "Morse_code",
    "base64_cipher",
    "reverse_letters_in_each_word",
    "reverse_fibonacci_indices_in_each_word",
    "base64_2x_cipher",
    "space_between_chars",
    "swap_even_odd_letters_in_each_word",
    "Python",
    "pirate_speak",
    "gzip_to_base64_encoded",
    "leet_speak",
]


# Available models
MODELS = [
    # {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI"},
]

# Dictionary to store Ray actor handles for each model
model_actors = {}


def populate_models():
    for _, row in df_data.iterrows():
        if not row["data"]["experiment_name"].startswith("math_cot"):
            continue

        # if len(MODELS) > 2:
        #     break

        encoding_scheme = row["data"]["experiment_params"]["encoding_scheme"]
        encoding_scheme = encoding_scheme.replace("speaking_", "")

        if encoding_scheme not in L_SHOWN_ENCODINGS:
            continue

        encoding_scheme = encoding_scheme.replace("_", " ")

        MODELS.append(
            {
                "id": row["experiment_hash"],
                "name": encoding_scheme,
                "provider": row["data"]["experiment_params"]["model"],
            }
        )


def init_ray_actors():
    # Create an actor for each unique provider
    scheduler = GPUInitScheduler.remote()

    for model in MODELS:
        model_id = model["id"]
        # Create VllmActor with the model path
        actor = VllmActor.remote(
            model_path=f"/ext_data/output/{model_id}/sft_model/last",
            gpu_init_scheduler=scheduler,
        )
        model_actors[model_id] = actor
        print(f"Initialized VllmActor for model: {model}")

    for actor in model_actors.values():
        ray.get(actor.__ray_ready__.remote())


load_dotenv()

app = Flask(__name__)
# Allow CORS from any localhost port
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available AI models"""
    return jsonify(MODELS)


@app.route("/api/prompts/<model_id>", methods=["GET"])
def get_prompts(model_id):
    """Get prompts for a specific model"""

    model_info = None
    for model in MODELS:
        if model["id"] == model_id:
            model_info = model
            break

    assert model_info is not None

    model_cipher = model["name"]
    model_cipher = model_cipher.replace(" ", "_")
    model_cipher = f"speaking_{model_cipher}"

    return get_translation_prompt(model_cipher)


@app.route("/api/messages", methods=["POST"])
def send_message():
    """Process messages and return AI response"""
    data = request.get_json()
    conversation_messages = data["messages"]
    model_id = data["model"]

    actor = model_actors[model_id]

    # Call the chat method through Ray
    response_content = ray.get(actor.chat.remote(conversation_messages, {}))
    return jsonify(
        {"content": response_content, "role": "assistant", "model": model_id}
    )


@app.route("/api/translate", methods=["POST"])
def translate_text():
    """Translate encoded text back to original using inverse encoding scheme"""
    try:
        data = request.get_json()
        encoded_text = data.get("text", "")
        model_id = data.get("model_id")

        if not model_id:
            return jsonify({"error": "model_id is required"}), 400

        # Find the model info to get the encoding scheme name
        model_info = None
        for model in MODELS:
            if model["id"] == model_id:
                model_info = model
                break

        if not model_info:
            return jsonify({"error": f"Model {model_id} not found"}), 404

        # Get the encoding scheme name (same logic as get_prompts)
        model_cipher = model_info["name"]
        model_cipher = model_cipher.replace(" ", "_")
        model_cipher = f"speaking_{model_cipher}"

        # Get the inverse encoding function (config param is not used in the function)
        try:
            inverse_fn = get_inverse_encoding_scheme(model_cipher, {})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

        # Decode the text (handle both sync and async functions)
        import inspect

        if inspect.iscoroutinefunction(inverse_fn):
            # Handle async functions
            decoded_text = asyncio.run(inverse_fn(encoded_text))
        else:
            # Handle sync functions
            decoded_text = inverse_fn(encoded_text)

        return jsonify(
            {
                "original_text": decoded_text,
                "encoded_text": encoded_text,
                "encoding_scheme": model_cipher,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    ray.init()
    populate_models()

    init_ray_actors()

    context = (
        "/etc/letsencrypt/live/cipheredreasoning.app/fullchain.pem",
        "/etc/letsencrypt/live/cipheredreasoning.app/privkey.pem",
    )

    app.run(debug=False, host="0.0.0.0", port=port, ssl_context=context)
