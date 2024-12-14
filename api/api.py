from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
api_key = os.getenv("OPENAI_API_KEY")
print("Api key")
print(api_key)

print(len(api_key))
if not api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada.")


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


output_dir = "model/emotion_generator"
model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

app = FastAPI()

class EmotionRequest(BaseModel):
    texto: str
    emocion: str

def generar_secuencia_emocional(texto, emocion):
    prompt = f"""
    Texto: {texto}
    Emoción: {emocion}
    Basado en el texto y la emoción proporcionada, genera una secuencia creativa y coherente que refleje la misma emoción en el contenido de manera corta y concisa,
    en el ámbito de pokemon.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un profesor pokemon y quiero que comentes dálogos basado en emociones."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.8,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=[".", "\n"]
        )
        print(chat_completion)
        return chat_completion.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar texto: {str(e)}")



class TextGenerationInput(BaseModel):
    prompt: str
    emotion: str = None
    max_length: int = 50
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

@app.get("/")
def read_root():
    return {"message": "Welcome to the text generation API!"}


@app.post("/generate-sequence-api/")
def generate_emotion_sequence(request: EmotionRequest):
    if request.emocion.lower() not in ["sadness", "joy", "love", "anger", "fear", "surprise"]:
        raise HTTPException(
            status_code=400,
            detail="La emoción proporcionada no es válida. Use una de las siguientes: sadness, joy, love, anger, fear, surprise."
        )
    
    try:
        generated_text = generar_secuencia_emocional(request.texto, request.emocion)
        # return {"original_message": request.texto, "emotion": request.emocion, "message": generated_text}
        return {"message": generated_text}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno al procesar la solicitud.")


# Definir el endpoint
@app.post("/generate-text/")
def generate_text(input: TextGenerationInput):
    def generate_text_with_control(model, tokenizer, prompt, max_length=50, emotion=None, repetition_penalty=1.2, temperature=0.7, top_k=50, top_p=0.95):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        if emotion and emotion in ["sadness", "joy", "love", "anger", "fear", "surprise"]:
            emotion_token = f"<{emotion}>"
            input_ids = tokenizer.encode(f"<start> {emotion_token} {prompt} <end>", return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids).to(device) 

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        model.config.pad_token_id = tokenizer.pad_token_id

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        for token in ["<start>", "<end>", "<sadness>", "<joy>", "<love>", "<anger>", "<fear>", "<surprise>"]:
            generated_text = generated_text.replace(token, "").strip()
        return generated_text

    # Generar el texto usando los parámetros enviados a la API
    generated_text = generate_text_with_control(
        model=model,
        tokenizer=tokenizer,
        prompt=input.prompt,
        max_length=input.max_length,
        emotion=input.emotion,
        repetition_penalty=input.repetition_penalty,
        temperature=input.temperature,
        top_k=input.top_k,
        top_p=input.top_p
    )
    return {"message": generated_text}
