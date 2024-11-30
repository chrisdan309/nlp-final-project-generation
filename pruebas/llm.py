import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

print("Cargando el dataset...")
df = pd.read_parquet('corpus/train-00000-of-00001.parquet')

print("Primeras filas del dataset:")
print(df.head())

emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

df['emotion'] = df['label'].map(emotion_map)

df['input_text'] = df.apply(lambda row: f"{row['emotion']}: {row['text']}", axis=1)

print("Convirtiendo el DataFrame a un Dataset de Hugging Face...")
dataset = Dataset.from_pandas(df[['input_text', 'emotion']])

print("Verificando el dataset:")
print(dataset['input_text'][:5])
print(dataset['emotion'][:5])

print("Cargando el modelo y el tokenizador preentrenados...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

print("Redimensionando los embeddings del modelo para aceptar los nuevos tokens...")
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    print("Tokenizando los textos...")
    inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = inputs['input_ids']
    return inputs


print("Aplicando la tokenización...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("Muestra de datos tokenizados:")
print(tokenized_datasets[0])

print("Configurando los parámetros de entrenamiento...")
training_args = TrainingArguments(
    output_dir="./results",          
    eval_strategy="epoch",
    learning_rate=2e-5,              
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,    
    num_train_epochs=3,              
    weight_decay=0.01,               
    logging_dir='./logs',            
)


trainer = Trainer(
    model=model,                       
    args=training_args,                
    train_dataset=tokenized_datasets,  
    eval_dataset=tokenized_datasets,   
)

print("Comenzando el entrenamiento del modelo...")
trainer.train()

def generate_text(emotion, prompt, max_length=50):
    print(f"Generando texto con la emoción '{emotion}' y el prompt '{prompt}'...")
    input_text = f"{emotion}: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

generated_text = generate_text("joy", "I just won a prize!")
print("Generado con emoción 'joy':")
print(generated_text)

generated_text = generate_text("sadness", "I lost my keys")
print("\nGenerado con emoción 'sadness':")
print(generated_text)

print("Guardando el modelo y el tokenizador ajustados...")
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print("Entrenamiento completado y modelo guardado.")
