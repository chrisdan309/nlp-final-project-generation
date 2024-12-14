import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_dataset(file_path, tokenizer, block_size=64):
    print(f"Preprocesando el dataset desde el archivo: {file_path}")
    dataset = load_dataset("text", data_files=file_path)["train"]
    print(f"Tokenizando los textos del archivo: {file_path}")
    dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size),
        batched=True
    )
    print(f"Preprocesamiento completado para el archivo: {file_path}")
    return dataset

def compute_metrics(p):
    predictions, labels = p
    logits = predictions
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}

def train_model(train_file, valid_file, output_dir):
    print("Iniciando la carga y entrenamiento del modelo...")

    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"bos_token": "<start>", "eos_token": "<end>"})
    tokenizer.pad_token = tokenizer.eos_token

    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for emotion in emotions:
        tokenizer.add_tokens(f"<{emotion}>")

    print(f"Modelo y tokenizador cargados: {model_name}")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    print("Cargando y preprocesando los datasets de entrenamiento y validaciÃ³n...")
    train_dataset = preprocess_dataset(train_file, tokenizer)
    valid_dataset = preprocess_dataset(valid_file, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        eval_strategy="no",
        eval_steps=10,
        save_steps=200,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=500,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        fp16=True
    )

    print("Configurando el entrenador...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Iniciando el entrenamiento del modelo...")
    torch.cuda.empty_cache()  # Liberar memoria GPU antes de entrenar

    trainer.train()

    print("Guardando el modelo entrenado...")
    trainer.save_model(output_dir)
    print(f"Modelo guardado en {output_dir}")
    
    print("Entrenamiento completado.")
    # trainer.evaluate(valid_dataset)
    torch.cuda.empty_cache()
    eval_results = trainer.evaluate(valid_dataset)

    # Guardar los resultados en un archivo JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    

if __name__ == "__main__":
    print("Ejecutando el script...")
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    output_dir = "model/emotion_generator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio {output_dir} creado.")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    train_model(train_file, valid_file, output_dir)