import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
    return dataset

def train_model(train_file, valid_file, output_dir):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for emotion in emotions:
        tokenizer.add_tokens(f"<{emotion}>")
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = preprocess_dataset(train_file, tokenizer)
    valid_dataset = preprocess_dataset(valid_file, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir, # Parámetros de salida
        overwrite_output_dir=True,
        num_train_epochs=10, # Parámetros de entrenamiento
        per_device_train_batch_size=8,
        save_steps=500, # Parámetros de guardado
        save_total_limit=2,
        logging_dir="./logs", # Parámetros de logging
        logging_steps=100,
        evaluation_strategy="steps", # Parámetros evaluación
        eval_steps=500,
        learning_rate=5e-5, # Parámetros optimizador
        warmup_steps=100,
        weight_decay=0.01,
        push_to_hub=False # Parámetros de subida
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(output_dir)
    print(f"Modelo guardado en {output_dir}")


if __name__ == "__main__":
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    output_dir = "model/emotion_generator"

    train_model(train_file, valid_file, output_dir)
