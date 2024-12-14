import pandas as pd
import os

def preprocess_twitter_data(input_parquet, output_file):
    label_to_emotion = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    df = pd.read_csv(input_parquet)
    df['emotion'] = df['label'].map(label_to_emotion)

    # Determinar el número mínimo de ejemplos por categoría
    min_count = df['emotion'].value_counts().min()
    
    balanced_df = df.groupby('emotion').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    balanced_df = df.groupby('emotion', group_keys=False).apply(
        lambda x: x.sample(n=min_count, random_state=42)
    ).reset_index(drop=True)

    # df['formatted'] = df['emotion'].apply(lambda x: f"<start> <{x}>") + " " + df['text'] + " <end>"

    balanced_df['formatted'] = balanced_df['emotion'].apply(lambda x: f"<start> <{x}>") + " " + balanced_df['text'] + " <end>"



    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(balanced_df['formatted']))

    print(f"Datos preprocesados guardados en {output_file}")


if __name__ == "__main__":
    input_parquet = "corpus/train-00000-of-00001.parquet"
    output_file_train = "data/train.txt"
    output_file_valid = "data/valid.txt"

    if not os.path.exists('data'):
        os.makedirs('data')

    df = pd.read_parquet(input_parquet)
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)

    train_df.to_csv("data/train_data.csv", index=False)
    valid_df.to_csv("data/valid_data.csv", index=False)

    preprocess_twitter_data("data/train_data.csv", output_file_train)
    preprocess_twitter_data("data/valid_data.csv", output_file_valid)