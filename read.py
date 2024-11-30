import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('corpus/train-00000-of-00001.parquet')

label_counts = df['label'].value_counts()

label_counts.plot(kind='bar', title='Distribución de emociones', xlabel='Emociones', ylabel='Frecuencia')
plt.show()

label_proportion = df['label'].value_counts(normalize=True)
print(label_proportion)


df['text_length'] = df['text'].apply(len)
print(df['text_length'].describe())

df['text_length'].plot(kind='hist', bins=30, title='Distribución de la longitud del texto')
plt.xlabel('Longitud del texto')
plt.show()

length_by_label = df.groupby('label')['text_length'].describe()
print(length_by_label)
