import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

def cargar_csv(ruta_archivo: str, columna_texto: str) -> pd.DataFrame:
    df = pd.read_csv(ruta_archivo)
    return df[[columna_texto]]

def dividir_texto(texto: str, tamano_chunk: int = 256) -> List[str]:
    palabras = texto.split()
    return [" ".join(palabras[i: i + tamano_chunk]) for i in range(0, len(palabras), tamano_chunk)]

def generar_embeddings(fragmentos_texto: List[str], modelo: str = "all-MiniLM-L6-v2") -> np.ndarray:
    embedder = SentenceTransformer(modelo)
    return np.array(embedder.encode(fragmentos_texto))

def almacenar_embeddings(embeddings: np.ndarray, ruta_indice: str = "vector_index.faiss") -> None:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, ruta_indice)
    print(f"Embeddings almacenados en {ruta_indice}")

def creacion_csv():
    data = {
        "texto": [
            "El aprendizaje automático está transformando la industria.",
            "Las redes neuronales profundas son un modelo poderoso.",
            "FAISS es una biblioteca eficiente para la búsqueda de similitud.",
            "Los embeddings convierten el texto en representaciones numéricas.",
            "La ciencia de datos es clave en la toma de decisiones empresariales.",
            "GPT-4 es un modelo avanzado de generación de lenguaje natural."
        ]
        }

    csv_file_path = "ejemplos_texto.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

    return csv_file_path


def main():
    ruta_archivo = creacion_csv()
    columna_texto = "texto"
    df = cargar_csv(ruta_archivo, columna_texto)
    fragmentos_texto = []
    
    for texto in df[columna_texto].dropna():
        fragmentos_texto.extend(dividir_texto(texto))
    
    embeddings = generar_embeddings(fragmentos_texto)
    almacenar_embeddings(embeddings)
    print("¡Proceso de generación de embeddings completado exitosamente!")

if __name__ == "__main__":

    main()
