import pytesseract
import cv2
import json
import time
from typing import Dict

def extraer_texto_ocr(ruta_imagen):
    inicio = time.time()
    imagen = cv2.imread(ruta_imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    texto_extraido = pytesseract.image_to_string(imagen_gris, lang='spa')
    fin = time.time()
    
    resultado = {
        "ruta_imagen": ruta_imagen,
        "texto_extraido": texto_extraido.strip(),
        "tiempo_procesamiento": round(fin - inicio, 2)
    }
    
    return resultado

def guardar_resultado_json(resultado, ruta_salida):
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=4)
    print(f"Resultados guardados en {ruta_salida}")

def main():
    ruta_imagen = "text.jpg"  
    ruta_salida = "resultado.json"
    
    resultado = extraer_texto_ocr(ruta_imagen)
    guardar_resultado_json(resultado, ruta_salida)
    
    print("Texto extraído:")
    print(resultado["texto_extraido"])
    print(f"Tiempo de procesamiento: {resultado['tiempo_procesamiento']} segundos")

if __name__ == "__main__":
    main()
