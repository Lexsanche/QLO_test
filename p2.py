import pytesseract
import cv2
import json
import time
from paddleocr import PaddleOCR
from typing import Dict

def extraer_texto_ocr(ruta_imagen: str) -> Dict:
    """Applies OCR on an image using Tesseract and returns extracted text."""
    inicio = time.time()
    imagen = cv2.imread(ruta_imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Apply Tesseract OCR
    texto_extraido = pytesseract.image_to_string(imagen_gris, lang='spa')
    
    fin = time.time()
    
    resultado = {
        "ruta_imagen": ruta_imagen,
        "texto_extraido": texto_extraido.strip(),
        "tiempo_procesamiento": round(fin - inicio, 2)
    }
    
    return resultado

def extraer_texto_paddleocr(ruta_imagen: str) -> Dict:
    """Applies OCR using PaddleOCR (better for structured text)."""
    inicio = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="es")  
    resultado_ocr = ocr.ocr(ruta_imagen, cls=True)

    # Extract text
    texto_extraido = "\n".join([line[1][0] for line in resultado_ocr[0] if line[1][0]])
    
    fin = time.time()
    
    resultado = {
        "ruta_imagen": ruta_imagen,
        "texto_extraido": texto_extraido.strip(),
        "tiempo_procesamiento": round(fin - inicio, 2)
    }
    
    return resultado

def guardar_resultado_json(resultado, ruta_salida):
    """Saves OCR results to a JSON file."""
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=4)
    print(f"Resultados guardados en {ruta_salida}")

def evaluar_ocr():
    """Tests OCR on different images and evaluates performance."""
    test_images = ["text.jpg", "images.png", "num.png"]
    
    for img in test_images:
        print(f"\n🔍 Evaluating OCR on: {img}")
        
        # Tesseract OCR
        resultado_tesseract = extraer_texto_ocr(img)
        guardar_resultado_json(resultado_tesseract, f"resultado_tesseract_{img}.json")

        # PaddleOCR (Alternative)
        resultado_paddleocr = extraer_texto_paddleocr(img)
        guardar_resultado_json(resultado_paddleocr, f"resultado_paddleocr_{img}.json")

        # Print Comparison
        print(f"📌 OCR Performance on {img}")
        print(f"Tesseract OCR Output:\n{resultado_tesseract['texto_extraido']}")
        print(f"Tesseract Processing Time: {resultado_tesseract['tiempo_procesamiento']}s")

        print(f"PaddleOCR Output:\n{resultado_paddleocr['texto_extraido']}")
        print(f"PaddleOCR Processing Time: {resultado_paddleocr['tiempo_procesamiento']}s")
        print("-" * 60)

if __name__ == "__main__":
    evaluar_ocr()
