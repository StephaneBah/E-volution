import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import pdfplumber


def pdf_to_images(pdf_path):
    """ Convertit un PDF en liste d'images (une par page) """
    return convert_from_path(pdf_path, dpi=300)

def detect_text_boxes(image):
    """ Détecte les zones de texte et retourne leurs bounding boxes """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return boxes

def extract_text_from_boxes(image, boxes):
    """ Applique l'OCR sur chaque zone détectée """
    extracted_texts = []
    for (x, y, w, h) in boxes:
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, lang='fra', config='--psm 6')
        extracted_texts.append(text.strip())
    return extracted_texts

def process_pdf(pdf_path):
    """ Convertit le PDF en images, détecte les zones de texte et applique l'OCR """
    images = pdf_to_images(pdf_path)
    all_texts = []

    for page_num, image in enumerate(images):
        open_cv_image = np.array(image)
        boxes = detect_text_boxes(open_cv_image)
        texts = extract_text_from_boxes(open_cv_image, boxes)

        print(f"\n📄 Page {page_num + 1} :\n")
        for i, text in enumerate(texts):
            print(f"Zone {i+1}: {text}")
        
        all_texts.append(texts)
    
    return all_texts

# Exécution
pdf_path = "/home/stef/Téléchargements/Cover_Letter_Internship_DataAnalyst_ASIN.pdf"  # Remplace par ton fichier PDF
process_pdf(pdf_path)
