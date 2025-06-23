from PyPDF2 import PdfReader
import csv
import re

def extract_text_from_pdf(pdfPath):
    reader = PdfReader(pdfPath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


text = extract_text_from_pdf("mlPdf/L11.pdf")

def save_to_csv(text, csv_filename):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for para in paragraphs:
            writer.writerow([para])

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    text = re.sub(r' +', ' ', text)     # Collapse multiple spaces
    return text.strip()

text = clean_text(text)

print(text)
save_to_csv(text, 'mlCsv/L11.csv')