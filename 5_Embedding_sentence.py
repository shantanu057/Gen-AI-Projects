from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from io import StringIO
import PyPDF2

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  #model we are using for embeddings
head_embedding =[]
def extract_headings_from_pdf(pdf_path):
    headings =[]

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            text = page.extract_text()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            headings.extend([line for line in lines if line[0].isupper()])
    return headings

path = './'

extracted_Headings = extract_headings_from_pdf(path + 'PDF2.pdf')
embeddings = model.encode(extracted_Headings)
for heading, embedding in zip(extracted_Headings,embeddings):
  embedded_array_string = '[' + ','.join(map(str, embedding)) + ']'
  print("embedded_array_string", embedded_array_string)

print("Embeddings", embeddings)
print("Headings", extracted_Headings)