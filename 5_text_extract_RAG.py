import pandas as pd
import os
from io import StringIO
import PyPDF2

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

extracted_Headings = extract_headings_from_pdf(path + 'Connectivity_issue.pdf')

print("Headings", extracted_Headings)