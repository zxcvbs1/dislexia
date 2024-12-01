# _*_ coding: iso-8859-1 _*_
from bs4 import BeautifulSoup
import re
import json
import logging
import pandas as pd
from datetime import datetime
 
# Configuración del logging
logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s\n-------------------\n'
)
 
def extract_datetime(title):
    """Extrae fecha y hora del atributo title"""
    match = re.search(r'(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2})', title)
    if match:
        date, time = match.groups()
        return date.replace('.', '/'), time
    return None, None
 
def extract_trade_section(text):
    """Extrae la sección entre FX Professor® Hedge Trade: y Disclaimer:"""
    try:
        match = re.search(r'Hedge Trade:(.*?)Disclaimer', 
                         text, 
                         re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    except Exception as e:
        logging.error(f"Error extracting trade section: {str(e)}")
        return None
 
def process_file(input_file, output_file_json, output_file_csv):
    messages = []
    null_trade_sections = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        # Encuentra todos los divs con clase "date details"
        date_divs = soup.find_all('div', class_='pull_right date details')
        
        for date_div in date_divs:
            date_title = date_div.get('title')
            if not date_title:
                continue
                
            date, time = extract_datetime(date_title)
            if not date or not time:
                logging.warning(f"Formato de fecha inválido: {date_title}")
                continue
            
            # Busca el div de texto más cercano
            text_div = date_div.find_next('div', class_='text')
            if not text_div:
                logging.warning(f"No se encontró texto para la fecha: {date} {time}")
                continue
                
            # Guarda el texto completo
            text_content = text_div.get_text()
            
            # Extrae la sección de trade
            trade_section = extract_trade_section(text_content)
            
            message = {
                "date": date,
                "time": time,
                "full_content": text_content,
                "trade_section": trade_section if trade_section else None
            }
            
            # Si trade_section es None, guarda el mensaje para el log
            if trade_section is None:
                null_trade_sections.append({
                    "date": date,
                    "time": time,
                    "content": text_content
                })
            
            messages.append(message)
        
        # Guarda los resultados en formato JSON
        with open(output_file_json, 'w', encoding='utf-8') as out_file:
            json.dump(messages, out_file, indent=2, ensure_ascii=False)
            
        # Guarda los resultados en formato CSV
        df = pd.DataFrame(messages)
        df.to_csv(output_file_csv, index=False, encoding='utf-8', sep=";")
        
        # Log de mensajes sin trade_section
        if null_trade_sections:
            logging.warning("Mensajes sin sección de trade encontrados:")
            for msg in null_trade_sections:
                logging.warning(f"""
Fecha: {msg['date']} {msg['time']}
Contenido:
{msg['content']}
----------------------------------------
""")
            
        logging.info(f"""
Resumen del procesamiento:
Total de mensajes procesados: {len(messages)}
Mensajes sin sección de trade: {len(null_trade_sections)}
""")
        
    except Exception as e:
        logging.error(f"Error en el procesamiento del archivo: {str(e)}")
 
if __name__ == "__main__":
    input_file = 'messages.html'
    output_file_json = 'messages.json'
    output_file_csv = 'messages.csv'
    process_file(input_file, output_file_json, output_file_csv)