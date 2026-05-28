import pymupdf.layout
import pymupdf4llm
import re
import pandas as pd

from utils.logging_config import setup_logger, log_and_print

logger = setup_logger(
    log_file_name="extract_from_pdf",
    logger_name="nectartron",
)


def clean_string(s):
    """Remove headers and footers from extracted text."""
    # Remove header pattern: date/time followed by whitespace and "Zimbra"
    # Example: "2/12/26, 6:44 PM    Zimbra"
    header_pattern = r'\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s+Zimbra'
    s = re.sub(header_pattern, '', s, flags=re.IGNORECASE)
    
    # Remove footer pattern: URL followed by whitespace and page numbers
    # Example: "https://llr-mail.in2p3.fr/h/printmessage?id=... 14/154"
    footer_pattern = r'https://llr-mail\.in2p3\.fr/h/printmessage\?id=[^\s]+\s+\d+/\d+'
    s = re.sub(footer_pattern, '', s, flags=re.IGNORECASE)
    
    return s.strip()


def extract_entries(text):
    """Extract all entries between start and end delimiters."""
    start_delim = "A new ELOG entry has been submitted:"
    end_delim = "ELOG V3.1.2-de42cea"
    
    entries = []
    start_idx = 0
    
    while True:
        # Find next entry start
        start = text.find(start_delim, start_idx)
        if start == -1:
            break
        
        # Find corresponding end
        end = text.find(end_delim, start)
        if end == -1:
            break
        
        # Extract text between delimiters (include the delimiters themselves)
        entry = text[start:end + len(end_delim)]
        entries.append(entry)
        
        # Move start index past this entry
        start_idx = end + len(end_delim)
    
    return entries


def extract_entry_info(entry):
    """Extract specific fields from an ELOG entry."""
    fields = [
        'Entry time', 
        'Author', 
        'Setup', 
        'Category', 
        'Keyword', 
        'ModuleCount', 
        'TriggerModes', 
        'LightSource'
    ]
    info = {}
    
    for field in fields:
        # Create a pattern to match "FieldName: value" until next field or end
        # The value continues until we hit another known field or end of string
        other_fields = [f for f in fields if f != field]
        field_names_pattern = '|'.join(re.escape(f) + r':' for f in other_fields)
        pattern = rf'{re.escape(field)}:\s*(.*?)(?=(?:{field_names_pattern})|$)'
        
        match = re.search(pattern, entry, re.DOTALL | re.IGNORECASE)
        
        if match:
            value = match.group(1).strip()
            # Remove markdown bold formatting (**text**)
            value = re.sub(r'\*\*(.*?)\*\*', r'\1', value)
            # Clean up multiple spaces/newlines, keeping single spaces
            value = re.sub(r'\s+', ' ', value).strip()
            # Remove ELOG version delimiter
            value = re.sub(r'ELOG\s+V[\d\.]+-[a-f0-9]+', '', value, flags=re.IGNORECASE)
            # Remove any remaining Zimbra patterns (both "date/time Zimbra" and "Zimbra date/time")
            value = re.sub(r'Zimbra\s+\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*(?:AM|PM)?', '', value, flags=re.IGNORECASE)
            value = re.sub(r'\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s+Zimbra', '', value, flags=re.IGNORECASE)
            # Final cleanup of extra spaces
            value = re.sub(r'\s+', ' ', value).strip()
            info[field] = value
        else:
            info[field] = ""
    
    return info


def main():

    doc = pymupdf.open("./data/saved_emails.pdf")

    md = pymupdf4llm.to_markdown(doc)

    # Extract all entries between delimiters
    entries = extract_entries(md)
    
    # Extract structured info from each entry
    data = []
    for entry in entries:
        cleaned = clean_string(entry)
        info = extract_entry_info(cleaned)
        data.append(info)
    
    # Create a DataFrame and display
    if data:
        df = pd.DataFrame(data)
        log_and_print(logger, df.to_string(index=False))
        log_and_print(logger, "\n")
        # Optionally save to CSV
        df.to_csv("./data/extracted_entries.csv", index=False)
        log_and_print(logger, "Data saved to extracted_entries.csv")
    else:
        log_and_print(logger, "No entries found.")



if __name__ == "__main__":
    main()