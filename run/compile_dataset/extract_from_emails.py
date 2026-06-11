import os
import pandas as pd

from dotenv import load_dotenv

from utils.webmail import (
    authenticate, search_messages, get_message, 
    extract_hidden_inputs_parser, extract_messageframe_content_parser, 
    extract_entry_time
)
from utils.logging_config import setup_logger, log_and_print


logger = setup_logger(
    log_file_name="extract_from_emails",
    logger_name="nectartron",
)

load_dotenv("./config/.env.local")

webmail_soap_url = os.getenv("WEBMAIL_SOAP_URL")
webmail_username = os.getenv("WEBMAIL_USERNAME")
webmail_password = os.getenv("WEBMAIL_PASSWORD")


fields = [
    'Message ID',
    'Entry time', 
    'Author', 
    'Subject',
    'Setup', 
    'Category', 
    'Keyword', 
    'ModuleCount', 
    'TriggerModes', 
    'LightSource',
    'Content'
]
    


def main():
    log_and_print(logger, "Starting process...")

    # Authenticate
    auth_token = authenticate(
        soap_url=webmail_soap_url,
        username=webmail_username,
        password=webmail_password
    )
    log_and_print(logger, "✓ Authenticated successfully.")

    target_folder = "/Inbox/NectarCAM"
    log_and_print(logger, f"Extracting messages from: {target_folder}")

    message_ids = search_messages(
        soap_url=webmail_soap_url,
        auth_token=auth_token,
        folder_path=target_folder,
        limit=200
    )
    log_and_print(logger, f"✓ Found {len(message_ids)} messages")

    message_ids_to_write = []
    entry_times = []
    authors = []
    subjects = []
    setups = []
    categories = []
    keywords = []
    module_counts = []
    trigger_modes = []
    light_sources = []
    contents = []

    # Retrieve and display the messages
    for msg_id in message_ids:
        message = get_message(
            soap_url=webmail_soap_url,
            auth_token=auth_token,
            message_id=msg_id
        )
        email_subject = message.get('subject', 'N/A')
        if "Logbook entry" in email_subject:
            body = message.get('body', '')

            log_and_print(logger, f"\n\n--- Message ID: {msg_id} ---")

            mail_data = extract_hidden_inputs_parser(html_text=body)
            content = extract_messageframe_content_parser(html_text=body)
            mail_data['Content'] = content
            entry_time = extract_entry_time(html_text=body)
            mail_data['Entry time'] = entry_time
            mail_data['Message ID'] = msg_id

            for field in fields:
                if field not in mail_data.keys():
                    mail_data[field] = " "

            message_ids_to_write.append(mail_data['Message ID'])            
            entry_times.append(mail_data['Entry time'])
            authors.append(mail_data['Author'])
            subjects.append(mail_data['Subject'])
            setups.append(mail_data['Setup'])
            categories.append(mail_data['Category'])
            keywords.append(mail_data['Keyword'])
            module_counts.append(mail_data['ModuleCount'])
            trigger_modes.append(mail_data['TriggerModes'])
            light_sources.append(mail_data['LightSource'])
            contents.append(mail_data['Content'])


    df = pd.DataFrame({
        'Message ID': message_ids_to_write,
        'Entry time': entry_times, 
        'Author': authors, 
        'Subject': subjects,
        'Setup': setups, 
        'Category': categories, 
        'Keyword': keywords, 
        'ModuleCount': module_counts, 
        'TriggerModes': trigger_modes, 
        'LightSource': light_sources,
        'Content': contents   
    })
    df.to_csv("./data/extracted_entries.csv", index=False)    
    log_and_print(logger, "Data saved to extracted_entries.csv")


if __name__ == "__main__":
    main()