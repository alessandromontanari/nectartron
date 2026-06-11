import os
from dotenv import load_dotenv

from utils.webmail import authenticate, get_folders, search_messages, get_message
from utils.logging_config import setup_logger, log_and_print

logger = setup_logger(
    log_file_name="download_emails",
    logger_name="nectartron",
)

load_dotenv("./config/.env.local")

webmail_soap_url = os.getenv("WEBMAIL_SOAP_URL")
webmail_username = os.getenv("WEBMAIL_USERNAME")
webmail_password = os.getenv("WEBMAIL_PASSWORD")



def main():
    # Authenticate
    auth_token = authenticate(
        soap_url=webmail_soap_url,
        username=webmail_username,
        password=webmail_password
    )
    log_and_print(logger, "✓ Authenticated successfully.")

    # List folders
    folders = get_folders(
        soap_url=webmail_soap_url,
        auth_token=auth_token,
        traverse_subfolders=True
    )
    log_and_print(logger, "✓ Folders correctly extracted.")

    # Example: Extract emails from a specific folder (Inbox)
    target_folder = "/Inbox/NectarCAM"
    log_and_print(logger, f"Extracting messages from: {target_folder}")

    message_ids = search_messages(
        soap_url=webmail_soap_url,
        auth_token=auth_token,
        folder_path=target_folder,
        limit=5
    )
    log_and_print(logger, f"✓ Found {len(message_ids)} messages")

    # Retrieve and display the messages
    for msg_id in message_ids:
        message = get_message(
            soap_url=webmail_soap_url,
            auth_token=auth_token,
            message_id=msg_id
        )
        email_subject = message.get('subject', 'N/A')
        if "Logbook entry" in email_subject:
            sent_by = f"From: {message.get('from', 'N/A')}"
            date = f"From: {message.get('from', 'N/A')}"
            subject = f"Subject: {email_subject}"
            body = message.get('body', '')

            log_and_print(logger, f"--- Message ID: {msg_id} ---")
            log_and_print(logger, sent_by)
            log_and_print(logger, date)
            log_and_print(logger, subject)
            


if __name__ == "__main__":
    main()