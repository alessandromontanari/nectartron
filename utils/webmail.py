import logging
import xml.etree.ElementTree as ET
import requests
import re

from html.parser import HTMLParser

from utils.logging_config import log_and_print

logger = logging.getLogger(__name__)


def authenticate(soap_url, username, password):
    """Authenticate to the webmail service provider

    Parameters
    ----------
    soap_url : str
        Path to the SOAP URL for the webmail service provider
    username : str
        Username for authentication
    password : str
        Password for authentication

    Returns 
    -------
    str
        Authentication token if successfull
    """
    auth_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>
    <context xmlns="urn:zimbra">
      <format type="xml"/>
    </context>
  </soap:Header>
  <soap:Body>
    <AuthRequest xmlns="urn:zimbraAccount">
      <account by="name">{username}</account>
      <password>{password}</password>
    </AuthRequest>
  </soap:Body>
</soap:Envelope>"""

    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "Accept": "text/xml"
    }

    log_and_print(logger, "Authenticating...")

    response = requests.post(
        soap_url, 
        data=auth_request, 
        headers=headers
    )

    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)
            auth_token = root.find(".//{urn:zimbraAccount}authToken")
            if auth_token is not None:
                log_and_print(logger, "✓ Successfull authentication...")
                return auth_token.text
            else:
                log_and_print(logger, f"✗ Failed... Auth token not found in response: {response.text}", level="error")
                raise Exception(f"Auth token not found in response: {response.text}")
        except ET.ParseError as e:
            log_and_print(logger, f"✗ Failed... Failed to parse XML response: {e}\nResponse: {response.text}", level="error")
            raise Exception(f"Failed to parse XML response: {e}\nResponse: {response.text}")
    else:
        log_and_print(logger, f"✗ Failed... Authentication failed: {response.text}", level="error")
        raise Exception(f"Authentication failed: {response.text}")


def get_folders(soap_url, auth_token, traverse_subfolders=True):
    """List all folders/directories in the mail account

    Parameters
    ----------
    soap_url : str
        Path to the SOAP URL for the webmail service provider
    auth_token : str
        Authentication token from authenticate()
    traverse_subfolders : bool
        Whether to recursively traverse subfolders

    Returns
    -------
    list
        List of folder names (strings)
    """
    folder_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>
    <context xmlns="urn:zimbra">
      <authToken>{auth_token}</authToken>
      <format type="xml"/>
    </context>
  </soap:Header>
  <soap:Body>
    <GetFolderRequest xmlns="urn:zimbraMail" traverseSubfolders="{str(traverse_subfolders).lower()}">
      <folder path="/"/>
    </GetFolderRequest>
  </soap:Body>
</soap:Envelope>"""

    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "Accept": "text/xml"
    }

    log_and_print(logger, "Requesting the folders' list...")

    response = requests.post(
        soap_url,
        data=folder_request,
        headers=headers
    )

    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)
            folders = []
            for folder in root.findall(".//{urn:zimbraMail}folder"):
                folder_name = folder.get("name")
                folder_path = folder.get("path")
                if folder_name:
                    folders.append(folder_name)
                elif folder_path:
                    folders.append(folder_path)
            log_and_print(logger, "✓ Folders successfully obtained...")
            return folders
        except ET.ParseError as e:
            log_and_print(logger, f"Failed to parse XML response: {e}\nResponse: {response.text}", level="error")
            raise Exception(f"Failed to parse XML response: {e}\nResponse: {response.text}")
    else:
        log_and_print(logger, f"Failed to get folders: {response.text}", level="error")
        raise Exception(f"Failed to get folders: {response.text}")


def search_messages(soap_url, auth_token, folder_path="/Inbox", limit=None):
    """Search for messages in a specific folder

    Parameters
    ----------
    soap_url : str
        Path to the SOAP URL
    auth_token : str
        Authentication token
    folder_path : str
        Path to the folder (e.g., "/Inbox", "/Sent", "/Drafts")
    limit : int or None
        Maximum number of messages to return. If None, no limit is applied
        (server default/maximum will be used)

    Returns
    -------
    list
        List of message IDs (strings)
    """
    if limit is not None:
        search_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>
    <context xmlns="urn:zimbra">
      <authToken>{auth_token}</authToken>
      <format type="xml"/>
    </context>
  </soap:Header>
  <soap:Body>
    <SearchRequest xmlns="urn:zimbraMail" limit="{limit}">
      <query>in:{folder_path}</query>
    </SearchRequest>
  </soap:Body>
</soap:Envelope>"""
    else:
        search_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>
    <context xmlns="urn:zimbra">
      <authToken>{auth_token}</authToken>
      <format type="xml"/>
    </context>
  </soap:Header>
  <soap:Body>
    <SearchRequest xmlns="urn:zimbraMail">
      <query>in:{folder_path}</query>
    </SearchRequest>
  </soap:Body>
</soap:Envelope>"""

    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "Accept": "text/xml"
    }

    log_and_print(logger, "Searching messages...")

    response = requests.post(
        soap_url,
        data=search_request,
        headers=headers
    )

    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)
            message_ids = []
            for msg in root.findall(".//{urn:zimbraMail}m"):
                msg_id = msg.get("id")
                if msg_id:
                    message_ids.append(msg_id)
            log_and_print(logger, "✓ Messages found...")
            return message_ids
        except ET.ParseError as e:
            log_and_print(logger, f"Failed to parse XML response: {e}\nResponse: {response.text}", level="error")
            raise Exception(f"Failed to parse XML response: {e}\nResponse: {response.text}")
    else:
        log_and_print(logger, f"Search failed: {response.text}", level="error")
        raise Exception(f"Search failed: {response.text}")


def clean_message_body(body):
    """Clean the message body by removing the email footer.

    Parameters
    ----------
    body : str
        HTML string containing the whole body of the message

    Returns
    -------
    str
        The text content cleaned of the email footer
    """
    
    body = body.split('<center><a href="https://midas.psi.ch/elog/"')[0]
    
    return body


def get_message(soap_url, auth_token, message_id):
    """Get the content of a specific message

    Parameters
    ----------
    soap_url : str
        Path to the SOAP URL
    auth_token : str
        Authentication token
    message_id : str
        ID of the message to retrieve

    Returns
    -------
    dict
        Dictionary containing message details (subject, body, etc.)
    """
    get_msg_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Header>
    <context xmlns="urn:zimbra">
      <authToken>{auth_token}</authToken>
      <format type="xml"/>
    </context>
  </soap:Header>
  <soap:Body>
    <GetMsgRequest xmlns="urn:zimbraMail">
      <m id="{message_id}"/>
    </GetMsgRequest>
  </soap:Body>
</soap:Envelope>"""

    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "Accept": "text/xml"
    }

    log_and_print(logger, "Getting the content of chosen message...")

    response = requests.post(
        soap_url,
        data=get_msg_request,
        headers=headers
    )

    if response.status_code == 200:
        try:
            root = ET.fromstring(response.text)
            msg = root.find(".//{urn:zimbraMail}m")
            if msg is not None:
                result = {
                    "id": msg.get("id"),
                    "subject": msg.find(".//{urn:zimbraMail}su").text if msg.find(".//{urn:zimbraMail}su") is not None else None,
                    "from": msg.find(".//{urn:zimbraMail}e").get("a") if msg.find(".//{urn:zimbraMail}e") is not None else None,
                    "from_name": msg.find(".//{urn:zimbraMail}e").get("p") if msg.find(".//{urn:zimbraMail}e") is not None else None,
                    "date": msg.find(".//{urn:zimbraMail}d").text if msg.find(".//{urn:zimbraMail}d") is not None else None,
                }

                # Extract body from message parts - Zimbra stores body in <content> tags
                body = ""
                for content_elem in msg.findall(".//{urn:zimbraMail}content"):
                    if content_elem.text:
                        body += content_elem.text + "\n"

                if not body:
                    for content_elem in msg.findall(".//content"):
                        if content_elem.text:
                            body += content_elem.text + "\n"

                # result["body"] = body.strip() if body else None
                body = body.strip() if body else None

                result["body"] = clean_message_body(body=body) if body else None            

                log_and_print(logger, "✓ Message extracted")

                return result
            else:
                log_and_print(logger, f"Message not found in response: {response.text}", level="error")
                raise Exception(f"Message not found in response: {response.text}")
        except ET.ParseError as e:
            log_and_print(logger, f"Failed to parse XML response: {e}\nResponse: {response.text}", level="error")
            raise Exception(f"Failed to parse XML response: {e}\nResponse: {response.text}")
    else:
        log_and_print(logger, f"Failed to get message: {response.text}", level="error")
        raise Exception(f"Failed to get message: {response.text}")
    

def extract_hidden_inputs(html_text):
    """Extract key-value pairs from hidden HTML input fields.

    Parameters
    ----------
    html_text : str
        HTML string containing input elements

    Returns
    -------
    dict
        Dictionary with name: value pairs from hidden inputs
    """
    pattern = r'<input\s+type="hidden"\s+name="([^"]+)"\s+value="([^"]*)"\s*/>'
    matches = re.findall(pattern, html_text)
    return {name: value for name, value in matches}


class HiddenInputParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden_inputs = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "input" and attrs_dict.get("type") == "hidden":
            name = attrs_dict.get("name", "")
            value = attrs_dict.get("value", "")
            if name:
                self.hidden_inputs[name] = value


def extract_hidden_inputs_parser(html_text):
    """Extract key-value pairs from hidden HTML input fields using HTMLParser.

    Parameters
    ----------
    html_text : str
        HTML string containing input elements

    Returns
    -------
    dict
        Dictionary with name: value pairs from hidden inputs
    """
    parser = HiddenInputParser()
    parser.feed(html_text)
    return parser.hidden_inputs


def extract_messageframe_content(html_text):
    """Extract the content from the messageframe td element.

    Parameters
    ----------
    html_text : str
        HTML string containing the messageframe

    Returns
    -------
    str
        The text content inside the messageframe element
    """
    pattern = r'<td[^>]*class="[^"]*messageframe[^"]*"[^>]*>(.*?)</td>'
    match = re.search(pattern, html_text, re.DOTALL)

    if match:
        inner_html = match.group(1)
        clean_text = re.sub(r'<[^>]+>', ' ', inner_html)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    return None


class MessageFrameParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_messageframe = False
        self.content = []
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "td" and attrs_dict.get("class", "").find("messageframe") != -1:
            self.in_messageframe = True
            self.current_data = []

    def handle_endtag(self, tag):
        if tag == "td" and self.in_messageframe:
            self.in_messageframe = False
            self.content = self.current_data.copy()

    def handle_data(self, data):
        if self.in_messageframe:
            self.current_data.append(data)

    def get_content(self):
        return " ".join(self.content).strip()


def extract_messageframe_content_parser(html_text):
    """Extract the content from the messageframe td element using HTMLParser.

    Parameters
    ----------
    html_text : str
        HTML string containing the messageframe

    Returns
    -------
    str
        The text content inside the messageframe element
    """
    parser = MessageFrameParser()
    parser.feed(html_text)
    return parser.get_content()


def extract_entry_time(html_text):
    """Extract the content from the messageframe td element using HTMLParser.

    Parameters
    ----------
    html_text : str
        HTML string containing the entry time

    Returns
    -------
    str
        The entry time string, or None if not found
    """
    # Pattern to match "Entry time: <b>TIMESTAMP</b>"
    pattern = r'Entry time:\s<b>(.*?)</b>'
    match = re.search(pattern, html_text)

    if match:
        return match.group(1).strip()
    return None