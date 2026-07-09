# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import re
from html.parser import HTMLParser


# ============================================================================
# Pre-compile regex patterns at module level
# ============================================================================
ENTRY_TIME_PATTERN = re.compile(r'Entry time:\s<b>(.*?)</b>')
MESSAGEFRAME_PATTERN = re.compile(
    r'<td[^>]*class="[^"]*messageframe[^"]*"[^>]*>(.*?)</td>',
    re.DOTALL
)
TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')


# ============================================================================
# CyHiddenInputParser - Cython-optimized HTML parser for hidden inputs
# Note: Must use 'class' not 'cdef class' because we inherit from HTMLParser (Python class)
# ============================================================================
class CyHiddenInputParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden_inputs = {}
    
    def handle_starttag(self, tag, attrs):
        # Only process input tags
        if tag != "input":
            return
            
        # Fast attribute lookup without dict creation
        name = ""
        value = ""
        is_hidden = False
        
        for attr in attrs:
            if attr[0] == "type" and attr[1] == "hidden":
                is_hidden = True
            elif attr[0] == "name":
                name = attr[1]
            elif attr[0] == "value":
                value = attr[1]
        
        if is_hidden and name:
            self.hidden_inputs[name] = value


# ============================================================================
# CyMessageFrameParser - Cython-optimized HTML parser for message frame content
# ============================================================================
class CyMessageFrameParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.content = []
        self.current_data = []
        self.in_messageframe = False
    
    def handle_starttag(self, tag, attrs):
        if tag == "td":
            for attr in attrs:
                if attr[0] == "class" and "messageframe" in attr[1]:
                    self.in_messageframe = True
                    self.current_data = []
                    break
    
    def handle_data(self, data):
        if self.in_messageframe:
            self.current_data.append(data)
    
    def handle_endtag(self, tag):
        if tag == "td" and self.in_messageframe:
            self.in_messageframe = False
            self.content.extend(self.current_data)
    
    def get_content(self):
        return " ".join(self.content).strip()


# ============================================================================
# Wrapper functions that match the original API from webmail.py
# ============================================================================

def cy_extract_hidden_inputs_parser(html_text):
    """Cython-optimized version of extract_hidden_inputs_parser"""
    parser = CyHiddenInputParser()
    parser.feed(html_text)
    return parser.hidden_inputs


def cy_extract_messageframe_content_parser(html_text):
    """Cython-optimized version of extract_messageframe_content_parser"""
    parser = CyMessageFrameParser()
    parser.feed(html_text)
    return parser.get_content()


def cy_extract_messageframe_content(html_text):
    """Cython-optimized regex-based version of extract_messageframe_content"""
    match = MESSAGEFRAME_PATTERN.search(html_text)
    if match:
        inner_html = match.group(1)
        clean_text = TAG_PATTERN.sub(' ', inner_html)
        clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
        return clean_text
    return None


def cy_extract_entry_time(html_text):
    """Cython-optimized version of extract_entry_time"""
    match = ENTRY_TIME_PATTERN.search(html_text)
    if match:
        return match.group(1).strip()
    return None
