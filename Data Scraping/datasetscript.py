import requests
from bs4 import BeautifulSoup

def fetch_page_text(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    soup = BeautifulSoup(response.text, "html.parser")

    structured_text = []
    
    for tag in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]:
        for element in soup.find_all(tag):
            structured_text.append(f"{tag.upper()}: {element.get_text(strip=True)}")
    
    return "\n".join(structured_text)

url = "https://sf.funcheap.com/"
text = fetch_page_text(url)

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("Text extracted and saved to output.txt")
