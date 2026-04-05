import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI

app = FastAPI()

def scrape_linkedin(url):
    # 1. You MUST use 'headers'. Without these, LinkedIn returns a 403 Forbidden error.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # 2. Send the request
    response = requests.get(url, headers=headers)
    print(response)    
    # if response.status_code != 200:
    #     return f"Failed to load page. Status Code: {response.status_code}"

    # 3. Parse with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # 4. LinkedIn uses specific metadata tags (OpenGraph) for the post preview
    # This is often the most reliable way to get text without JavaScript.
    meta_tag = soup.find("meta", property="og:description")
    
    if meta_tag:
        return meta_tag["content"]
    
    # Fallback: Try to find the common article body tag
    body_tag = soup.find("article") or soup.find("div", class_="share-full-text")
    if body_tag:
        return body_tag.get_text(strip=True)

    return "Text not found. The post might be private or require JavaScript to render."

# # Usage
# url = "https://www.linkedin.com/posts/wafaa-al-harbi-b935571bb_proptech-digitaltransformation-aepaesaesaevaewaesabraepaesaezaeqaetaey-share-7446257088823070720-1NYj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAETjpGUBdUq8JXs2BJ3LXaG8My4yAhbYQrI"

@app.get("/")
def read_root(url: str):
    return scrape_linkedin(url)
