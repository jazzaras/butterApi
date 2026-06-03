import os
import logging
import asyncio
import json
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("butter-api")

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Butter & Dough API",
    description="A FastAPI backend leveraging Google Gemini API. Features 'Butter' (concise Arabic summaries of LinkedIn posts) and 'Dough' (professional detailed Arabic post expansion).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Pydantic Schemas for API Requests and Responses
# ----------------------------------------------------

class ButterRequest(BaseModel):
    linkedin_url: str = Field(
        ...,
        description="The full URL of the LinkedIn post to scrape and summarize.",
        examples=["https://www.linkedin.com/posts/example-user_example-post-id"]
    )

class ButterResponse(BaseModel):
    success: bool = Field(True, description="Indicates if the request was handled successfully.")
    feature: str = Field("butter", description="The feature name associated with this endpoint.")
    output: list[str] = Field(
        ...,
        description="A list of concise Arabic bullet points summarizing the key ideas of the LinkedIn post."
    )

class DoughRequest(BaseModel):
    text: str = Field(
        ...,
        description="A short text, idea, note, or rough thought to expand.",
        examples=["AI is changing software development"]
    )

class DoughResponse(BaseModel):
    success: bool = Field(True, description="Indicates if the request was handled successfully.")
    feature: str = Field("dough", description="The feature name associated with this endpoint.")
    output: str = Field(
        ...,
        description="The expanded, detailed, and engaging professional content in Arabic, suitable for LinkedIn."
    )

# ----------------------------------------------------
# Gemini LLM Response Schemas (for Structured Output)
# ----------------------------------------------------

class ButterSummaryLLM(BaseModel):
    bullet_points: list[str] = Field(
        ...,
        description="A list of up to 5 concise bullet points in clean, simple Arabic summarizing the post."
    )

class DoughLLM(BaseModel):
    expanded_content: str = Field(
        ...,
        description="The expanded professional content in Arabic."
    )

# ----------------------------------------------------
# Helper Utilities
# ----------------------------------------------------

def get_gemini_client() -> genai.Client:
    """
    Initializes and returns the Gemini API client using the environment key.
    Raises HTTPException if the key is not set or is the placeholder.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY environment variable is not set or has placeholder value.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API Key is not configured. Please set the GEMINI_API_KEY in your environment or .env file."
        )
    return genai.Client(api_key=api_key)


def _scrape_linkedin_sync(url: str) -> str:
    """
    Synchronously scrapes a LinkedIn post URL to extract its main content.
    Uses headers to prevent 403 blocks and parses OpenGraph/fallback tags.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        logger.info(f"Fetching LinkedIn URL: {url}")
        response = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        logger.error(f"Network error while fetching LinkedIn URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch LinkedIn URL due to a network error: {str(e)}"
        )

    if response.status_code != 200:
        logger.error(f"Failed to fetch LinkedIn URL. Status code: {response.status_code}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load LinkedIn page. HTTP Status Code: {response.status_code}"
        )

    soup = BeautifulSoup(response.content, "html.parser")

    # 1. Primary extractor: LinkedIn OpenGraph description (highly reliable without JS)
    meta_tag = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "twitter:description"})
    if meta_tag and meta_tag.get("content"):
        content = meta_tag["content"].strip()
        # Clean up common prepends like "Post by User..." if present
        if content:
            logger.info("Content extracted successfully using og/twitter description meta tag.")
            return content

    # 2. First Fallback: Article body or common post sharing container tags
    body_tag = soup.find("article") or soup.find("div", class_="share-full-text") or soup.find("div", class_="attributed-text-segment")
    if body_tag:
        content = body_tag.get_text(strip=True)
        if content:
            logger.info("Content extracted successfully using article or div tags.")
            return content

    # 3. Second Fallback: Generic page description
    meta_desc = soup.find("meta", name="description")
    if meta_desc and meta_desc.get("content"):
        content = meta_desc["content"].strip()
        if content:
            logger.info("Content extracted successfully using standard description meta tag.")
            return content

    # If all extraction methods fail
    logger.warning("Could not extract any content from the provided LinkedIn URL.")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Text content could not be extracted from the LinkedIn URL. The post may be private, require login, or contain unsupported formatting."
    )


async def scrape_linkedin(url: str) -> str:
    """
    Asynchronous wrapper for scraping to keep the FastAPI event loop unblocked.
    """
    return await asyncio.to_thread(_scrape_linkedin_sync, url)

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------

@app.get("/", summary="Root Health Check & Welcome")
def read_root():
    """
    Welcome endpoint returning API status and details.
    """
    return {
        "app": "Butter & Dough API",
        "status": "online",
        "docs": "/docs",
        "features": {
            "butter": "POST /butter - Summarize LinkedIn posts in concise Arabic bullet points",
            "dough": "POST /dough - Expand brief ideas into professional Arabic LinkedIn posts"
        }
    }


@app.post(
    "/butter",
    response_model=ButterResponse,
    summary="Feature 1: Butter",
    description="Accepts a LinkedIn post URL, extracts its content, and generates a concise Arabic summary with up to 5 high-impact bullet points."
)
async def butter_endpoint(request: ButterRequest):
    # Validate the URL format minimally
    url = request.linkedin_url.strip()
    if not url.startswith(("http://", "https://")) or "linkedin.com" not in url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL. Please provide a valid LinkedIn URL containing 'linkedin.com'."
        )

    # 1. Scrape the LinkedIn post
    post_content = await scrape_linkedin(url)
    logger.info(f"Successfully scraped post content length: {len(post_content)} characters.")

    # 2. Initialize Gemini Client
    client = get_gemini_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # 3. Request summary from Gemini with Pydantic structured output validation
    prompt = f"""
You are Butter.

Your job is not to summarize.
Your job is to extract what the person is REALLY saying.

Given a LinkedIn post, ignore:
- corporate language
- motivational language
- emotional language
- gratitude paragraphs
- networking language
- buzzwords
- hashtags

Return the post as if a friend were explaining it in one short Arabic sentence.

Rules:
- Be funny and slightly sarcastic.
- Do not be mean or insulting.
- Use Saudi Arabic accent.
- Maximum 10 words.
- Focus only on the actual achievement, event, announcement, or message.
- Remove all fluff.
- No introductions.
- No bullet points.
- No quotation marks.
- Return only the butter.

Examples:

Post: "I'm excited to share that I completed the AWS Cloud Practitioner certification."
Butter:
طلع شهادة AWS.

Post: "Happy to announce that I've joined Aramco as a data analyst."
Butter:
توظف في أرامكو.

Post: "Honored to be selected as a speaker at LEAP 2026."
Butter:
طلع متحدث في ليب.

Post: "After months of hard work, our team launched the new platform."
Butter:
نزلوا المنصة أخيراً.

Post: "Thankful for this amazing internship experience at STC."
Butter:
خلص تدريب في stc.

LinkedIn Post:
{post_content}
"""

    try:
        logger.info(f"Sending prompt to Gemini model '{model_name}' for Butter summary...")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ButterSummaryLLM,
                system_instruction="You are Butter, an expert AI assistant that summarizes long LinkedIn posts into high-impact, bite-sized bullet points in clean, simple Arabic."
            )
        )
        
        # 4. Parse the structured output
        if not response.text:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response received from the Gemini API."
            )

        data = json.loads(response.text)
        points = data.get("bullet_points", [])

        # Ensure bullet points are returned properly and limit to a max of 5 points
        if not isinstance(points, list):
            points = [str(points)]
        points = [p.strip() for p in points if p.strip()][:5]

        if not points:
            raise ValueError("No bullet points generated by the model.")

        logger.info("Successfully generated Butter summary.")
        return ButterResponse(
            success=True,
            feature="butter",
            output=points
        )

    except APIError as e:
        logger.error(f"Gemini API Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error occurred: {str(e)}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini structured JSON output: {e}. Raw text: {response.text}")
        # Fallback to simple line parsing
        lines = [line.strip("- *• ").strip() for line in response.text.split("\n") if line.strip()]
        fallback_points = [l for l in lines if l][:5]
        return ButterResponse(
            success=True,
            feature="butter",
            output=fallback_points if fallback_points else ["حدث خطأ أثناء معالجة الملخص."]
        )
    except Exception as e:
        logger.error(f"Error processing summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the summary: {str(e)}"
        )


@app.post(
    "/dough",
    response_model=DoughResponse,
    summary="Feature 2: Dough",
    description="Accepts a short text, idea, or rough thought, and expands it into detailed, engaging, and professional Arabic content suitable for a LinkedIn post."
)
async def dough_endpoint(request: DoughRequest):
    input_text = request.text.strip()
    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty."
        )

    # 1. Initialize Gemini Client
    client = get_gemini_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # 2. Request expansion from Gemini with Pydantic structured output validation
    prompt = f"""
You are Dough.

Your job is to take a simple event and turn it into an absurdly overdramatic LinkedIn post.

The result should sound like a stereotypical LinkedIn influencer who discovers life-changing leadership lessons in ordinary daily activities.

Rules:
- Write in English.
- use line breakes and good indentation 
- Be funny.
- Be intentionally exaggerated.
- Add fake lessons about leadership, growth, resilience, innovation, mindset, networking, or success.
- Turn small events into major life experiences.
- Include unnecessary gratitude.
- Include at least 3 LinkedIn-style hashtags.
- Sound sincere, not obviously joking.
- The more mundane the input, the more dramatic the output should be.
- Length: 1-3 paragraphs.

Examples:

Input:
اشتريت حليب

Output:
اليوم تذكرت أن النجاح مو دايم يكون في القرارات الكبيرة، أحيانًا يبدأ من رف بسيط في البقالة.

وأنا أختار الحليب، استوعبت أن القيادة الحقيقية هي القدرة على اتخاذ القرار حتى وسط عشرات الخيارات. كل منتج كان يمثل مسار مختلف، لكن التردد ما يصنع نتائج.

ممتن لهذه التجربة البسيطة اللي ذكرتني أن النمو يبدأ بخطوة صغيرة.

#Leadership #GrowthMindset #DecisionMaking #Success

Input:
شحنت جوالي

Output:
في عالم سريع التغير، من السهل ننسى أهمية إعادة شحن أنفسنا قبل إعادة شحن أعمالنا.

وأنا أوصل الشاحن اليوم، استوعبت أن الأداء العالي يحتاج توقفات قصيرة. حتى أقوى الأجهزة تحتاج مصدر طاقة، وكذلك القادة.

درس بسيط لكنه عميق.

#Leadership #Productivity #Mindset #SelfDevelopment

Input:
نمت 8 ساعات

Output:
بعد فترة طويلة من مطاردة الإنجازات، تعلمت اليوم أن الراحة ليست عدو النجاح.

ثمان ساعات من النوم أعادت لي وضوح التفكير والطاقة والتركيز. أحيانًا أفضل استثمار تسويه مو في مشروع جديد، بل في نفسك.

الاستدامة تتفوق على الاندفاع.

#GrowthMindset #Leadership #Wellbeing #Success

Event:
{input_text}
"""

    try:
        logger.info(f"Sending prompt to Gemini model '{model_name}' for Dough expansion...")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DoughLLM,
                system_instruction="You are Dough, an expert LinkedIn content writer. Your job is to take brief, rough notes and expand them into engaging, professional, and detailed posts in clean and elegant Arabic."
            )
        )

        # 3. Parse structured output
        if not response.text:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response received from the Gemini API."
            )

        data = json.loads(response.text)
        expanded_content = data.get("expanded_content", "").strip()

        if not expanded_content:
            raise ValueError("No expanded content generated by the model.")

        logger.info("Successfully generated Dough expansion.")
        return DoughResponse(
            success=True,
            feature="dough",
            output=expanded_content
        )

    except APIError as e:
        logger.error(f"Gemini API Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error occurred: {str(e)}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini structured JSON output: {e}. Raw text: {response.text}")
        # Fallback to direct raw output
        fallback_content = response.text.replace("```json", "").replace("```", "").strip()
        return DoughResponse(
            success=True,
            feature="dough",
            output=fallback_content if fallback_content else "حدث خطأ أثناء توسيع المحتوى."
        )
    except Exception as e:
        logger.error(f"Error processing expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while expanding the content: {str(e)}"
        )
