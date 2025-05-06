import os
import json
import openai  # Ensure openai is installed
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Remove the SSL_CERT_FILE variable from the environment if it exists.
os.environ.pop("SSL_CERT_FILE", None)

# Configuration: Fetch the DeepSeek API key and set the Base URL.
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API')

if not DEEPSEEK_API_KEY:
    print("API key not found. Please check your .env file.")
else:
    print("API key loaded successfully.")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Initialize the OpenAI client to work with DeepSeek.
client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url=DEEPSEEK_BASE_URL
)

def generate_flashcards(raw_text: str, model: str = "deepseek-chat") -> list:
    """
    Generates flashcards from the provided legal text using DeepSeek's chat completions API.
    
    This function crafts a prompt instructing the model to analyze the legal text and generate flashcards.
    Each flashcard will have a key legal term as 'Front' and a very succinct definition as 'Back'.
    
    Parameters:
        raw_text (str): The legal text to analyze.
        model (str): The DeepSeek model to use (default: "deepseek-chat").
    
    Returns:
        list: A list of flashcard dictionaries in the format 
              [{'Front': 'Key Term', 'Back': 'Definition'}, ...].
              If the response is empty or cannot be parsed as JSON, returns the raw response content.
    """
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that generates flashcards for legal text."
        },
        {
            "role": "user", 
            "content": f"Analyze the following legal text and generate flashcards. Each flashcard should include a key legal term as 'Front' and a very succinct definition as 'Back':\n\n{raw_text}"
        }
    ]
    
    # Send the prompt to DeepSeek's chat completions API.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    
    # Extract the message content from the response.
    message_content = response.choices[0].message.content
    
    # Check if the response content is empty.
    if not message_content.strip():
        return "No response from API."
    
    # Try to parse the response as JSON. If that fails, return the raw message.
    try:
        flashcards = json.loads(message_content)
    except json.JSONDecodeError:
        flashcards = message_content
    
    return flashcards

if __name__ == "__main__":
    # legal_text = input("Please enter legal text: \n")

    legal_text='Brown v. Board of Education (1954) Brown v. Board of Education of Topeka, Opinion; May 17, 1954; Records of the Supreme Court of the United States; Record Group 267; National Archives. \
    In this milestone decision, the Supreme Court ruled that separating children in public schools on the basis of race was unconstitutional. It signaled the end of legalized racial segregation in the \
    schools of the United States, overruling the "separate but equal" principle set forth in the 1896 Plessy v. Ferguson case. \
    On May 17, 1954, U.S. Supreme Court Justice Earl Warren delivered the unanimous ruling in the landmark civil rights case Brown v. Board of Education of Topeka, Kansas. \
    State-sanctioned segregation of public schools was a violation of the 14th amendment and was therefore unconstitutional. This historic decision marked the end of the  \
    "separate but equal" precedent set by the Supreme Court nearly 60 years earlier in Plessy v. Ferguson and served as a catalyst for the expanding civil rights movement during \
    the decade of the 1950s. Arguments were to be heard during the next term to determine just how the ruling would be imposed. Just over one year later, on May 31, 1955, Warren \
    read the Courts unanimous decision, now referred to as Brown II, instructing the states to begin desegregation plans "with all deliberate speed. \
    Despite two unanimous decisions and careful, if vague, wording, there was considerable resistance to the Supreme Courts ruling in Brown v. Board of Education. \
    In addition to the obvious disapproving segregationists were some constitutional scholars who felt that the decision went against legal tradition by relying heavily on data \
    supplied by social scientists rather than precedent or established law. Supporters of judicial restraint believed the Court had overstepped its constitutional powers by essentially writing new law. \
    However, minority groups and members of the civil rights movement were buoyed by the Brown decision even without specific directions for implementation. Proponents of judicial \
    activism believed the Supreme Court had appropriately used its position to adapt the basis of the Constitution to address new problems in new times. The Warren Court stayed this \
    course for the next 15 years, deciding cases that significantly affected not only race relations, but also the administration of criminal justice, the operation of the political \
    process, and the separation of church and state.'

    # legal_text='Brown v. Board of Education (1954) Brown v. Board of Education of Topeka, Opinion; May 17, 1954; Records of the Supreme Court of the United States; Record Group 267; National Archives. In this milestone decision, the Supreme Court ruled that separating children in public schools on the basis of race was unconstitutional. It signaled the end of legalized racial segregation in the schools of the United States, overruling the "separate but equal" principle set forth in the 1896 Plessy v. Ferguson case. On May 17, 1954, U.S. Supreme Court Justice Earl Warren delivered the unanimous ruling in the landmark civil rights case Brown v. Board of Education of Topeka, Kansas. State-sanctioned segregation of public schools was a violation of the 14th amendment and was therefore unconstitutional. This historic decision marked the end of the...' 
    print(legal_text)
    print("legal text parsed...")
    flashcards = generate_flashcards(legal_text)
    print("\nFlashcards generated by DeepSeek:")
    print(flashcards)
