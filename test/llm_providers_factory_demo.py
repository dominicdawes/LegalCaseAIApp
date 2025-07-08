# test/test_llm_providers.py

import os
import json
import traceback
from dotenv import load_dotenv

from utils.llm_factory import LLMFactory

load_dotenv()

def main():
    """
    Iterate through each provider/model combo, send a simple "Hello, world!" prompt,
    and print out either the LLM's response or the error encountered.
    """
    # Define which providers and models to test. 
    # Replace model names below with whichever specific versions you want to “ping.”
    tests = {
        "openai":       ["gpt-3.5-turbo"],             # e.g. "gpt-4o-mini" or "gpt-3.5-turbo"
        "deepseek":     ["deepseek-chat"],               # e.g. "deepseek-chat", "deepseek-reasoner"
        "anthropic":    ["claude-3-7-sonnet-latest"],    # e.g. "claude-sonnet-4-20250514", "claude-3-5-haiku-latest"
        "gemini":       ["gemini-2.5-flash-preview-05-20"],  # e.g. "gemini-2.5-flash", "gemini-2.5-pro"
        # "qwen":       ["qwen-2-5-pro"],               # uncomment once QWEN credentials are set
    }

    prompt_text = "Hello, world!"

    for provider, model_list in tests.items():
        for model_name in model_list:
            print("=" * 60)
            print(f"Testing provider={provider!r}, model_name={model_name!r}")
            try:
                # Instantiate the client via your factory
                llm_client = LLMFactory.get_client(
                    provider=provider,
                    model_name=model_name,
                    temperature=0.0,       # use 0.0 temperature for deterministic “ping”
                    streaming=False,
                    callback_manager=None
                )

                # Send a very simple prompt
                response = llm_client.chat(prompt_text)
                print("✅ Success:")
                print(response.strip().replace("\n", " "))
            except Exception as e:
                print("❌ Failed:")
                traceback.print_exc()

    print("=" * 60)
    print("All tests completed.")

if __name__ == "__main__":
    main()
