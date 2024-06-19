import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

def chat_with_llama(pipeline):
  print("Welcome! Chat with Llama-3. Type 'quit' to exit.")
  while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
      print("Goodbye!")
      break

    try:
      response = pipeline(user_input, model_kwargs={"torch_dtype": torch.bfloat16})[0]["generated_text"]
      print(f"Bot-3: {response}")
    except (transformers.errors.ModelError, RuntimeError) as e:
      print("error occurred")
      print("Attempting to recover...")

if __name__ == "__main__":
  pipeline = transformers.pipeline(
      "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
  )
  chat_with_llama(pipeline)