# pip install torch transformers
# pip install sentencepiece
# pip install protobuf

import argparse
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main():
    
    parser = argparse.ArgumentParser(description="Abstractive Text Summarization Script")
    parser.add_argument("input_files", nargs='+', help="Paths to the input text files")

    args = parser.parse_args()
    input_files = args.input_files

    for input_file_path in input_files:
        # Check if the file exists
        if not os.path.exists(input_file_path):
            print(f"Error: File '{input_file_path}' does not exist.")
            continue

        # Read input text from the file
        try:
            with open(input_file_path, "r", encoding="utf-8") as file:
                input_text = file.read()

            # Check if the file is empty
            if not input_text.strip():
                print(f"Error: File '{input_file_path}' is empty.")
                continue
        
        # Check if the file is corrupt
        except Exception as e:
            print(f"Error reading file '{input_file_path}': {e}")
            continue
        
        summary = abstractive_summarization(input_file_path)
        
        print(summary)


def abstractive_summarization(file):
    
    # Load pre-trained model and tokenizer only once
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=1024, legacy=False)  # Set your preferred max_length
    
    # Tokenize and generate summary
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)  # Set your preferred max_length
        summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary


if __name__ == "__main__":
    main()