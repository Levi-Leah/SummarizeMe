import argparse
import os
import concurrent.futures
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import logging

def main(input_files):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_file, input_files)

    for result in results:
        if result is not None:
            for filepath, summary in result.items():
                print(f"Filepath: {filepath}\nSummary: {summary}\n\n")

def abstractive_summarization(file):
    # Load pre-trained model and tokenizer only once
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum", model_max_length=1024)

    # Tokenize and generate summary
    with open(file, "r") as f:
        text = f.read()

        inputs = tokenizer("summarize: " + text, return_tensors="pt", padding="longest", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], min_length=150, max_length=300, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {file: summary}


def process_file(input_file_path):
    try:
        with open(input_file_path, "r", encoding="utf-8") as file:
            input_text = file.read()

        # Check if the file is empty
        if not input_text.strip():
            print(f"Error: File '{input_file_path}' is empty.")
            return None

    except Exception as e:
        print(f"Error reading file '{input_file_path}': {e}")
        return None

    summary_dict = abstractive_summarization(input_file_path)
    return summary_dict


input_files = ['/home/levi/rhel-8-docs/rhel-9/modules/performance/ref_vdo-thread-types.adoc', '/home/levi/rhel-8-docs/rhel-9/modules/performance/proc_configuring-the-cpu-affinity.adoc', '/home/levi/rhel-8-docs/rhel-9/modules/performance/proc_speeding-up-discard-operations.adoc']


main(input_files)