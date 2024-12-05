import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GENAI_API_KEY"])

def translate_matlab_to_python(matlab_path) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    matlab_code = genai.upload_file(matlab_path, mime_type="text/plain")

    prompt = """Translate this MATLAB code to Python such that it produces identical results to the original code.
                The comments from the original MATLAB code should be adapted so they make sense in the context of the translated Python code.
                Do not use any packages beyond numpy, scipy, matplotlib, and pandas (only import these if they are needed).
                If the MATLAB code uses a function that is not defined in the code snippet, you can assume that it is already defined elsewhere.
                Please do this quietly, suppressing any output besides what I have requested.
        """
    response = model.generate_content([prompt, matlab_code])

    return response.text

def list_matlab_files(directory) -> list:
    matlab_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".m") and file != "release_notes.m":
                matlab_files.append(os.path.join(root, file))
    return matlab_files

# Example usage
directory = "C:\\Users\\djr33\\OneDrive\\Desktop\\Projects\\bctpy\\BCT-main"
matlab_files = list_matlab_files(directory)
output_dir = os.path.join(directory, "BCT-python")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for matlab_file in matlab_files:
    python_code = translate_matlab_to_python(matlab_file)
    python_code = python_code.replace("python", "").replace("`", "") # get rid of markdown formatting
    output_file = os.path.join(output_dir, os.path.basename(matlab_file).replace(".m", ".py"))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Translated from {os.path.basename(matlab_file)}\n")
        f.write(python_code + "\n")