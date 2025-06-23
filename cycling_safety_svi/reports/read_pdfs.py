import os
from pypdf import PdfReader
import sys

def read_pdf(file_path):
    """Reads a PDF file and returns its text content."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        return f"Error reading {file_path}: {e}"

def main():
    """Main function to read specified PDF files and save content to text files."""
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Define output directory for text files
    output_dir = os.path.join(os.path.dirname(__file__), 'pdf_texts')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Text files will be saved in: {output_dir}")

    pdf_files = [
        "references/route_choice/1-s2.0-S1369847823002358-main.pdf",
        "references/route_choice/1-s2.0-S1361920921000651-main.pdf",
        "references/route_choice/1-s2.0-S1361920922002917-main.pdf",
        "references/Scientific_Paper.pdf",
        "references/ssrn-4886041 (2).pdf",
    ]

    for pdf_path in pdf_files:
        full_path = os.path.join(project_root, pdf_path)
        
        # Create a valid filename for the output text file
        txt_filename = os.path.splitext(os.path.basename(pdf_path))[0].replace(' ', '_') + '.txt'
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        print(f"--- Processing: {pdf_path} ---")
        if os.path.exists(full_path):
            content = read_pdf(full_path)
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Content saved to: {txt_filepath}")
        else:
            print(f"File not found: {full_path}")
        print(f"--- Finished: {pdf_path} ---\n")

if __name__ == "__main__":
    try:
        import pypdf
    except ImportError:
        print("The 'pypdf' library is required. Please install it by running: pip install pypdf")
        sys.exit(1)
    main() 