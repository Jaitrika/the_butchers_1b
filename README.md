# the_butchers_1b
# PDF Section Intelligence Extractor

_A smart pipeline to extract, filter, and rank relevant sections from PDF documents based on persona-driven task queries. Harnesses modern NLP embedding models for flexible, domain-agnostic information retrieval._

## 📁 Project Structure

```
.
├── app.py          # Main application logic
├── parser.py       # PDF sectioning and heading extraction module
├── documents/      # Folder containing PDF files to process (please add input PDFs from your collection into this folder)
├── input.json      # Input configuration (contains PDF file names, persona, job to be done in proper format)
├── output.json     # Output results (generated)
├── Dockerfile      # Container setup for portable execution
└── README.md       # you are currently reading it :)

````

## ⭐ Overview

This project enables extraction of the most relevant content from PDF files, tailored to a **persona** and a **Job To Be Done (JTBD)**, as specified in `input.json`. Key features:

- **PDF Structure Analysis**: Parses headings, detects section levels, and clusters content beneath document headings.
- **Semantic Search & Filtering**: Finds and ranks sections most relevant to your task using advanced embedding models.
- **Persona-Driven Query**: Considers the role/persona and task to personalise relevance.
- **Constraint Parsing**: Supports nuanced inclusion/exclusion criteria in natural language queries (e.g., "gluten-free", "no sugar", "under 200 calories").

---

## Quick Start

### Option 1: Run via Docker (Recommended)

> If using **Windows Powershell**:

```bash
docker run --rm -it -v "${PWD}:/app" pdf-parser-app
````

Make sure you've built the image first (see Docker section below).

---

###  Option 2: Run Locally (Without Docker)

#### 1. Install Dependencies

```bash
pip install PyMuPDF sentence-transformers scikit-learn torch
```

> (You may need to install `torch` separately if not already present: `pip install torch`)

#### 2. Prepare Input Files

* **PDFs**: Place your `.pdf` files in the `documents/` folder.
* **input.json**: Provide metadata and query details:

```json
{
  "documents": [
    {"filename": "doc1.pdf"},
    {"filename": "doc2.pdf"}
  ],
  "persona": {"role": "Research scientist"},
  "job_to_be_done": {"task": "Identify the primary research findings on topic X, excluding preliminary studies"}
}
```

#### 3. Run the Pipeline

```bash
python app.py
```

#### 4. View Results

* Output is saved to `final_output.json`.
* Shows the most relevant sections (with titles, documents, page numbers, ranking, and full refined text).

---

## Docker Usage

### 1. Build the Docker Image

In the root project directory (where the Dockerfile is located), run:

```bash
docker build -t pdf-parser-app .
```

This sets up the container environment with all dependencies.

---

###  2. Run the App in a Container

#### For PowerShell or Unix-like shells:

```bash
docker run --rm -it -v "${PWD}:/app" pdf-parser-app
```

This mounts your current folder (containing `input.json`, `documents/`, etc.) into the container at `/app`.

---

### Output Location

* The output file `output.json` will be saved to your host directory.

## How It Works

### 1. PDF Section Parsing (**parser.py**)

* Extracts all text blocks from each PDF with formatting, position, and heading analysis.
* Classifies headings into hierarchical levels (e.g., section, subsection).
* Groups paragraphs under their detected headings.
* Outputs a structured list of sections with metadata (title, text, page number).

### 2. Query Interpretation & Semantic Search (**app.py**)

* Loads `input.json` to form a query using persona and JTBD.
* Uses `Sentence Transformers` to encode the query and all sections.
* Computes cosine similarity to shortlist top sections.
* Parses query for constraints like `"no sugar"`, `"gluten-free"`, etc.
* Applies inclusion/exclusion filters on matched sections.
* Re-ranks final candidates using a cross-encoder for precision.
* Saves the top results to `output.json`.

---

## Advanced Usage

* To process dozens of PDFs, just list them in `input.json`.
* Modify `top_k` in `app.py` to control how many results to return.
* You can enhance the filtering logic to support complex custom constraints.

---
