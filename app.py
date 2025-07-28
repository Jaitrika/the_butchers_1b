import fitz 
import os
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import torch
import parser
import re

def extract_sections_from_pdf(file_path):
    sections = parser.process_multiple_documents([file_path])
    return sections


def extract_from_multiple_pdfs(folder_path, filenames):
    all_sections = []

    for filename in filenames:
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            sections = extract_sections_from_pdf(file_path)
            all_sections.extend(sections)

    return all_sections

with open("input/input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

documents_info = input_data["documents"]
persona_description = input_data["persona"]["role"]
job_to_be_done = input_data["job_to_be_done"]["task"]
query = persona_description + " " + job_to_be_done

pdf_folder = "documents"
pdf_files = [doc["filename"] for doc in documents_info]

# ---- STEP 2: Extract Sections ----
# You must have this function already defined and working
all_content = extract_from_multiple_pdfs(pdf_folder, pdf_files)
print("Extracted:", len(all_content), "sections")
all_sections = [sec for sec in all_content if sec.get("text") and sec["text"].strip()]
texts = [sec["text"] for sec in all_sections]
print("Number of non-empty sections:", len(texts))

# ---- STEP 3: Sentence Embeddings ----

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",cache_folder="/root/.cache",local_files_only=True)
query_embedding = model.encode(query, convert_to_tensor=True)
section_embeddings = model.encode(texts, convert_to_tensor=True)
cosine_scores = util.cos_sim(query_embedding, section_embeddings)[0]

# Step 4A: Bi-encoder Top-K
cosine_top_k = 40  # or adjust based on dataset size
top_indices = torch.topk(cosine_scores, k=cosine_top_k).indices.tolist()

top_sections_raw = [all_sections[i] for i in top_indices]
top_texts_raw = [texts[i] for i in top_indices]
def parse_generic_query(query):
    """
    Generic query parser that extracts constraints from natural language.
    Works for any domain - food, technology, business, etc.
    """
    query_lower = query.lower()
    filters = {
        "exclude": [],
        "include": [], 
        "must_have": [],
        "cannot_have": [],
        "numeric": {},
        "special_requirements": []
    }
    
    # Pattern 1: Handle X-free patterns first (before other patterns)
    free_patterns = [
        r'\b(\w+)[-\s]free\b',  # matches "gluten-free", "dairy free", etc.
    ]
    
    for pattern in free_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            filters["exclude"].append(match.strip())
    
    # Pattern 2: Direct exclusions - "no X", "without X", "not X", "avoid X"
    exclude_patterns = [
        r'\b(?:no|not|non|never|avoid|without|exclude|excluding)\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)',
        r'\b(?:don\'t|dont)\s+(?:want|use|include)\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)',
        r'\b(?:free\s+from|free\s+of)\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)',
    ]
    
    for pattern in exclude_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            # Skip if it's part of an X-free phrase we already processed
            if not any(f"{match}-free" in query_lower or f"{match} free" in query_lower for _ in [1]):
                filters["exclude"].append(match.strip())
    
    
    include_patterns = [
        r'\b(?:with|contains|containing|has|having)\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)',
        r'\b(?:must\s+have|need|needs|require|requires|should\s+have)\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)',
        r'\b(?:only|exclusively)\s+(\w+(?:\s+\w+)*?)(?:\s|$|,|\.)',
    ]
    
    for pattern in include_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            # Skip if it's a X-free requirement (like "including gluten-free")
            if not re.search(r'\w+[-\s]free', match):
                filters["include"].append(match.strip())
    
    # Pattern 4: Special dietary/category requirements
    special_req_patterns = [
        r'\b(vegetarian|vegan|kosher|halal|organic|natural|fresh|local)\b',
        r'\b(budget[-\s]friendly|cheap|affordable|expensive|premium|luxury)\b',
        r'\b(quick|fast|slow|easy|simple|complex|advanced|beginner)\b',
        r'\b(indoor|outdoor|portable|stationary|mobile|desktop)\b',
    ]
    
    for pattern in special_req_patterns:
        matches = re.findall(pattern, query_lower)
        filters["special_requirements"].extend(matches)
    
    # Pattern 5: Numeric constraints
    numeric_patterns = [
        (r'\b(?:under|less\s+than|below|maximum|max)\s+(\d+)', 'max'),
        (r'\b(?:over|more\s+than|above|minimum|min|at\s+least)\s+(\d+)', 'min'),
        (r'\b(?:exactly|equal\s+to|precisely)\s+(\d+)', 'exact'),
        (r'\b(\d+)\s*[-–]\s*(\d+)', 'range'),  # matches "5-10", "20–30"
    ]
    
    for pattern, constraint_type in numeric_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if constraint_type == 'range':
                filters["numeric"]["min"] = int(match[0])
                filters["numeric"]["max"] = int(match[1])
            else:
                filters["numeric"][constraint_type] = int(match)
    
    return filters

def create_dynamic_keyword_lists(filters, model=None, all_texts=None):
    """
    Dynamically create keyword lists using multiple strategies:
    1. Linguistic pattern analysis (X-free, negations, etc.)
    2. Semantic similarity using embeddings
    3. Predefined mappings for common cases as fallback
    """
    
    # Strategy 1: Linguistic Pattern Analysis
    def analyze_term_patterns(term):
        """Analyze linguistic patterns in terms"""
        term_lower = term.lower().strip()
        exclusions = set()
        
        # Handle X-free patterns
        if re.match(r'(\w+)[-\s]free$', term_lower):
            base = re.match(r'(\w+)[-\s]free$', term_lower).group(1)
            exclusions.add(base)
            
            # Add common variations
            if base == 'gluten':
                exclusions.update(['wheat', 'barley', 'rye', 'flour', 'bread', 'pasta', 'noodles'])
            elif base == 'dairy':
                exclusions.update(['milk', 'cheese', 'butter', 'cream', 'yogurt'])
            elif base == 'sugar':
                exclusions.update(['sweetener', 'honey', 'syrup', 'candy'])
                
        # Handle negation prefixes
        elif term_lower.startswith(('non', 'un', 'dis', 'anti')):
            # Extract base term and add it to exclusions
            for prefix in ['non', 'un', 'dis', 'anti']:
                if term_lower.startswith(prefix):
                    base = term_lower[len(prefix):].lstrip('-')
                    exclusions.add(base)
                    break
        
        # Handle dietary requirements
        elif term_lower == 'vegetarian':
            exclusions.update(['egg', 'meat', 'beef', 'pork', 'chicken', 'fish', 'seafood', 'bacon', 'ham', 'turkey', 'lamb', 'duck', 'sausage'])
        elif term_lower == 'vegan':
            exclusions.update(['meat', 'beef', 'pork', 'chicken', 'fish', 'dairy', 'milk', 'cheese', 'butter', 'cream', 'egg', 'honey'])
            
        # Handle quality/complexity terms
        elif term_lower in ['simple', 'easy', 'basic']:
            exclusions.update(['complex', 'complicated', 'advanced', 'difficult'])
        elif term_lower in ['quick', 'fast']:
            exclusions.update(['slow', 'lengthy', 'time-consuming'])
        elif term_lower in ['cheap', 'budget', 'affordable']:
            exclusions.update(['expensive', 'premium', 'luxury', 'costly'])
            
        # Handle technology terms
        elif term_lower == 'mobile':
            exclusions.update(['desktop', 'stationary', 'fixed'])
        elif term_lower == 'wireless':
            exclusions.update(['wired', 'cable', 'corded'])
            
        return list(exclusions)
    
    # Strategy 2: Semantic Similarity (if model available)
    def get_semantic_exclusions(term, similarity_threshold=0.4):
        """Find semantically related terms that might need to be excluded"""
        if not model or not all_texts:
            return []
            
        try:
            # Get all unique words from texts
            all_words = set()
            for text in all_texts:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                all_words.update(words)
            
            # Remove common stop words and the term itself
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'with', 'this', 'that', 'have', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other'}
            relevant_words = [w for w in all_words if w not in stop_words and w != term.lower() and len(w) > 2]
            
            if not relevant_words:
                return []
            
            # Get embeddings
            term_embedding = model.encode([term])
            word_embeddings = model.encode(relevant_words[:200])  # Limit to avoid memory issues
            
            # Find similar terms
            similarities = util.cos_sim(term_embedding, word_embeddings)[0]
            similar_terms = []
            
            for i, sim in enumerate(similarities):
                if sim > similarity_threshold:
                    similar_terms.append((relevant_words[i], float(sim)))
            
            # Sort by similarity and return top terms
            similar_terms.sort(key=lambda x: x[1], reverse=True)
            return [term for term, _ in similar_terms[:5]]
            
        except Exception as e:
            print(f"Error in semantic analysis for '{term}': {e}")
            return []
    
    # Main processing
    dynamic_excludes = set()
    dynamic_includes = set()
    
    # Process special requirements
    for req in filters.get('special_requirements', []):
        # Use linguistic pattern analysis
        pattern_exclusions = analyze_term_patterns(req)
        dynamic_excludes.update(pattern_exclusions)
        
        # Use semantic similarity as additional source
        if model and all_texts:
            semantic_exclusions = get_semantic_exclusions(req)
            # Be more conservative with semantic matches to avoid noise
            dynamic_excludes.update(semantic_exclusions[:3])
    
    # Process explicit exclusions
    for exclude_term in filters.get('exclude', []):
        dynamic_excludes.add(exclude_term)
        
        # Add pattern-based exclusions
        pattern_exclusions = analyze_term_patterns(exclude_term)
        dynamic_excludes.update(pattern_exclusions)
        
        # Add semantic exclusions (more conservative)
        if model and all_texts:
            semantic_exclusions = get_semantic_exclusions(exclude_term)
            dynamic_excludes.update(semantic_exclusions[:2])
    
    # Process inclusions
    for include_term in filters.get('include', []):
        dynamic_includes.add(include_term)
        
        # For inclusions, we might want to find similar terms to also include
        if model and all_texts:
            semantic_inclusions = get_semantic_exclusions(include_term, similarity_threshold=0.5)
            dynamic_includes.update(semantic_inclusions[:2])
    
    return list(dynamic_excludes), list(dynamic_includes)

def check_constraints(text, filters, model=None, all_texts=None):
    """
    Generic constraint checker that works for any domain.
    """
    text_lower = text.lower()
    
    # Get dynamic keyword lists based on detected requirements
    dynamic_excludes, dynamic_includes = create_dynamic_keyword_lists(filters, model, all_texts)
    
    # Check dynamic exclusions (from special requirements like "vegetarian")
    
    for keyword in dynamic_excludes:
        if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
            return False
    
    # Check explicit exclusions from query parsing
    for exclude_term in filters.get("exclude", []):
        if re.search(rf'\b{re.escape(exclude_term)}\b', text_lower):
            return False
    
    # Check required inclusions
    all_includes = filters.get("include", []) + dynamic_includes
    for include_term in all_includes:
        if not re.search(rf'\b{re.escape(include_term)}\b', text_lower):
            return False
    
    # Check numeric constraints (if text contains numbers)
    numbers_in_text = re.findall(r'\b(\d+)\b', text_lower)
    if numbers_in_text and filters.get("numeric"):
        text_numbers = [int(n) for n in numbers_in_text]
        numeric_filters = filters["numeric"]
        
        if "max" in numeric_filters:
            if any(n > numeric_filters["max"] for n in text_numbers):
                return False
        
        if "min" in numeric_filters:
            if any(n < numeric_filters["min"] for n in text_numbers):
                return False
        
        if "exact" in numeric_filters:
            if numeric_filters["exact"] not in text_numbers:
                return False
    
    return True
filters = parse_generic_query(query)
# print(f"Parsed filters: {filters}")

top_sections = []
top_texts = []
for sec, txt in zip(top_sections_raw, top_texts_raw):
    if check_constraints(txt, filters, model, texts):
        top_sections.append(sec)
        top_texts.append(txt)

print(f"After filtering: {len(top_sections)} sections remain out of {len(top_sections_raw)}")

# Improved fallback: if everything is filtered due to strict constraints,
# try to relax constraints or increase the initial top_k
if not top_sections:
    print("Warning: All sections were filtered out. Expanding search...")
    # Try with more sections
    cosine_top_k_expanded = min(50, len(all_sections))  # Expand search
    top_indices_expanded = torch.topk(cosine_scores, k=cosine_top_k_expanded).indices.tolist()
    
    top_sections_expanded = [all_sections[i] for i in top_indices_expanded]
    top_texts_expanded = [texts[i] for i in top_indices_expanded]
    
    for sec, txt in zip(top_sections_expanded, top_texts_expanded):
        if check_constraints(txt, filters, model, texts):
            top_sections.append(sec)
            top_texts.append(txt)
    
    # If still nothing found, there might be an issue with the documents
    if not top_sections:
        print("Error: No sections match the constraints!")
        print("Sample texts being filtered:")
        for i, txt in enumerate(top_texts_raw[:3]):
            print(f"Text {i+1}: {txt[:100]}...")

# Step 4B: Rerank using Cross-Encoder
if not top_sections:
    print("No sections found after filtering - skipping cross-encoder step")
    reranked_sections = []
else:
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",local_files_only=True,cache_folder="/root/.cache")
    cross_input = [(query, txt) for txt in top_texts]
    cross_scores = cross_model.predict(cross_input)

    # Step 4C: Zip and sort
    reranked_sections = [
        {
            "score": float(score),
            "text": sec["text"],
            "document": sec.get("document", "unknown.pdf"),
            "section_title": sec.get("section_title", ""),
            "page_number": sec.get("page", -1)
        }
        for score, sec in zip(cross_scores, top_sections)
    ]

    # Sort descending by score
    reranked_sections.sort(key=lambda x: x["score"], reverse=True)


output = {
    "metadata": {
        "input_documents": pdf_files,
        "persona": persona_description,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": datetime.now().isoformat()

    },
    "extracted_sections": [],
    "subsection_analysis": []
}

top_k =5
for idx, sec in enumerate(reranked_sections[:top_k]):
    output["extracted_sections"].append({
        "document": sec["document"],
        "section_title": sec["section_title"],
        "importance_rank": idx + 1,
        "page_number": sec["page_number"]+1
    })
    output["subsection_analysis"].append({
        "document": sec["document"],
        "refined_text": sec["text"],
        "page_number": sec["page_number"]+1
    })

# ---- STEP 6: Save Output ----
with open("output/output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(" Output saved")