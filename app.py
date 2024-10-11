from flask import Flask, render_template, request, send_from_directory
from probabilistic_model import preprocess, load_documents, compute_statistics, search_bm25

app = Flask(__name__)

# Load documents and compute statistics
DOCS_FOLDER = 'Dataset'
documents = load_documents(DOCS_FOLDER)
doc_freqs, doc_lengths, avg_doc_length, N = compute_statistics(documents)

RESULTS_PER_PAGE = 7  # Set the number of results per page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query') or request.args.get('query')
    page = int(request.args.get('page', 1))
    
    # Run the search query
    results = search_bm25(query, documents, doc_freqs, doc_lengths, avg_doc_length, N)
    
    # Pagination logic: slice the results list
    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    paginated_results = results[start:end]
    
    # Calculate if there's a next or previous page
    next_page = page + 1 if end < len(results) else None
    prev_page = page - 1 if page > 1 else None
    
    # Format results for display (keep the full filename including .txt)
    formatted_results = [{"filename": doc_id, "score": round(score, 4)} for doc_id, score in paginated_results]
    
    return render_template('results.html', query=query, results=formatted_results, next_page=next_page, prev_page=prev_page)

@app.route('/documents/<doc_id>')
def view_document(doc_id):
    # Serve the document from the 'Dataset' folder
    return send_from_directory(DOCS_FOLDER, doc_id)

if __name__ == '__main__':
    app.run(debug=True)
