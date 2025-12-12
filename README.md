## LLM Response Evaluation Pipeline

This repository implements an automated real-time evaluation system for AI chatbot responses using Google Gemini. The pipeline assesses responses across three critical dimensions: Relevance & Completeness, Factual Accuracy (Hallucination Detection), and Latency & Cost.

---

### Local Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/llm-eval-pipeline.git
   cd llm-eval-pipeline
   ```

2. Create and activate a virtual environmeny
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

5. Prepare input files
   - Place your conversation and context files in the project root
   - Ensure files follow valid JSON format

6. Run the evaluation
   ```bash
   python eval_pipeline.py
   ```

> **Important Note on JSON Format**  
> Input JSON files must be strictly valid JSON:
> - No JavaScript-style comments (`// ...` or `/* ... */`)
> - No trailing commas after the last item in objects/arrays
> - All strings must be properly quoted

---

### Architecture

The evaluation pipeline is a lightweight, modular Python script designed for real-time reliability assessment of LLM responses. It accepts two standard JSON inputs: a chat conversation file and a context vectors file retrieved from a vector database.

The system begins by parsing both JSON files. From the conversation file, it extracts the last user message (the query) and the last AI/Chatbot response. From the context vectors file, it aggregates all retrieved text snippets into a single contextual reference.

This data is then passed to two independent evaluation modules, both powered by Google Gemini 2.5 Flash:

Relevance & Completeness Evaluator: This module uses a structured prompt to assess how directly the AI response addresses the user’s query and how thoroughly it covers the key information present in the retrieved context. It outputs a score from 0 to 5 for each dimension along with a brief justification.

Hallucination & Factual Accuracy Evaluator: This module verifies whether all factual claims in the AI response are supported by the provided context. It assigns a hallucination score (0–5, where 0 means no hallucination) and a factual accuracy rating (“high”, “medium”, or “low”), also with a short explanation.

Both evaluators enforce JSON-formatted outputs using Gemini’s `response_mime_type` configuration to ensure reliable parsing. The pipeline measures the latency of each evaluation step and estimates cost based on token usage using official Gemini Flash pricing. Finally, all metrics and reasoning are aggregated into a single structured JSON result for easy consumption by monitoring or logging systems.

---

### Why this approach over alternatives?

1. Gemini 2.5 Flash as evaluator 
   - Chosen for its balance of speed, cost, and instruction-following capability
   - Superior to rule-based methods for nuanced evaluation of relevance and factual accuracy
   - More reliable than embedding-based similarity for hallucination detection

2. Separate evaluation prompts  
   - Isolates concerns: relevance/completeness vs factual accuracy
   - Enables targeted prompt engineering and clearer failure diagnosis
   - Avoids conflating different evaluation dimensions

3. Standard JSON parser 
   - Enforces production-grade data standards
   - Avoids unnecessary dependencies that could break in minimal environments

4. Clear prompt design  
   - Structured instructions with clear scoring rubrics
   - Enforces JSON output format for reliable parsing
   - Includes concise reasoning for interpretability

---

### Scalability & Efficiency at Scale

For millions of daily evaluations, this pipeline minimizes latency and cost through:

1. Model Selection  
   Uses Gemini 2.5 Flash — Google's fastest and most cost-effective model ($0.00001875/1k input tokens, $0.000075/1k output tokens)

2. Cost Efficiency  
   - ~$0.0003 per evaluation
   - Token-efficient prompts with precise instructions

3. Latency Optimization  
   - Stateless design enables horizontal scaling
   - Can be deployed on serverless platforms (Cloud Run, Lambda) with auto-scaling
   - Parallel evaluation of multiple conversations

5. Batch Processing Option  
   While designed for real-time use, the architecture easily supports:
   - Batch evaluation jobs for offline analysis
   - Caching of repeated queries to avoid redundant evaluations
   - Asynchronous processing for non-blocking workflows

This design ensures the pipeline remains cost-effective and responsive even under massive scale, making it suitable for both real-time monitoring and large-scale retrospective analysis.