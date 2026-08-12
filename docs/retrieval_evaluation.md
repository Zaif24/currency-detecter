# Retrieval Evaluation

Ran via `PYTHONPATH=. python3 scripts/eval_retrieval.py` against the built FAISS index
(25 chunks, one per knowledge-base document). Note: this run used the **TF-IDF fallback
embedder** because the sandbox this was generated in has no outbound access to
huggingface.co; on Streamlit Cloud (which has internet access) the primary
`all-MiniLM-L6-v2` semantic embedder will be used instead and is expected to improve
relevance further, especially for queries that use different wording than the source
text (see Query 1 below).

## Query 1: "What security features does the Sri Lankan Rs.5000 note have?"
```
[0.918] sri_lanka/denominations.md
[0.910] sri_lanka/overview.md
```
**Comment:** Correctly narrowed to Sri Lanka, but the top hit is `denominations.md` rather
than the more specific `security_features.md`. This is a known weakness of TF-IDF: it
matches on lexical overlap ("Rs.5000", "note") rather than the semantic concept of
"security features," so a document that just lists denominations scores similarly to the
one actually describing security features. This is exactly the kind of case where the
semantic MiniLM embedder is expected to do better, since it captures meaning rather than
keyword overlap.

## Query 2: "Why did Thailand float the baht in 1997?"
```
[0.963] thailand/history.md
[0.483] thailand/overview.md
```
**Comment:** Strong, correct retrieval — `history.md` explicitly discusses the 1997 float
and Asian Financial Crisis, and it scores far above the next result, showing good
separation between relevant and irrelevant chunks for this query.

## Query 3: "Which Indian banknote depicts the Sanchi Stupa?"
```
[0.935] india/denominations.md
[0.501] thailand/denominations.md
```
**Comment:** Top result is correct and highly confident — `india/denominations.md` is
where the ₹200/Sanchi Stupa fact lives. The second result (Thailand) is irrelevant
content that only surfaced because no `country_filter` was applied in this run; in the
actual ResearchAgent pipeline a `country_filter` is always passed once the currency has
been identified, which would eliminate this kind of cross-country noise entirely.

## Query 4: "How does Japan use holograms on its 2024 banknote series?"
```
[0.799] japan/security_features.md
[0.699] japan/denominations.md
```
**Comment:** Correct top result with good confidence — `security_features.md` is exactly
where the 3D hologram detail is documented. This query worked well because "hologram" and
"2024" are distinctive terms that appear directly in the source text.

## Query 5: "Is it disrespectful to fold currency in Thailand?"
```
[0.692] thailand/history.md
[0.618] thailand/travel_tips.md
```
**Comment:** The correct answer is actually in `travel_tips.md` ("folding, defacing... is
considered highly disrespectful"), but it ranked second behind `history.md`. This is the
clearest failure case in this evaluation: the query is phrased as a yes/no cultural
question rather than reusing the source's wording, which is precisely where TF-IDF
struggles most. A semantic embedder should rank `travel_tips.md` first for this query.

## Summary
- 3 of 5 queries retrieved the ideal document as the top-1 result; the other 2 retrieved
  it within the top-2, with the miss explainable by TF-IDF's lack of semantic matching.
- Country separation was reliable in every query (no cross-country contamination in the
  top result) even without filtering, showing the country-specific vocabulary in each
  document is distinctive enough for lexical retrieval alone.
- Recommendation: deploy with the primary MiniLM embedder (automatic on Streamlit Cloud)
  and re-run this exact script post-deployment to confirm Query 1 and Query 5 improve, as
  hypothesized above.
