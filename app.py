import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import csv
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="centered",
)

st.markdown("""
<style>
    .block-container { max-width: 780px; }
    .stMultiSelect [data-baseweb="tag"] { background-color: #e8f4f8; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Artifact paths — update these to match your deployment ───────────────────
BASE = "artifacts"
ARTIFACTS = {
    "cbf_sim_matrix": f"{BASE}/cbf_sim_matrix.npy",
    "cbf_meta":       f"{BASE}/cbf_meta.pkl",
    "svd_model":      f"{BASE}/svd_model.pkl",
    "ratings_df":     f"{BASE}/ratings_filtered.pkl",
}
BOOKS_CSV   = "books_with_image_url.csv"
EVAL_CSV    = "evaluations.csv"
N_RANDOM    = 10   # books shown for selection
N_RECS      = 10   # recommendations returned

alpha = 0.8  # default to more CF influence, since CBF alone is often too generic


# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    missing = [p for p in ARTIFACTS.values() if not os.path.exists(p)]
    if missing:
        return None, f"Missing artifact files:\n" + "\n".join(missing)

    cbf_sim_matrix = np.load(ARTIFACTS["cbf_sim_matrix"]).astype(np.float64)

    with open(ARTIFACTS["cbf_meta"], "rb") as f:
        meta = pickle.load(f)

    with open(ARTIFACTS["svd_model"], "rb") as f:
        svd_model = pickle.load(f)

    with open(ARTIFACTS["ratings_df"], "rb") as f:
        rd = pickle.load(f)

    books = pd.read_csv(BOOKS_CSV, index_col=0)

    return {
        "cbf_sim_matrix":  cbf_sim_matrix,
        "cbf_df":          meta["cbf_df"],
        "cbf_title_to_idx":meta["cbf_title_to_idx"],
        "cbf_id_to_idx":   meta["cbf_id_to_idx"],
        "svd_model":       svd_model,
        "ratings_filtered":rd["ratings_filtered"],
        "cf_all_book_ids": rd["cf_all_book_ids"],
        "books":           books,
    }, None


# ── Recommendation logic ──────────────────────────────────────────────────────
def get_cbf_scores_from_list(titles, art):
    cbf_df          = art["cbf_df"]
    cbf_title_to_idx= art["cbf_title_to_idx"]
    cbf_sim_matrix  = art["cbf_sim_matrix"]

    valid = [t for t in titles if t in cbf_title_to_idx]
    if not valid:
        return {}
    rows   = np.array([cbf_sim_matrix[cbf_title_to_idx[t]] for t in valid])
    agg    = rows.mean(axis=0)
    mn, mx = agg.min(), agg.max()
    normed = (agg - mn) / (mx - mn) if mx != mn else np.zeros_like(agg)
    return {cbf_df.iloc[i]["book_id"]: float(normed[i]) for i in range(len(normed))}


def get_cf_scores_from_list(titles, art):
    cbf_df    = art["cbf_df"]
    svd_model = art["svd_model"]
    cf_ids    = art["cf_all_book_ids"]

    inner_ids = []
    for title in titles:
        row = cbf_df[cbf_df["title"] == title]
        if row.empty:
            continue
        bid = row["book_id"].values[0]
        try:
            inner_ids.append(svd_model.trainset.to_inner_iid(bid))
        except ValueError:
            pass

    if not inner_ids:
        return {}

    pseudo_user = svd_model.qi[inner_ids].mean(axis=0)
    global_mean = svd_model.trainset.global_mean
    raw = {}
    for bid in cf_ids:
        try:
            iid   = svd_model.trainset.to_inner_iid(bid)
            score = global_mean + svd_model.bi[iid] + np.dot(pseudo_user, svd_model.qi[iid])
            raw[bid] = score
        except ValueError:
            continue

    vals   = np.array(list(raw.values()))
    mn, mx = vals.min(), vals.max()
    normed = (vals - mn) / (mx - mn) if mx != mn else np.zeros_like(vals)
    return {bid: float(normed[i]) for i, bid in enumerate(raw)}


def recommend_from_books(read_titles, art, n=10, alpha=0.8):
    books      = art["books"]
    cbf_df     = art["cbf_df"]

    cbf_scores = get_cbf_scores_from_list(read_titles, art)
    cf_scores  = get_cf_scores_from_list(read_titles, art)

    if not cbf_scores and not cf_scores:
        return None, "None of the selected books were found in the dataset."

    eff_alpha = alpha
    if not cf_scores:
        eff_alpha = 0.0
    if not cbf_scores:
        eff_alpha = 1.0

    read_ids     = set(cbf_df[cbf_df["title"].isin(read_titles)]["book_id"])
    all_book_ids = set(cf_scores) | set(cbf_scores)

    results = []
    for bid in all_book_ids:
        if bid in read_ids:
            continue
        cf_s   = cf_scores.get(bid, 0.0)
        cbf_s  = cbf_scores.get(bid, 0.0)
        hybrid = eff_alpha * cf_s + (1 - eff_alpha) * cbf_s

        row = books[books["book_id"] == bid]
        if row.empty:
            row = books[books["id"] == bid]
        if row.empty:
            continue

        results.append({
            "Title":        row["title"].values[0],
            "Authors":      row["raw_authors"].values[0],
            "Original Publication Year": row["raw_year"].values[0],
            "Image URL":    row["image_url"].values[0],
            "CF Score":     round(cf_s,   4),
            "CBF Score":    round(cbf_s,  4),
            "Hybrid Score": round(hybrid, 4),
        })

    if not results:
        return None, "No recommendations found."

    df = (
        pd.DataFrame(results)
        .sort_values("Hybrid Score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    df.index     = range(1, len(df) + 1)
    df.index.name = "Rank"
    return df, None


# ── Evaluation helpers ────────────────────────────────────────────────────────
def save_evaluation(selected_books, recs_df, rating, helpful, comments):
    file_exists = os.path.isfile(EVAL_CSV)
    with open(EVAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "selected_books", "recommended_titles",
                "satisfaction_rating", "found_helpful", "comments"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            " | ".join(selected_books),
            " | ".join(recs_df["Title"].tolist()) if recs_df is not None else "",
            " | ".join(recs_df["Hybrid Score"].astype(str).tolist()) if recs_df is not None else "",
            rating,
            helpful,
            comments,
        ])


# ── Session state init ────────────────────────────────────────────────────────
if "random_books" not in st.session_state:
    st.session_state.random_books = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "eval_submitted" not in st.session_state:
    st.session_state.eval_submitted = False
if "selected_books" not in st.session_state:
    st.session_state.selected_books = []

# ── Load model ────────────────────────────────────────────────────────────────
art, err = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Book Recommender")
st.caption("A hybrid Content-Based + Collaborative Filtering recommendation system.")

if err:
    st.error(err)
    st.info("Make sure your artifact files and `books_clean.csv` are in the correct paths, then refresh.")
    st.stop()

cbf_df = art["cbf_df"]
all_titles = cbf_df["title"].dropna().unique().tolist()

# ── Step 1 · Pick your books ─────────────────────────────────────────────────
st.markdown("## Pick books you've enjoyed")
st.caption("Search for books you've read, then click to select them.")

books_df = art["books"]
selected = st.session_state.selected_books.copy()

# Search bar
search_query = st.text_input(
    "Search for a book by title or author:",
    placeholder="e.g., 'Harry Potter' or 'J.K. Rowling'",
    label_visibility="collapsed"
)

# Filter books based on search query
if search_query.strip():
    search_lower = search_query.lower()
    # Search in both title and authors columns
    mask = (
        books_df["title"].str.lower().str.contains(search_lower, na=False) |
        books_df["raw_authors"].str.lower().str.contains(search_lower, na=False)
    )
    search_results = books_df[mask]["title"].dropna().unique().tolist()
    # Limit results to avoid overwhelming the UI
    search_results = search_results[:50]
else:
    search_results = []

# Display search results or prompt
if search_query.strip():
    if search_results:
        st.caption(f"Found {len(search_results)} book(s)")
        
        # Create a grid of book cards
        cols = st.columns(5)  # 5 cards per row
        for idx, book_title in enumerate(search_results):
            col = cols[idx % 5]
            
            book_row = books_df[books_df["title"] == book_title]
            image_url = book_row["image_url"].values[0] if not book_row.empty and pd.notna(book_row["image_url"].values[0]) else None
            
            with col:
                with st.container():
                    # Create a card-like container
                    is_selected = book_title in selected
                    
                    # Display image
                    if image_url:
                        try:
                            st.image(image_url, use_container_width=True)
                        except:
                            st.markdown("📖")
                    else:
                        st.markdown("📖")
                    
                    # Display title as a button/selector with color change based on selection
                    if st.button(
                        f"{book_title}",
                        key=f"book_card_{book_title}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        if book_title in selected:
                            selected.remove(book_title)
                        else:
                            selected.append(book_title)
                        st.session_state.selected_books = selected
                        st.rerun()
    else:
        st.info(f"No books found matching '{search_query}'. Try a different search.")
else:
    st.info("👆 Start typing to search for books by title or author")

st.session_state.selected_books = selected

# ── Show selected books ───────────────────────────────────────────────────────
if selected:
    st.markdown("### Your selected books:")
    cols = st.columns(5)  # 5 cards per row
    for idx, book_title in enumerate(selected):
        col = cols[idx % 5]
        
        book_row = books_df[books_df["title"] == book_title]
        image_url = book_row["image_url"].values[0] if not book_row.empty and pd.notna(book_row["image_url"].values[0]) else None
        
        with col:
            with st.container():
                # Display image
                if image_url:
                    try:
                        st.image(image_url, use_container_width=True)
                    except:
                        st.markdown("📖")
                else:
                    st.markdown("📖")
                
                # Display title as remove button
                if st.button(
                    f"✓ {book_title}",
                    key=f"remove_book_{book_title}",
                    use_container_width=True,
                    type="primary"
                ):
                    selected.remove(book_title)
                    st.session_state.selected_books = selected
                    st.rerun()

# ── Alpha slider ──────────────────────────────────────────────────────────────
# with st.expander("⚙️ Advanced — blending weight", expanded=False):
#     alpha = st.slider(
#         "CF weight (alpha) — 0 = pure content, 1 = pure collaborative",
#         min_value=0.0, max_value=1.0, value=0.5, step=0.05,
#     )
# else:
#     alpha = 0.5


# ── Get recommendations ───────────────────────────────────────────────────────
if st.button("Get recommendations", type="primary", disabled=len(selected) == 0):
    with st.spinner("Building your reading list…"):
        recs, rec_err = recommend_from_books(selected, art, n=N_RECS, alpha=alpha)
    if rec_err:
        st.warning(rec_err)
    else:
        st.session_state.recommendations = recs
        st.session_state.eval_submitted = False

# ── Step 2 · Show results ─────────────────────────────────────────────────────
if st.session_state.recommendations is not None:
    recs_df = st.session_state.recommendations
    st.markdown("---")
    st.markdown(f"## Your top {len(recs_df)} recommendations")
    st.caption(
        f"Based on **{len(selected)} book(s)** you selected · "
        # f"CF weight: {alpha:.2f} · CBF weight: {1-alpha:.2f}"
    )

    for i, row in recs_df.iterrows():
        with st.container():
            c1, c2 = st.columns([5, 2])
            with c1:
                st.markdown(f"**{i}. {row['Title']}**")
                st.caption(f"{row['Original Publication Year']} · {row['Authors']}")
            with c2:
                st.image(f"{row['Image URL']}")
                # st.metric("Hybrid score", f"{row['Hybrid Score']:.3f}")

    # with st.expander("📊 See full score table"):
    #     st.dataframe(recs_df, use_container_width=True)

    # ── Step 3 · Evaluation ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Result Evaluation?")
    st.caption("Your feedback helps improve the model.")

    if not st.session_state.eval_submitted:
        with st.form("evaluation_form"):
            rating = st.radio(
                "Overall satisfaction with these recommendations:",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: {
                    1: "Very dissatisfied",
                    2: "Dissatisfied",
                    3: "Neutral",
                    4: "Satisfied",
                    5: "Very satisfied",
                }[x],
                horizontal=True,
            )

            helpful = st.radio(
                "Did the recommendations feel relevant to the books you picked?",
                options=["Yes", "Somewhat", "No"],
                horizontal=True,
            )

            comments = st.text_area(
                "Any comments? (optional)",
                placeholder="e.g. 'Too many similar authors' or 'Loved the variety!'",
                max_chars=500,
            )

            submitted = st.form_submit_button("Submit feedback", type="primary")

        if submitted:
            save_evaluation(selected, recs_df, rating, helpful, comments)
            st.session_state.eval_submitted = True
            st.rerun()

    else:
        st.success("Thanks for your feedback! It's been saved.")
        if st.button("Start over"):
            for key in ["random_books", "recommendations", "eval_submitted", "selected_books"]:
                st.session_state[key] = None if key != "selected_books" else []
            st.rerun()
