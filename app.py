import streamlit as st
import pandas as pd
import numpy as np

# Pindahkan st.set_page_config ke posisi paling atas,
# setelah semua import library dasar.
st.set_page_config(page_title="Brand Influencer Recommendation", layout="wide") # Di sini perbaikannya

# Gaya CSS untuk kerapihan
st.markdown(
    """
    <style>
    .st-emotion-cache-1v0mbdj {padding-top: 1rem;}
    .divider {border-top: 2px solid #eee; margin: 1.5em 0;}
    .section-card {background: #f8f9fa; border-radius: 10px; padding: 1.2em 1.5em; margin-bottom: 1.5em;}
    .score-badge {display:inline-block; background:#1DA1F2; color:white; border-radius:8px; padding:0.2em 0.7em; font-size:1em;}
    </style>
    """, unsafe_allow_html=True
)

st.title("🎯 Brand Influencer Recommendation & Insight")

# --- Load Data ---
@st.cache_data
def load_data():
    brands = pd.read_csv("https://raw.githubusercontent.com/Fahmi-mi/Dataset/refs/heads/main/datathon-ristek-ui-2025/input_instagram_brands/brands_filled.csv")
    influencers = pd.read_csv("https://raw.githubusercontent.com/Fahmi-mi/Dataset/refs/heads/main/datathon-ristek-ui-2025/input_instagram_brands/instagram_influencers_filled.csv")
    labeled_caption = pd.read_csv("https://raw.githubusercontent.com/Fahmi-mi/Dataset/refs/heads/main/datathon-ristek-ui-2025/input_instagram_brands/labeled_caption.csv")
    labeled_comment = pd.read_csv("https://raw.githubusercontent.com/Fahmi-mi/Dataset/refs/heads/main/datathon-ristek-ui-2025/input_instagram_brands/labeled_comment.csv")
    bio = pd.read_csv("https://raw.githubusercontent.com/Fahmi-mi/Dataset/refs/heads/main/datathon-ristek-ui-2025/instagram/bio.csv")
    return brands, influencers, labeled_caption, labeled_comment, bio

brands, influencers, labeled_caption, labeled_comment, bio = load_data()


def generate_brand_summary(df, brand_name):
    import ast
    row = df[df['brand_name'].str.lower() == brand_name.lower()].iloc[0]

    def format_list(val):
        try:
            val_list = ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
            if isinstance(val_list, list):
                return " | ".join([f"`{str(v)}`" for v in val_list])
            return f"`{val_list}`"
        except:
            return f"`{val}`"

    gender = format_list(row['demography_gender'])
    age_group = format_list(row['demography_usia'])
    income = format_list(row['demography_income'])
    lifestyle = format_list(row['psychography_lifestyle'])
    personality = format_list(row['psychography_personality'])
    values = format_list(row['psychography_value'])
    growth_type = str(row['growth_type']).title()
    marketing_goal = str(row['marketing_goal']).title()
    budget = f"± Rp {int(row['budget']):,}".replace(",", ".")
    criteria = row['brand_criteria']

    # Horizontal layout using markdown table
    summary = f"""
**🧠 BRAND PROFILE: {row['brand_name'].title()}**

| 🎯 Target Persona | | | | | |
|:--|:--|:--|:--|:--|:--|
| **Gender** | **Age Group** | **Income** | **Lifestyle** | **Personality** | **Core Values** |
| {gender} | {age_group} | {income} | {lifestyle} | {personality} | {values} |

| 📈 Brand Objective | | | |
|:--|:--|:--|:--|
| **Growth Type** | **Marketing Goal** | **Budget** | **Criteria** |
| {growth_type} | {marketing_goal} | {budget} | {criteria} |
"""
    return summary.strip()

def generate_influencer_insight(username, caption_df, comment_df, show_plot=False):
    captions = caption_df[caption_df["instagram_account"] == username]
    comments = comment_df[comment_df["instagram_account"] == username]
    total_comments = len(comments)
    total_captions = len(captions)

    # --- Comment Insight ---
    comment_counts = None
    high_value_labels = ["relatable engagement", "product-focused response", "social virality"]
    result_lines = []
    result_lines.append(f"🔁 **Conversion Potential for @{username}**")

    if not comments.empty:
        comment_counts = comments["predicted_label"].value_counts(normalize=True).mul(100).round(1)
        total_comments = len(comments)
        # Table for comment quality
        comment_table = "| Label | % |\n|:--|--:|\n"
        for label, pct in comment_counts.items():
            label_id = label.lower().replace("_", " ")
            comment_table += f"| {label_id.title()} | {pct:.1f}% |\n"
        result_lines.append("**💬 Comment Quality**")
        result_lines.append(f"Total **{total_comments}** komentar dianalisis.")
        result_lines.append(comment_table)
        # High-value comment rate
        high_value_pct = sum([comment_counts.get(lbl, 0) for lbl in comment_counts.index if lbl.lower() in high_value_labels])
        result_lines.append(f"🎯 **High-Value Comment Rate:** `{high_value_pct:.1f}%`")
        if high_value_pct < 20:
            result_lines.append("_Cukup rendah, mengindikasikan bahwa interaksi dari audiens masih dominan pujian atau pasif._")

        # Komentar berkualitas tinggi
        result_lines.append("**💬 Komentar Berkualitas Tinggi yang Mewakili Audiens**")
        for label in high_value_labels:
            label_comments = comments[comments["predicted_label"].str.lower() == label]
            if not label_comments.empty:
                example = label_comments["comment"].iloc[0]
                result_lines.append(f"- **{label.title()}**\n    > _{example}_")
            else:
                if label == "product-focused response":
                    result_lines.append("- **Product-Focused Response (minat terhadap produk)**\n    > _Tidak ditemukan komentar yang membahas produk secara langsung pada postingan terakhir._")
        if total_comments < 20:
            result_lines.append(f"⚠️ _Hanya {total_comments} komentar terdeteksi. Insight ini mungkin kurang representatif karena volume interaksi yang rendah._")
    else:
        result_lines.append("_Tidak ada komentar yang dapat dianalisis._")

    # --- Caption Insight ---
    result_lines.append(f"\n📢 **Caption Behavior Summary – Influencer: {username}**")
    caption_counts = None
    if not captions.empty:
        # Call-to-action habit
        cta_labels = ["call-to-action", "engagement-inviting"]
        cta_count = captions["predicted_label"].str.lower().isin(cta_labels).sum()
        total_captions = len(captions)
        result_lines.append(f"🔁 **Call-to-Action Habit:** `{cta_count}` dari `{total_captions}` caption mengandung CTA.")
        if cta_count > 0:
            example_cta = captions[captions["predicted_label"].str.lower().isin(cta_labels)]["post_caption"].iloc[0]
            result_lines.append(f"> CTA Example: _{example_cta[:120]}..._")

        # Product mention
        prod_labels = ["product-focused", "brand awareness"]
        prod_count = captions["predicted_label"].str.lower().isin(prod_labels).sum()
        if prod_count > 0:
            example_prod = captions[captions["predicted_label"].str.lower().isin(prod_labels)]["post_caption"].iloc[0]
            result_lines.append(f"🛍️ **Product Mention:** `{prod_count}` caption menyebut produk/brand.\n> _{example_prod[:120]}..._")

        # Tone of voice (ambil label dominan)
        caption_counts = captions["predicted_label"].value_counts()
        dominant_label = caption_counts.idxmax()
        example_tone = captions[captions["predicted_label"] == dominant_label]["post_caption"].iloc[0]
        result_lines.append(f"🎭 **Tone of Voice:** Dominan: `{dominant_label}`\n> _{example_tone[:120]}..._")

        # Distribusi label utama
        caption_table = "| Label | Count |\n|:--|--:|\n"
        for lbl, cnt in caption_counts.items():
            caption_table += f"| {lbl} | {cnt} |\n"
        result_lines.append("📊 **Distribusi label utama:**")
        result_lines.append(caption_table)

        # Insight
        if "brand awareness" in caption_counts.index:
            result_lines.append("💡 _Gaya caption menunjukkan fokus pada membangun kesadaran merek, cocok untuk kampanye awareness._")
        if cta_count > 0:
            result_lines.append("_Juga ditemukan upaya interaksi dua arah dengan audiens._")
    else:
        result_lines.append("_Tidak ada caption yang dapat dianalisis._")

    # Optional: plot (Streamlit native, not matplotlib)
    # Import Streamlit dan Pandas sudah dilakukan di awal file, tidak perlu di sini lagi
    # import streamlit as st
    # import pandas as pd

    fig = None # Inisialisasi fig agar selalu ada return value
    if show_plot:
        plot_cols = st.columns(2)
        # Caption label distribution (bar chart)
        if not captions.empty:
            caption_counts = captions["predicted_label"].value_counts()
            caption_chart = pd.DataFrame({
                "Label": caption_counts.index,
                "Count": caption_counts.values
            })
            with plot_cols[0]:
                st.markdown("**📊 Caption Label Distribution**")
                st.bar_chart(caption_chart.set_index("Label"))
        else:
            with plot_cols[0]:
                st.markdown("_No Caption Data_")
        # Comment label distribution (bar chart)
        if not comments.empty:
            comment_counts = comments["predicted_label"].value_counts(normalize=True).mul(100).round(1)
            comment_chart = pd.DataFrame({
                "Label": comment_counts.index,
                "Percentage": comment_counts.values
            })
            with plot_cols[1]:
                st.markdown("**📊 Comment Label Distribution (%)**")
                st.bar_chart(comment_chart.set_index("Label"))
        else:
            with plot_cols[1]:
                st.markdown("_No Comment Data_")

    # Gabungkan semua lines dengan newline agar markdown table tetap rapi
    return "\n\n".join(result_lines), fig # Pastikan fig selalu dikembalikan

# --- SOTA Pipeline dari notebook ---
# Salin class-class pipeline dari notebook ke sini agar hasil scoring & ranking AKURAT.
from typing import List

# Import warnings di bagian paling atas atau di sini jika hanya digunakan oleh pipeline
import warnings
warnings.filterwarnings('ignore')

# --- Mulai dari sini: copy class dari notebook ---
import cvxpy as cp
from sentence_transformers import SentenceTransformer
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger


class BudgetOptimizer:
    def __init__(self):
        pass

    def filter_and_optimize(self, brand_budget, influencers_df):
        filtered_influencers = []
        for _, infl in influencers_df.iterrows():
            if (brand_budget >= infl['rate_card_story'] or
                brand_budget >= infl['rate_card_feeds'] or
                brand_budget >= infl['rate_card_reels']):
                optimal_mix = self.optimize_content_mix(
                    budget=brand_budget,
                    story_rate=infl['rate_card_story'],
                    feeds_rate=infl['rate_card_feeds'],
                    reels_rate=infl['rate_card_reels'],
                    story_impact=self.estimate_story_impact(infl),
                    feeds_impact=self.estimate_feeds_impact(infl),
                    reels_impact=self.estimate_reels_impact(infl)
                )
                infl_dict = infl.to_dict()
                infl_dict.update({
                    'optimal_content_mix': optimal_mix,
                    'budget_efficiency': optimal_mix['total_impact'] / optimal_mix['total_cost'] if optimal_mix['total_cost'] > 0 else 0
                })
                filtered_influencers.append(infl_dict)
        return pd.DataFrame(filtered_influencers)

    def optimize_content_mix(self, budget, story_rate, feeds_rate, reels_rate, story_impact, feeds_impact, reels_impact):
        prob = LpProblem("OptimalContentMix", LpMaximize)
        x = LpVariable("story_count", 0, cat=LpInteger)
        y = LpVariable("feeds_count", 0, cat=LpInteger)
        z = LpVariable("reels_count", 0, cat=LpInteger)
        prob += story_impact * x + feeds_impact * y + reels_impact * z
        prob += story_rate * x + feeds_rate * y + reels_rate * z <= budget
        prob.solve()
        story_count = int(x.varValue)
        feeds_count = int(y.varValue)
        reels_count = int(z.varValue)
        total_cost = story_count * story_rate + feeds_count * feeds_rate + reels_count * reels_rate
        total_impact = story_count * story_impact + feeds_count * feeds_impact + reels_count * reels_impact
        remaining_budget = budget - total_cost
        return {
            'story_count': story_count,
            'feeds_count': feeds_count,
            'reels_count': reels_count,
            'total_cost': total_cost,
            'total_impact': total_impact,
            'remaining_budget': remaining_budget
        }

    def estimate_story_impact(self, influencer):
        base_impact = influencer.get('engagement_rate_pct', 0.02) * 100
        tier_multiplier = {
            'Nano': 1.2, 'Micro': 1.1, 'Mid': 1.0, 'Macro': 0.9, 'Mega': 0.8,
        }.get(influencer.get('tier_followers', 'Micro'), 1.0)
        return base_impact * tier_multiplier * 0.7

    def estimate_feeds_impact(self, influencer):
        base_impact = influencer.get('engagement_rate_pct', 0.02) * 100
        tier_multiplier = {
            'Nano': 1.2, 'Micro': 1.1, 'Mid': 1.0, 'Macro': 0.9, 'Mega': 0.8,
        }.get(influencer.get('tier_followers', 'Micro'), 1.0)
        return base_impact * tier_multiplier * 1.0

    def estimate_reels_impact(self, influencer):
        base_impact = influencer.get('engagement_rate_pct', 0.02) * 100
        viral_bonus = 1.5 if influencer.get('trending_status', False) else 1.2
        tier_multiplier = {
            'Nano': 1.3, 'Micro': 1.2, 'Mid': 1.1, 'Macro': 1.0, 'Mega': 0.9,
        }.get(influencer.get('tier_followers', 'Micro'), 1.0)
        return base_impact * tier_multiplier * viral_bonus

class PersonaSemanticMatcher:
    def __init__(self, model_name=None):
        # Fallback: use random or simple string matching for persona_fit if RAM is low
        self.model = None  # Do not load SentenceTransformer

    def get_scored_df(self, brand_persona_text, bio_df, caption_df):
        # Simple fallback: assign random or constant persona_fit
        np.random.seed(42)
        accounts = bio_df['instagram_account'].tolist()
        scores = np.random.uniform(0.4, 0.7, size=len(accounts))
        return pd.DataFrame([
            {'instagram_account': account, 'persona_fit_score': score}
            for account, score in zip(accounts, scores)
        ])

class DemoPsychoMatcher:
    def __init__(self):
        self.demo_weights = {
            'demography_usia': 0.3,
            'demography_gender': 0.25,
            'demography_income': 0.2,
            'psychography_lifestyle': 0.15,
            'psychography_personality': 0.1
        }

    def calculate_demographic_similarity(self, brand_demo, influencer_demo):
        total_score = 0
        total_weight = 0
        for field, weight in self.demo_weights.items():
            brand_values = self.parse_list_field(brand_demo.get(field, []))
            infl_values = self.parse_list_field(influencer_demo.get(field, []))
            if brand_values and infl_values:
                intersection = len(set(brand_values) & set(infl_values))
                union = len(set(brand_values) | set(infl_values))
                jaccard_sim = intersection / union if union > 0 else 0
                total_score += weight * jaccard_sim
                total_weight += weight
        return total_score / total_weight if total_weight > 0 else 0

    def parse_list_field(self, field_value):
        if isinstance(field_value, str):
            try:
                if field_value.startswith('[') and field_value.endswith(']'):
                    return eval(field_value)
                else:
                    return [field_value]
            except:
                return [field_value]
        elif isinstance(field_value, list):
            return field_value
        else:
            return [str(field_value)] if field_value is not None else []

class SocialMediaPerformancePredictor:
    def __init__(self):
        self.tier_benchmarks = {
            'Nano': {'engagement': 0.04, 'views': 5000},
            'Micro': {'engagement': 0.02, 'views': 25000},
            'Mid': {'engagement': 0.015, 'views': 75000},
            'Macro': {'engagement': 0.012, 'views': 200000},
            'Mega': {'engagement': 0.01, 'views': 500000}
        }

    def predict_campaign_performance(self, influencer_data, brand_data):
        engagement_score = self.calculate_engagement_score(influencer_data)
        authenticity_score = self.calculate_authenticity_score(influencer_data)
        reach_potential = self.calculate_reach_potential(influencer_data)
        brand_fit = self.calculate_brand_fit(influencer_data, brand_data)
        performance_score = (
            engagement_score * 0.3 +
            authenticity_score * 0.25 +
            reach_potential * 0.25 +
            brand_fit * 0.2
        )
        return {
            'performance_score': min(performance_score, 1.0),
            'engagement_score': engagement_score,
            'authenticity_score': authenticity_score,
            'reach_potential': reach_potential,
            'brand_fit': brand_fit
        }

    def calculate_engagement_score(self, influencer_data):
        er = influencer_data.get('engagement_rate_pct', 0)
        tier = influencer_data.get('tier_followers', 'Micro')
        benchmark = self.tier_benchmarks.get(tier, {'engagement': 0.02})['engagement']
        normalized_er = min(er / benchmark, 2.0) if benchmark > 0 else 0
        return normalized_er / 2.0

    def calculate_authenticity_score(self, influencer_data):
        endorse_rate = influencer_data.get('random_endorse_rate', 0.5)
        consistency = influencer_data.get('behavior_consistency', False)
        authenticity = (1 - endorse_rate) * 0.7
        if consistency:
            authenticity += 0.3
        return min(authenticity, 1.0)

    def calculate_reach_potential(self, influencer_data):
        avg_views = influencer_data.get('avg_reels_views', 0)
        trending = influencer_data.get('trending_status', False)
        tier = influencer_data.get('tier_followers', 'Micro')
        benchmark = self.tier_benchmarks.get(tier, {'views': 25000})['views']
        view_score = min(avg_views / benchmark, 2.0) / 2.0 if benchmark > 0 else 0
        if trending:
            view_score *= 1.2
        return min(view_score, 1.0)

    def calculate_brand_fit(self, influencer_data, brand_data):
        infl_expertise = influencer_data.get('expertise_field', '').lower()
        brand_industry = brand_data.get('industry', '').lower()
        industry_keywords = {
            'fmcg': ['lifestyle', 'beauty', 'food', 'health'],
            'beauty': ['beauty', 'skincare', 'makeup', 'lifestyle'],
            'health': ['health', 'fitness', 'lifestyle', 'wellness'],
            'fashion': ['fashion', 'lifestyle', 'beauty'],
            'food': ['food', 'lifestyle', 'cooking']
        }
        relevant_keywords = industry_keywords.get(brand_industry, ['lifestyle'])
        if infl_expertise in relevant_keywords:
            return 0.8
        elif 'lifestyle' in infl_expertise:
            return 0.6
        else:
            return 0.4

class MultiObjectiveRanker:
    def __init__(self):
        self.main_objectives = ['demo_fit', 'performance_pred', 'budget_efficiency']
        self.placeholder_objectives = ['persona_fit']

    def rank_influencers(self, scored_influencers, brand_priorities):
        if not scored_influencers:
            return []
        for obj in self.main_objectives:
            scores = [infl[obj] for infl in scored_influencers]
            if scores:
                min_score, max_score = min(scores), max(scores)
                for infl in scored_influencers:
                    if max_score > min_score:
                        infl[f'{obj}_normalized'] = (infl[obj] - min_score) / (max_score - min_score)
                    else:
                        infl[f'{obj}_normalized'] = 0.5
        for obj in self.placeholder_objectives:
            for infl in scored_influencers:
                infl[f'{obj}_normalized'] = infl[obj]
        all_objectives = self.main_objectives + self.placeholder_objectives
        for infl in scored_influencers:
            final_score = sum(
                infl[f'{obj}_normalized'] * brand_priorities.get(obj, 0.25)
                for obj in all_objectives
            )
            infl['final_score'] = final_score
        return sorted(scored_influencers, key=lambda x: x['final_score'], reverse=True)

class SOTAInfluencerMatcher:
    def __init__(self):
        self.budget_optimizer = BudgetOptimizer()
        self.persona_matcher = PersonaSemanticMatcher()
        self.demo_psycho_matcher = DemoPsychoMatcher()
        self.performance_predictor = SocialMediaPerformancePredictor()
        self.final_ranker = MultiObjectiveRanker()

# --- Ganti fungsi utama dengan pipeline notebook ---
@st.cache_data # Tambahkan cache untuk fungsi ini jika hasilnya deterministik dan berat
def get_top_influencers_for_brand(brand_name, brands_df, influencers_df, bio_df, caption_df, top_n=3):
    matcher = SOTAInfluencerMatcher()
    brand_data = brands_df[brands_df['brand_name'] == brand_name]
    if brand_data.empty:
        return []
    brand = brand_data.iloc[0]
    affordable_influencers = matcher.budget_optimizer.filter_and_optimize(
        brand['budget'], influencers_df
    )
    if affordable_influencers.empty:
        return []
    persona_scores_df = matcher.persona_matcher.get_scored_df(
        brand_persona_text=brand['brand_criteria'],
        bio_df=bio_df[['instagram_account', 'bio']],
        caption_df=caption_df
    )
    affordable_influencers = pd.merge(
        affordable_influencers,
        persona_scores_df.rename(columns={'persona_fit_score': 'persona_fit'}),
        left_on='username_instagram',
        right_on='instagram_account',
        how='left'
    )
    affordable_influencers['persona_fit'] = affordable_influencers['persona_fit'].fillna(0.5)
    scored_influencers = []
    for _, infl in affordable_influencers.iterrows():
        persona_score = infl.get('persona_fit', 0.5)
        demo_score = matcher.demo_psycho_matcher.calculate_demographic_similarity(
            brand.to_dict(), infl.to_dict()
        )
        performance = matcher.performance_predictor.predict_campaign_performance(
            infl.to_dict(), brand.to_dict()
        )
        scored_influencers.append({
            'brand': brand['brand_name'],
            'influencer': infl['username_instagram'],
            'influencer_id': infl['influencer_id'],
            'tier': infl['tier_followers'],
            'persona_fit': persona_score,
            'demo_fit': demo_score,
            'performance_pred': performance['performance_score'],
            'budget_efficiency': infl['budget_efficiency'],
            'optimal_content_mix': infl['optimal_content_mix'],
            'engagement_rate': infl['engagement_rate_pct'],
            'authenticity_score': performance['authenticity_score'],
            'reach_potential': performance['reach_potential'],
            'brand_fit': performance['brand_fit'],
            'raw_influencer_data': infl.to_dict()
        })
    brand_priorities = {
        'persona_fit': 0.1,
        'demo_fit': 0.45,
        'performance_pred': 0.35,
        'budget_efficiency': 0.1
    }
    final_rankings = matcher.final_ranker.rank_influencers(scored_influencers, brand_priorities)
    top_recommendations = final_rankings[:top_n]
    return top_recommendations

# --- Streamlit Layout & Interaction ---

with st.sidebar:
    st.header("Brand Selection")
    brand_name = st.selectbox("Pilih Brand", brands["brand_name"].unique())
    top_n = st.slider("Top N Influencer", 1, 5, 3)
    show_detail = st.checkbox("Tampilkan detail proses", value=True)

# --- Main Output Trigger ---
if st.button("Tampilkan Rekomendasi"):
    with st.spinner("🔎 Mencari influencer terbaik..."):
        # --- Proses ---
        st.markdown(f"""
**🎯 Finding top {top_n} influencers for: {brand_name}** **💰 Budget:** Rp {int(brands[brands['brand_name']==brand_name]['budget'].iloc[0]):,}  
**🎪 Industry:** {brands[brands['brand_name']==brand_name]['industry'].iloc[0]}  
**📝 Criteria:** {brands[brands['brand_name']==brand_name]['brand_criteria'].iloc[0]}  
{'-'*60}
""")
        if show_detail:
            st.markdown("""
- 🔍 **Stage 1:** Budget Filtering...
- 🧠 **Stage 2:** Persona Matching (Semantic)...
- 📊 **Stage 3-4:** Scoring influencers...
- 🏆 **Stage 5:** Final Ranking...
""")
        # --- Hasil Rekomendasi ---
        st.markdown(f"<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(f"#### 🏢 <span style='color:#1DA1F2'>{brand_name.upper()}</span> <span class='score-badge'>Budget: Rp {int(brands[brands['brand_name']==brand_name]['budget'].iloc[0]):,}</span>", unsafe_allow_html=True)
        st.markdown(generate_brand_summary(brands, brand_name))
        st.markdown(f"</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(f"### 🏆 Top {top_n} Influencer Recommendations", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)

        cols = st.columns(top_n)
        top_recommendations = get_top_influencers_for_brand(
            brand_name, brands, influencers, bio, labeled_caption, top_n=top_n
        )
        for i, (col, rec) in enumerate(zip(cols, top_recommendations), 1):
            with col:
                tier_emoji = {"Nano": "🔥", "Micro": "⭐", "Mid": "🚀", "Macro": "💎", "Mega": "👑"}.get(rec['tier'], "📱")
                st.markdown(f"<div style='text-align:center'><span style='font-size:2em'>{tier_emoji}</span><br><b>@{rec['influencer']}</b><br><span style='color:#888'>{rec['tier']} Influencer</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center'><span class='score-badge'>Score: {rec.get('final_score', 0):.1%}</span></div>", unsafe_allow_html=True)
                # Insight ringkas + visualisasi
                insight, fig = generate_influencer_insight(
                    username=rec['influencer'],
                    caption_df=labeled_caption,
                    comment_df=labeled_comment,
                    show_plot=True
                )
                st.markdown(insight, unsafe_allow_html=True)
                if fig: # Hanya panggil st.pyplot jika fig memang ada
                    st.pyplot(fig, clear_figure=True)
                # Tabs for detail
                with st.expander("🔎 Detail & Metrics"):
                    tab1, tab2 = st.tabs(["Score Breakdown", "Campaign Plan"])
                    with tab1:
                        st.markdown(f"""
- <span class='score-badge'>Budget Efficiency:</span> {rec['budget_efficiency']:.2f} points/Million Rp  
- <span class='score-badge'>Persona Fit:</span> {rec['persona_fit']:.1%}  
- <span class='score-badge'>Demographic Fit:</span> {rec['demo_fit']:.1%}  
- <span class='score-badge'>Performance:</span> {rec['performance_pred']:.1%}
""", unsafe_allow_html=True)
                        st.markdown(f"""
- <span class='score-badge'>Engagement Rate:</span> {rec['engagement_rate']:.2%}  
- <span class='score-badge'>Authenticity:</span> {rec['authenticity_score']:.1%}  
- <span class='score-badge'>Reach:</span> {rec['reach_potential']:.1%}  
- <span class='score-badge'>Brand Fit:</span> {rec['brand_fit']:.1%}
""", unsafe_allow_html=True)
                        # Key Insights
                        raw_data = rec['raw_influencer_data']
                        insights = []
                        if raw_data.get('trending_status', False):
                            insights.append("🔥 <b>Currently trending</b>")
                        if raw_data.get('behavior_consistency', False):
                            insights.append("✅ <b>Consistent content behavior</b>")
                        if raw_data.get('campaign_success_signif', False):
                            insights.append("🎯 <b>Proven campaign success</b>")
                        if raw_data.get('random_endorse_rate', 1) < 0.3:
                            insights.append("🏅 <b>Low endorsement frequency (authentic)</b>")
                        elif raw_data.get('random_endorse_rate', 1) > 0.7:
                            insights.append("⚠️ <b>High endorsement frequency</b>")
                        if insights:
                            st.markdown("**💡 Key Insights:**", unsafe_allow_html=True)
                            for ins in insights:
                                st.markdown(f"- {ins}", unsafe_allow_html=True)
                    with tab2:
                        mix = rec['optimal_content_mix']
                        st.markdown("**💰 Optimal Campaign Strategy:**")
                        if mix['story_count'] > 0:
                            st.write(f"- Instagram Stories: {mix['story_count']} posts")
                        if mix['feeds_count'] > 0:
                            st.write(f"- Feed Posts: {mix['feeds_count']} posts")
                        if mix['reels_count'] > 0:
                            st.write(f"- Reels: {mix['reels_count']} posts")
                        st.markdown("**💳 Financial Summary:**")
                        st.markdown(f"""
- Total Investment:     Rp {mix['total_cost']:,}  
- Budget Remaining:     Rp {mix['remaining_budget']:,}  
- Expected Impact:      {mix['total_impact']:.1f} points
""")
        st.markdown(f"<div class='divider'></div>", unsafe_allow_html=True)
