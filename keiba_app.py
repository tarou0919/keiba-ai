"""
keiba_app.py
============
Streamlit 競馬予想アプリ
CSV / Excel ファイルを読み込んでAI予想を実行する

【インストール】
  pip install streamlit lightgbm scikit-learn pandas numpy openpyxl xlrd

【起動方法】
  streamlit run keiba_app.py

【対応フォーマット】
  CSV / Excel (.xlsx / .xls)

【必須列】（最低限これがあれば動きます）
  horse_name  : 馬名
  odds        : 単勝オッズ
  
【あると予想精度が上がる列】
  jockey          : 騎手名
  age             : 年齢
  weight_carried  : 斤量
  horse_weight    : 馬体重
  weight_change   : 馬体重増減
  distance        : 距離(m)
  course          : 芝 or ダート
  track_condition : 馬場状態（良/稍重/重/不良）
  trainer         : 調教師
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

# ══════════════════════════════════════════
#  ページ設定
# ══════════════════════════════════════════
st.set_page_config(
    page_title="KEIBA AI 予想",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── カスタムCSS ──
st.markdown("""
<style>
  .main { background-color: #0d1a0f; }
  .stApp { background-color: #0d1a0f; color: #fdf6e3; }
  h1, h2, h3 { color: #c9a84c !important; font-family: serif; }
  .metric-box {
    background: linear-gradient(135deg, rgba(26,77,46,0.4), rgba(13,26,15,0.8));
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 8px; padding: 16px; text-align: center; margin: 4px;
  }
  .metric-val { font-size: 1.8rem; font-weight: bold; color: #f0cc6e; }
  .metric-label { font-size: 0.8rem; color: rgba(253,246,227,0.6); margin-top: 4px; }
  .pick-honmei { background:#c9a84c; color:#0d1a0f; padding:2px 8px; border-radius:3px; font-weight:bold; font-size:0.85rem; }
  .pick-taikou { background:rgba(201,168,76,0.2); color:#c9a84c; border:1px solid #c9a84c; padding:2px 8px; border-radius:3px; font-size:0.85rem; }
  .pick-ana    { background:rgba(255,255,255,0.1); color:rgba(253,246,227,0.7); padding:2px 8px; border-radius:3px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  サイドバー：レース情報入力
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏟 レース情報")
    race_name      = st.text_input("レース名", value="大阪杯")
    distance       = st.selectbox("距離 (m)", [1200,1400,1600,1800,2000,2200,2400,2500,3000,3200], index=4)
    course         = st.radio("コース", ["芝", "ダート"], horizontal=True)
    track_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])
    venue          = st.selectbox("競馬場", ["東京","中山","阪神","京都","中京","新潟","福島","札幌","函館","小倉"])

    st.markdown("---")
    st.markdown("## 📂 データ読み込み")
    uploaded = st.file_uploader(
        "CSV / Excel をアップロード",
        type=["csv", "xlsx", "xls"],
        help="出走馬のデータファイルを選択してください"
    )

    st.markdown("---")
    st.markdown("## ⚙️ 予想設定")
    top_n = st.slider("表示頭数", min_value=3, max_value=18, value=8)
    show_prob = st.toggle("勝利確率を表示", value=True)
    show_detail = st.toggle("詳細データを表示", value=False)


# ══════════════════════════════════════════
#  サンプルデータ定義
# ══════════════════════════════════════════
SAMPLE_DATA = pd.DataFrame({
    "horse_name":      ["ドウデュース","プログノーシス","ローシャムパーク","ソールオリエンス",
                        "ジャックドール","ベラジオオペラ","リカンカブール","ヒシイグアス"],
    "jockey":          ["武豊","川田将雅","戸崎圭太","横山武史","藤岡佑介","横山和生","デムーロ","松山弘平"],
    "odds":            [3.2, 4.1, 5.8, 6.3, 8.5, 11.0, 28.0, 41.0],
    "age":             [5, 5, 5, 4, 6, 4, 4, 8],
    "weight_carried":  [57, 57, 57, 57, 57, 57, 57, 57],
    "horse_weight":    [490, 498, 510, 476, 520, 484, 478, 502],
    "weight_change":   [2, -4, 0, 6, -2, 0, 4, -2],
    "trainer":         ["友道康夫","中内田充正","高野友和","木村哲也","藤原英昭","上村洋行","矢作芳人","堀宣行"],
})

JOCKEY_WIN_RATES = {
    "武豊":0.22,"川田将雅":0.21,"ルメール":0.25,"戸崎圭太":0.18,
    "横山武史":0.17,"横山典弘":0.15,"松山弘平":0.16,"デムーロ":0.17,
    "坂井瑠星":0.18,"藤岡佑介":0.14,"北村友一":0.13,"三浦皇成":0.12,
    "横山和生":0.15,"菱田裕二":0.10,"荻野極":0.09,
}
TRAINER_WIN_RATES = {
    "友道康夫":0.19,"中内田充正":0.22,"高野友和":0.16,
    "木村哲也":0.18,"藤原英昭":0.17,"矢作芳人":0.21,
    "上村洋行":0.14,"堀宣行":0.18,
}
TRACK_COND_MAP = {"良":0,"稍重":1,"重":2,"不良":3}


# ══════════════════════════════════════════
#  データ読み込み・前処理
# ══════════════════════════════════════════
@st.cache_data
def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # 文字コード自動判定
        raw = uploaded_file.read()
        for enc in ["utf-8", "utf-8-sig", "shift_jis", "cp932"]:
            try:
                df = pd.read_csv(io.StringIO(raw.decode(enc)))
                return df
            except Exception:
                continue
        raise ValueError("文字コードの判定に失敗しました")
    else:
        return pd.read_excel(uploaded_file)


def build_features(df: pd.DataFrame, distance: int, course: str, track_condition: str) -> pd.DataFrame:
    df = df.copy()

    # 列名を小文字・スペース除去に正規化
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 必須列チェック
    if "horse_name" not in df.columns:
        # 馬名っぽい列を自動検出
        for col in df.columns:
            if "馬" in col or "name" in col.lower():
                df = df.rename(columns={col: "horse_name"})
                break
    if "odds" not in df.columns:
        for col in df.columns:
            if "オッズ" in col or "odds" in col.lower() or "単勝" in col:
                df = df.rename(columns={col: "odds"})
                break

    # 数値変換
    for col in ["odds", "age", "weight_carried", "horse_weight", "weight_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # デフォルト値で補完
    df["odds"]           = df.get("odds",           pd.Series([10.0]*len(df))).fillna(10.0)
    df["age"]            = df.get("age",             pd.Series([4]*len(df))).fillna(4)
    df["weight_carried"] = df.get("weight_carried",  pd.Series([57.0]*len(df))).fillna(57.0)
    df["horse_weight"]   = df.get("horse_weight",    pd.Series([490.0]*len(df))).fillna(490.0)
    df["weight_change"]  = df.get("weight_change",   pd.Series([0.0]*len(df))).fillna(0.0)
    df["jockey"]         = df.get("jockey",          pd.Series(["不明"]*len(df))).fillna("不明")
    df["trainer"]        = df.get("trainer",         pd.Series(["不明"]*len(df))).fillna("不明")

    # 特徴量生成
    df["distance_num"]       = distance
    df["course_num"]         = 0 if course == "芝" else 1
    df["track_cond_num"]     = TRACK_COND_MAP.get(track_condition, 0)
    df["favorite_num"]       = df["odds"].rank(method="first").astype(int)
    df["jockey_win_rate"]    = df["jockey"].map(JOCKEY_WIN_RATES).fillna(0.12)
    df["trainer_win_rate"]   = df["trainer"].map(TRAINER_WIN_RATES).fillna(0.12)
    df["horse_avg_rank_last5"] = 5.0  # 本番はDB参照

    return df


# ══════════════════════════════════════════
#  デモモデル生成（学習済みモデルがない場合）
# ══════════════════════════════════════════
@st.cache_resource
def get_model():
    np.random.seed(42)
    n = 8000
    X = pd.DataFrame({
        "distance_num":         np.random.choice([1200,1400,1600,1800,2000,2400], n),
        "course_num":           np.random.choice([0,1], n, p=[0.6,0.4]),
        "track_cond_num":       np.random.choice([0,1,2,3], n, p=[0.6,0.2,0.15,0.05]),
        "age":                  np.random.randint(3,8,n),
        "weight_carried":       np.random.choice([54,55,56,57,58], n),
        "horse_weight":         np.random.normal(490,20,n),
        "weight_change":        np.random.choice([-6,-4,-2,0,2,4,6], n),
        "odds":                 np.abs(np.random.exponential(8,n))+1.1,
        "favorite_num":         np.random.randint(1,18,n),
        "jockey_win_rate":      np.random.uniform(0.05,0.25,n),
        "trainer_win_rate":     np.random.uniform(0.03,0.22,n),
        "horse_avg_rank_last5": np.random.uniform(1,16,n),
    })
    strength = (
        - X["odds"] * 0.06
        + X["jockey_win_rate"] * 4
        + X["trainer_win_rate"] * 2
        - X["horse_avg_rank_last5"] * 0.1
        + np.random.randn(n) * 0.4
    )
    y = (strength > strength.quantile(0.93)).astype(int)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    model.fit(X, y)
    return model


FEATURE_COLS = [
    "distance_num","course_num","track_cond_num","age",
    "weight_carried","horse_weight","weight_change",
    "odds","favorite_num","jockey_win_rate","trainer_win_rate","horse_avg_rank_last5",
]


# ══════════════════════════════════════════
#  予想実行
# ══════════════════════════════════════════
def predict(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    total = probs.sum()
    norm  = probs / total if total > 0 else np.ones(len(probs)) / len(probs)

    # AIスコア（20〜99）
    mn, mx = norm.min(), norm.max()
    scores = ((norm - mn) / (mx - mn) * 75 + 20).astype(int) if mx > mn else np.full(len(norm), 50)

    df = df.copy()
    df["win_prob"]  = np.round(norm * 100, 1)
    df["ai_score"]  = scores
    df["ai_rank"]   = pd.Series(norm).rank(ascending=False, method="first").astype(int)

    pick_map = {1:"◎ 本命", 2:"○ 対抗", 3:"▲ 単穴"}
    df["pick"] = df["ai_rank"].map(lambda r: pick_map.get(r, ""))

    return df.sort_values("ai_rank")


# ══════════════════════════════════════════
#  メイン画面
# ══════════════════════════════════════════
st.markdown("# 🏇 KEIBA AI 予想システム")
st.markdown(f"### {venue} / {race_name} ／ {course}{distance}m ／ 馬場:{track_condition}")
st.markdown("---")

# データ準備
if uploaded is not None:
    try:
        raw_df = load_file(uploaded)
        st.success(f"✅ {uploaded.name} を読み込みました（{len(raw_df)}頭）")
        if show_detail:
            with st.expander("📋 読み込んだ生データ"):
                st.dataframe(raw_df, use_container_width=True)
    except Exception as e:
        st.error(f"❌ ファイル読み込みエラー: {e}")
        raw_df = SAMPLE_DATA.copy()
else:
    st.info("📂 サイドバーからCSV/Excelをアップロードしてください。今はサンプルデータで表示しています。")
    raw_df = SAMPLE_DATA.copy()

# 特徴量生成 & 予測
with st.spinner("🤖 AI分析中..."):
    try:
        feat_df = build_features(raw_df, distance, course, track_condition)
        model   = get_model()
        result  = predict(feat_df, model)
    except Exception as e:
        st.error(f"予測エラー: {e}")
        st.stop()

result_top = result.head(top_n)

# ── サマリーメトリクス ──
col1, col2, col3, col4 = st.columns(4)
honmei = result[result["ai_rank"] == 1].iloc[0]
taikou = result[result["ai_rank"] == 2].iloc[0]
ana    = result[result["ai_rank"] == 3].iloc[0]

with col1:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">◎ 本命</div>
        <div class="metric-val">{honmei['horse_name']}</div>
        <div class="metric-label">AIスコア {honmei['ai_score']} / オッズ {honmei['odds']}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">○ 対抗</div>
        <div class="metric-val">{taikou['horse_name']}</div>
        <div class="metric-label">AIスコア {taikou['ai_score']} / オッズ {taikou['odds']}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">▲ 単穴</div>
        <div class="metric-val">{ana['horse_name']}</div>
        <div class="metric-label">AIスコア {ana['ai_score']} / オッズ {ana['odds']}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">出走頭数</div>
        <div class="metric-val">{len(result)}頭</div>
        <div class="metric-label">分析完了</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── 予想テーブル ──
st.markdown("### 📊 AI予想結果")

display_cols = ["ai_rank", "pick", "horse_name"]
if "jockey" in result.columns:    display_cols.append("jockey")
display_cols.append("odds")
if "horse_weight" in result.columns: display_cols.append("horse_weight")
if "weight_change" in result.columns: display_cols.append("weight_change")
display_cols.append("ai_score")
if show_prob: display_cols.append("win_prob")

rename_map = {
    "ai_rank":"AI順位","pick":"印","horse_name":"馬名","jockey":"騎手",
    "odds":"単勝オッズ","horse_weight":"馬体重","weight_change":"増減",
    "ai_score":"AIスコア","win_prob":"勝利確率(%)",
}

show_df = result_top[display_cols].rename(columns=rename_map)

# スタイリング
def style_row(row):
    rank = row.get("AI順位", 99)
    if rank == 1:   return ["background-color: rgba(201,168,76,0.15)"] * len(row)
    elif rank == 2: return ["background-color: rgba(201,168,76,0.08)"] * len(row)
    else:           return [""] * len(row)

styled = show_df.style.apply(style_row, axis=1).format({
    "単勝オッズ": "{:.1f}",
    "AIスコア": "{}",
    "勝利確率(%)": "{:.1f}%",
})

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── AIスコアバーチャート ──
st.markdown("### 📈 AIスコア比較")
chart_df = result_top[["horse_name", "ai_score"]].set_index("horse_name")
st.bar_chart(chart_df, color="#c9a84c")

# ── 買い目提案 ──
st.markdown("### 🎯 AI買い目提案")
b1, b2, b3 = st.columns(3)

h1_name = result[result["ai_rank"]==1]["horse_name"].values[0]
h2_name = result[result["ai_rank"]==2]["horse_name"].values[0]
h3_name = result[result["ai_rank"]==3]["horse_name"].values[0]
h1_odds = result[result["ai_rank"]==1]["odds"].values[0]
h2_odds = result[result["ai_rank"]==2]["odds"].values[0]

with b1:
    st.markdown("**単勝**")
    st.markdown(f"◎ **{h1_name}**")
    st.caption(f"オッズ {h1_odds} 倍")

with b2:
    st.markdown("**馬連**")
    st.markdown(f"◎{h1_name} — ○{h2_name}")
    st.caption(f"◎{h1_name} — ▲{h3_name}")

with b3:
    st.markdown("**三連複**")
    st.markdown(f"◎{h1_name} / ○{h2_name} / ▲{h3_name}")
    st.caption("本線フォーメーション")

st.markdown("---")

# ── CSVサンプルダウンロード ──
st.markdown("### 📥 入力テンプレート")
st.caption("このCSVをダウンロードして馬の情報を入力し、アップロードしてください")
template = pd.DataFrame({
    "horse_name":     ["馬名A","馬名B","馬名C"],
    "jockey":         ["騎手名","騎手名","騎手名"],
    "odds":           [3.5, 5.0, 8.0],
    "age":            [4, 5, 6],
    "weight_carried": [57, 57, 55],
    "horse_weight":   [490, 480, 470],
    "weight_change":  [0, 2, -2],
    "trainer":        ["調教師名","調教師名","調教師名"],
})
csv_bytes = template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("📄 入力テンプレートをダウンロード", csv_bytes, "keiba_template.csv", "text/csv")

st.markdown("---")
st.caption("※ 本アプリの予想はAIによる参考情報です。馬券購入は自己責任でお願いします。")
