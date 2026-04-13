"""
keiba_app.py  v2
================
Streamlit 競馬予想アプリ（毎月レース管理対応版）

【新機能】
  - 年月でレースを管理・保存
  - 月別履歴タブで過去レースを一覧表示
  - 統計タブで的中率・競馬場別成績を集計
  - 履歴のエクスポート/インポート（JSON）

【起動方法】
  streamlit run keiba_app.py
"""

import io
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

# ══════════════════════════════════════════
#  ページ設定 & CSS
# ══════════════════════════════════════════
st.set_page_config(
    page_title="KEIBA AI 予想",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1a0f; color: #fdf6e3; }
  h1,h2,h3 { color: #c9a84c !important; font-family: serif; }
  .metric-box {
    background: linear-gradient(135deg, rgba(26,77,46,0.4), rgba(13,26,15,0.8));
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 8px; padding: 16px; text-align: center; margin: 4px;
  }
  .metric-val   { font-size: 1.6rem; font-weight: bold; color: #f0cc6e; }
  .metric-label { font-size: 0.8rem; color: rgba(253,246,227,0.6); margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  セッション初期化
# ══════════════════════════════════════════
if "race_history" not in st.session_state:
    st.session_state.race_history = {}   # {year: {month: [race_dict, ...]}}


# ══════════════════════════════════════════
#  定数
# ══════════════════════════════════════════
JOCKEY_WIN_RATES = {
    "武豊":0.22, "川田将雅":0.21, "ルメール":0.25, "戸崎圭太":0.18,
    "横山武史":0.17, "横山典弘":0.15, "松山弘平":0.16, "デムーロ":0.17,
    "坂井瑠星":0.18, "藤岡佑介":0.14, "北村友一":0.13, "三浦皇成":0.12,
    "横山和生":0.15, "菱田裕二":0.10, "荻野極":0.09,
}
TRAINER_WIN_RATES = {
    "友道康夫":0.19, "中内田充正":0.22, "高野友和":0.16,
    "木村哲也":0.18, "藤原英昭":0.17, "矢作芳人":0.21,
    "上村洋行":0.14, "堀宣行":0.18,
}
TRACK_COND_MAP = {"良":0, "稍重":1, "重":2, "不良":3}
FEATURE_COLS = [
    "distance_num","course_num","track_cond_num","age",
    "weight_carried","horse_weight","weight_change",
    "odds","favorite_num","jockey_win_rate","trainer_win_rate","horse_avg_rank_last5",
]
SAMPLE_DATA = pd.DataFrame({
    "horse_name":     ["ドウデュース","プログノーシス","ローシャムパーク","ソールオリエンス","ジャックドール","ベラジオオペラ","リカンカブール","ヒシイグアス"],
    "jockey":         ["武豊","川田将雅","戸崎圭太","横山武史","藤岡佑介","横山和生","デムーロ","松山弘平"],
    "odds":           [3.2, 4.1, 5.8, 6.3, 8.5, 11.0, 28.0, 41.0],
    "age":            [5, 5, 5, 4, 6, 4, 4, 8],
    "weight_carried": [57, 57, 57, 57, 57, 57, 57, 57],
    "horse_weight":   [490, 498, 510, 476, 520, 484, 478, 502],
    "weight_change":  [2, -4, 0, 6, -2, 0, 4, -2],
    "trainer":        ["友道康夫","中内田充正","高野友和","木村哲也","藤原英昭","上村洋行","矢作芳人","堀宣行"],
})


# ══════════════════════════════════════════
#  モデル（初回のみ学習・キャッシュ）
# ══════════════════════════════════════════
@st.cache_resource
def get_model():
    np.random.seed(42)
    n = 8000
    X = pd.DataFrame({
        "distance_num":         np.random.choice([1200,1400,1600,1800,2000,2400], n),
        "course_num":           np.random.choice([0,1], n, p=[0.6,0.4]),
        "track_cond_num":       np.random.choice([0,1,2,3], n, p=[0.6,0.2,0.15,0.05]),
        "age":                  np.random.randint(3, 8, n),
        "weight_carried":       np.random.choice([54,55,56,57,58], n),
        "horse_weight":         np.random.normal(490, 20, n),
        "weight_change":        np.random.choice([-6,-4,-2,0,2,4,6], n),
        "odds":                 np.abs(np.random.exponential(8, n)) + 1.1,
        "favorite_num":         np.random.randint(1, 18, n),
        "jockey_win_rate":      np.random.uniform(0.05, 0.25, n),
        "trainer_win_rate":     np.random.uniform(0.03, 0.22, n),
        "horse_avg_rank_last5": np.random.uniform(1, 16, n),
    })
    strength = (
        -X["odds"] * 0.06 + X["jockey_win_rate"] * 4
        + X["trainer_win_rate"] * 2 - X["horse_avg_rank_last5"] * 0.1
        + np.random.randn(n) * 0.4
    )
    y = (strength > strength.quantile(0.93)).astype(int)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    m.fit(X, y)
    return m


# ══════════════════════════════════════════
#  ユーティリティ関数
# ══════════════════════════════════════════
@st.cache_data
def load_file(data: bytes, name: str) -> pd.DataFrame:
    if name.endswith(".csv"):
        for enc in ["utf-8", "utf-8-sig", "shift_jis", "cp932"]:
            try:
                return pd.read_csv(io.StringIO(data.decode(enc)))
            except Exception:
                continue
        raise ValueError("文字コードの判定に失敗しました")
    return pd.read_excel(io.BytesIO(data))


def build_features(df, distance, course, track_condition):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in ["odds","age","weight_carried","horse_weight","weight_change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["odds"]           = df.get("odds",           pd.Series([10.0]*len(df))).fillna(10.0)
    df["age"]            = df.get("age",             pd.Series([4]*len(df))).fillna(4)
    df["weight_carried"] = df.get("weight_carried",  pd.Series([57.0]*len(df))).fillna(57.0)
    df["horse_weight"]   = df.get("horse_weight",    pd.Series([490.0]*len(df))).fillna(490.0)
    df["weight_change"]  = df.get("weight_change",   pd.Series([0.0]*len(df))).fillna(0.0)
    df["jockey"]         = df.get("jockey",          pd.Series(["不明"]*len(df))).fillna("不明")
    df["trainer"]        = df.get("trainer",         pd.Series(["不明"]*len(df))).fillna("不明")
    df["distance_num"]       = distance
    df["course_num"]         = 0 if course == "芝" else 1
    df["track_cond_num"]     = TRACK_COND_MAP.get(track_condition, 0)
    df["favorite_num"]       = df["odds"].rank(method="first").astype(int)
    df["jockey_win_rate"]    = df["jockey"].map(JOCKEY_WIN_RATES).fillna(0.12)
    df["trainer_win_rate"]   = df["trainer"].map(TRAINER_WIN_RATES).fillna(0.12)
    df["horse_avg_rank_last5"] = 5.0
    return df


def predict(df, model):
    X     = df[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    total = probs.sum()
    norm  = probs / total if total > 0 else np.ones(len(probs)) / len(probs)
    mn, mx = norm.min(), norm.max()
    scores = ((norm - mn) / (mx - mn) * 75 + 20).astype(int) if mx > mn else np.full(len(norm), 50)
    df = df.copy()
    df["win_prob"] = np.round(norm * 100, 1)
    df["ai_score"] = scores
    df["ai_rank"]  = pd.Series(norm).rank(ascending=False, method="first").astype(int)
    pick_map = {1: "◎ 本命", 2: "○ 対抗", 3: "▲ 単穴"}
    df["pick"] = df["ai_rank"].map(lambda r: pick_map.get(r, ""))
    return df.sort_values("ai_rank")


def save_race(year, month, info):
    h = st.session_state.race_history
    h.setdefault(str(year), {}).setdefault(str(month), [])
    existing = h[str(year)][str(month)]
    for i, r in enumerate(existing):
        if r["race_name"] == info["race_name"] and r["date"] == info["date"]:
            existing[i] = info
            return
    existing.append(info)


def metric_box(label, val):
    return f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-val">{val}</div></div>'


# ══════════════════════════════════════════
#  サイドバー
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏇 KEIBA AI")
    tab = st.radio("メニュー", ["🔮 予想する", "📅 月別履歴", "📊 統計"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("## 🏟 レース情報")
    now = datetime.date.today()
    sel_year   = st.selectbox("年", list(range(2022, now.year + 2))[::-1])
    sel_month  = st.selectbox("月", list(range(1, 13)), index=now.month - 1)
    sel_date   = st.date_input("開催日", value=now)
    race_name  = st.text_input("レース名", value="大阪杯")
    grade      = st.selectbox("グレード", ["G1","G2","G3","OP","条件戦"])
    venue      = st.selectbox("競馬場", ["東京","中山","阪神","京都","中京","新潟","福島","札幌","函館","小倉"])
    distance   = st.selectbox("距離(m)", [1200,1400,1600,1800,2000,2200,2400,2500,3000,3200], index=4)
    course     = st.radio("コース", ["芝","ダート"], horizontal=True)
    track_cond = st.selectbox("馬場状態", ["良","稍重","重","不良"])
    st.markdown("---")
    st.markdown("## 📂 出走馬データ")
    uploaded = st.file_uploader("CSV / Excel をアップロード", type=["csv","xlsx","xls"])


# ══════════════════════════════════════════
#  タブ：🔮 予想する
# ══════════════════════════════════════════
if tab == "🔮 予想する":
    st.markdown("# 🏇 KEIBA AI 予想システム")
    st.markdown(f"### {venue}／{race_name} {grade}／{course}{distance}m／馬場:{track_cond}")
    st.markdown("---")

    if uploaded:
        try:
            raw_df = load_file(uploaded.read(), uploaded.name)
            st.success(f"✅ {uploaded.name} 読み込み完了（{len(raw_df)}頭）")
        except Exception as e:
            st.error(f"❌ {e}")
            raw_df = SAMPLE_DATA.copy()
    else:
        st.info("📂 サイドバーからCSV/Excelをアップロードしてください。今はサンプルデータで表示しています。")
        raw_df = SAMPLE_DATA.copy()

    with st.spinner("🤖 AI分析中..."):
        feat_df = build_features(raw_df, distance, course, track_cond)
        model   = get_model()
        result  = predict(feat_df, model)

    honmei = result[result["ai_rank"] == 1].iloc[0]
    taikou = result[result["ai_rank"] == 2].iloc[0]
    ana    = result[result["ai_rank"] == 3].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    for col, label, row in zip([c1,c2,c3], ["◎ 本命","○ 対抗","▲ 単穴"], [honmei,taikou,ana]):
        with col:
            st.markdown(metric_box(label, row["horse_name"]) + f'<div style="text-align:center;font-size:0.8rem;color:rgba(253,246,227,0.6)">スコア {row["ai_score"]} / オッズ {row["odds"]}</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(metric_box("出走頭数", f"{len(result)}頭"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 AI予想結果")
    dcols = ["ai_rank","pick","horse_name"] + (["jockey"] if "jockey" in result.columns else []) + ["odds","ai_score","win_prob"]
    rmap  = {"ai_rank":"AI順位","pick":"印","horse_name":"馬名","jockey":"騎手","odds":"単勝オッズ","ai_score":"AIスコア","win_prob":"勝利確率(%)"}
    st.dataframe(result.head(10)[dcols].rename(columns=rmap), use_container_width=True, hide_index=True)

    st.markdown("### 📈 AIスコア比較")
    st.bar_chart(result.head(10).set_index("horse_name")[["ai_score"]], color="#c9a84c")

    st.markdown("### 🎯 AI買い目提案")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**単勝**")
        st.markdown(f"◎ **{honmei['horse_name']}**  {honmei['odds']}倍")
    with b2:
        st.markdown("**馬連**")
        st.markdown(f"◎{honmei['horse_name']} — ○{taikou['horse_name']}")
        st.markdown(f"◎{honmei['horse_name']} — ▲{ana['horse_name']}")
    with b3:
        st.markdown("**三連複**")
        st.markdown(f"◎{honmei['horse_name']} / ○{taikou['horse_name']} / ▲{ana['horse_name']}")

    st.markdown("---")
    st.markdown("### 💾 レース結果を履歴に保存")
    with st.expander("実際の結果を入力して保存する"):
        a1   = st.text_input("1着馬名", placeholder="例: ドウデュース")
        a2   = st.text_input("2着馬名", placeholder="例: プログノーシス")
        a3   = st.text_input("3着馬名", placeholder="例: ローシャムパーク")
        memo = st.text_area("メモ（任意）", placeholder="レースの所感など")
        if st.button("💾 保存する", type="primary"):
            info = {
                "date": str(sel_date), "year": sel_year, "month": sel_month,
                "race_name": race_name, "grade": grade, "venue": venue,
                "distance": distance, "course": course, "track_cond": track_cond,
                "honmei": honmei["horse_name"], "taikou": taikou["horse_name"], "ana": ana["horse_name"],
                "actual_1st": a1, "actual_2nd": a2, "actual_3rd": a3,
                "memo": memo, "ai_hit": a1 == honmei["horse_name"],
            }
            save_race(sel_year, sel_month, info)
            st.success(f"✅ {race_name} を {sel_year}年{sel_month}月 の履歴に保存しました！")

    st.markdown("---")
    st.markdown("### 📥 入力テンプレート")

    SHEET_URL = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms/copy"

    st.markdown("""
    <div style="background:rgba(26,77,46,0.3);border:1px solid rgba(201,168,76,0.3);border-radius:8px;padding:16px;margin-bottom:12px;">
      <div style="color:#f0cc6e;font-weight:bold;margin-bottom:8px;">📋 Googleスプレッドシートの使い方</div>
      <ol style="color:rgba(253,246,227,0.8);font-size:0.85rem;line-height:2;padding-left:1.2rem;">
        <li>下のボタンでテンプレートをGoogleドライブにコピー</li>
        <li>出走馬の情報を入力する</li>
        <li>「ファイル → ダウンロード → CSV形式」で保存</li>
        <li>サイドバーからCSVをアップロード</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.link_button("🔗 スプレッドシートテンプレートを開く", SHEET_URL, use_container_width=True)
    with col_b:
        tmpl = pd.DataFrame({
            "horse_name":    ["馬名A","馬名B","馬名C"],
            "jockey":        ["騎手名","騎手名","騎手名"],
            "odds":          [3.5, 5.0, 8.0],
            "age":           [4, 5, 6],
            "weight_carried":[57, 57, 55],
            "horse_weight":  [490, 480, 470],
            "weight_change": [0, 2, -2],
            "trainer":       ["調教師名","調教師名","調教師名"],
        })
        st.download_button(
            "📄 CSVテンプレート（予備）",
            tmpl.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "keiba_template.csv", "text/csv",
            use_container_width=True,
        )

    st.markdown("""
    <div style="background:rgba(201,168,76,0.1);border:1px solid rgba(201,168,76,0.2);border-radius:6px;padding:12px;margin-top:8px;">
      <div style="color:#f0cc6e;font-size:0.82rem;font-weight:bold;">📌 入力する列の説明</div>
      <table style="width:100%;font-size:0.78rem;color:rgba(253,246,227,0.75);margin-top:8px;border-collapse:collapse;">
        <tr><td style="padding:4px 8px;color:#f0cc6e">horse_name</td><td style="padding:4px 8px">馬名（必須）</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">jockey</td><td style="padding:4px 8px">騎手名</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">odds</td><td style="padding:4px 8px">単勝オッズ（必須）例: 3.2</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">age</td><td style="padding:4px 8px">馬の年齢 例: 5</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">weight_carried</td><td style="padding:4px 8px">斤量 例: 57</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">horse_weight</td><td style="padding:4px 8px">馬体重(kg) 例: 490</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">weight_change</td><td style="padding:4px 8px">馬体重増減 例: +2 / -4</td></tr>
        <tr><td style="padding:4px 8px;color:#f0cc6e">trainer</td><td style="padding:4px 8px">調教師名</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  タブ：📅 月別履歴
# ══════════════════════════════════════════
elif tab == "📅 月別履歴":
    st.markdown("# 📅 月別レース履歴")
    st.markdown("---")
    history = st.session_state.race_history

    if not history:
        st.info("まだ保存されたレースがありません。「予想する」タブでレースを保存してください。")
    else:
        years = sorted(history.keys(), reverse=True)
        sel_y = st.selectbox("年を選択", years)
        months = sorted(history.get(sel_y, {}).keys(), key=lambda x: int(x))

        if not months:
            st.info(f"{sel_y}年のデータがありません")
        else:
            month_tabs = st.tabs([f"{m}月" for m in months])
            for tab_obj, month in zip(month_tabs, months):
                with tab_obj:
                    races = history[sel_y][month]
                    st.markdown(f"#### {sel_y}年{month}月 ── {len(races)}レース保存済み")
                    for race in sorted(races, key=lambda x: x["date"]):
                        with st.expander(f"📍 {race['date']}　{race['race_name']} [{race.get('grade','')}]　{race['venue']}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**AI予想**")
                                st.write(f"◎ {race['honmei']}")
                                st.write(f"○ {race['taikou']}")
                                st.write(f"▲ {race['ana']}")
                            with col2:
                                st.markdown("**実際の結果**")
                                st.write(f"1着: {race.get('actual_1st','未入力') or '未入力'}")
                                st.write(f"2着: {race.get('actual_2nd','未入力') or '未入力'}")
                                st.write(f"3着: {race.get('actual_3rd','未入力') or '未入力'}")
                            with col3:
                                st.markdown("**レース情報**")
                                st.write(f"{race['course']}{race['distance']}m / {race['track_cond']}")
                                st.write(f"AI本命: {'✅ 的中' if race.get('ai_hit') else '❌ 外れ'}")
                                if race.get("memo"):
                                    st.caption(f"📝 {race['memo']}")

        st.markdown("---")
        json_str = json.dumps(history, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 履歴をエクスポート（JSON）",
            json_str.encode("utf-8"),
            f"keiba_history_{datetime.date.today()}.json",
            "application/json",
        )

    st.markdown("### 📤 過去の履歴をインポート")
    imp = st.file_uploader("JSONファイルをアップロード", type=["json"], key="import")
    if imp:
        try:
            imported = json.loads(imp.read().decode("utf-8"))
            for year, md in imported.items():
                for month, races in md.items():
                    for race in races:
                        save_race(int(year), int(month), race)
            st.success("✅ インポート完了！")
            st.rerun()
        except Exception as e:
            st.error(f"インポートエラー: {e}")


# ══════════════════════════════════════════
#  タブ：📊 統計
# ══════════════════════════════════════════
elif tab == "📊 統計":
    st.markdown("# 📊 AI予想統計")
    st.markdown("---")
    history   = st.session_state.race_history
    all_races = [r for yd in history.values() for races in yd.values() for r in races]

    if not all_races:
        st.info("まだ保存されたレースがありません。「予想する」タブでレースを保存してください。")
    else:
        total    = len(all_races)
        entered  = sum(1 for r in all_races if r.get("actual_1st", ""))
        hits     = sum(1 for r in all_races if r.get("ai_hit", False))
        hit_rate = (hits / entered * 100) if entered > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1,c2,c3,c4],
            ["総レース数","結果入力済","本命的中","的中率"],
            [f"{total}R", f"{entered}R", f"{hits}R", f"{hit_rate:.1f}%"],
        ):
            with col:
                st.markdown(metric_box(label, val), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📅 月別集計")
        monthly = {}
        for r in all_races:
            key = f"{r['year']}年{r['month']}月"
            monthly.setdefault(key, {"レース数":0, "結果入力":0, "的中":0})
            monthly[key]["レース数"] += 1
            if r.get("actual_1st", ""):
                monthly[key]["結果入力"] += 1
                if r.get("ai_hit", False):
                    monthly[key]["的中"] += 1
        mdf = pd.DataFrame(monthly).T.reset_index().rename(columns={"index":"月"})
        mdf["的中率(%)"] = (mdf["的中"] / mdf["結果入力"].replace(0, 1) * 100).round(1)
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        st.markdown("### 🏟 競馬場別集計")
        vd = {}
        for r in all_races:
            v = r.get("venue", "不明")
            vd.setdefault(v, {"レース数":0, "的中":0})
            vd[v]["レース数"] += 1
            if r.get("ai_hit", False):
                vd[v]["的中"] += 1
        vdf = pd.DataFrame(vd).T.reset_index().rename(columns={"index":"競馬場"})
        vdf["的中率(%)"] = (vdf["的中"] / vdf["レース数"] * 100).round(1)
        st.dataframe(vdf, use_container_width=True, hide_index=True)

        st.markdown("### 🏆 グレード別集計")
        gd = {}
        for r in all_races:
            g = r.get("grade", "不明")
            gd.setdefault(g, {"レース数":0, "的中":0})
            gd[g]["レース数"] += 1
            if r.get("ai_hit", False):
                gd[g]["的中"] += 1
        gdf = pd.DataFrame(gd).T.reset_index().rename(columns={"index":"グレード"})
        gdf["的中率(%)"] = (gdf["的中"] / gdf["レース数"] * 100).round(1)
        st.dataframe(gdf, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("※ 本アプリの予想はAIによる参考情報です。馬券購入は自己責任でお願いします。")
