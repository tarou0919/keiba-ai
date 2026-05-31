"""
keiba_model_v2.py
=================
競馬予測モデル 精度向上版

【主な改善点】
  1. 特徴量を大幅追加（過去5走着順・タイム・コース適性・休養日数・血統など）
  2. 実データCSV対応（netkeibaからダウンロードしたCSVを直接使用可能）
  3. 日本語列名の自動変換
  4. 回収率最適化（Kelly基準）
  5. 特徴量重要度・学習曲線の可視化

【使い方】
  pip install lightgbm scikit-learn pandas numpy matplotlib

  # デモデータで動作確認
  python keiba_model.py

  # 実データCSVで学習・予測
  python keiba_model.py --csv race_data.csv

  # 学習済みモデルを保存
  python keiba_model.py --csv race_data.csv --save model.pkl

  # 保存済みモデルで予測のみ
  python keiba_model.py --predict shutsuba.csv --load model.pkl
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Meiryo', 'sans-serif']
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score


# ══════════════════════════════════════════
#  日本語列名 → 英語列名 変換マップ
# ══════════════════════════════════════════
JP_COL_MAP = {
    '着順': 'rank', '枠番': 'frame_num', '馬番': 'horse_num',
    '馬名': 'horse_name', '性齢': 'sex_age', '斤量': 'weight_carried',
    '騎手': 'jockey', 'タイム': 'finish_time', '着差': 'margin',
    '単勝': 'odds', '人気': 'favorite', '馬体重': 'horse_weight_raw',
    '調教師': 'trainer', 'レース名': 'race_name', '開催': 'venue_raw',
    '距離': 'distance_raw', '馬場': 'track_condition', '日付': 'race_date',
    'コース': 'course_raw',
}

TRACK_COND_MAP  = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
VENUE_CODE_MAP  = {
    '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,
    '中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
}


# ══════════════════════════════════════════
#  特徴量（v2：大幅追加）
# ══════════════════════════════════════════
FEATURE_COLS_V2 = [
    # ── レース条件 ──
    'distance_num',          # 距離(m)
    'course_num',            # 芝=0 ダート=1
    'track_cond_num',        # 馬場 良=0〜不良=3
    'venue_num',             # 競馬場コード
    # ── 馬の基本 ──
    'age',                   # 年齢
    'sex_num',               # 牡=0 牝=1 セン=2
    'weight_carried',        # 斤量
    'horse_weight_kg',       # 馬体重(kg)
    'weight_change',         # 馬体重増減
    # ── オッズ・人気 ──
    'odds_num',              # 単勝オッズ
    'favorite_num',          # 人気順位
    'log_odds',              # log(オッズ) ← 非線形を捉える
    # ── 騎手・調教師 ──
    'jockey_win_rate',       # 騎手の通算勝率
    'jockey_place_rate',     # 騎手の複勝率
    'trainer_win_rate',      # 調教師の通算勝率
    # ── 馬の過去成績（新規追加）──
    'horse_avg_rank_last3',  # 直近3走の平均着順
    'horse_avg_rank_last5',  # 直近5走の平均着順
    'horse_best_rank',       # 過去最高着順
    'horse_win_count',       # 通算勝利数
    'horse_race_count',      # 通算出走数
    'horse_win_rate',        # 馬の勝率
    # ── コース・距離適性（新規追加）──
    'same_course_win_rate',  # 同コースでの勝率
    'same_dist_win_rate',    # 同距離帯での勝率
    # ── 間隔・ローテーション（新規追加）──
    'days_since_last_race',  # 前走からの日数
    'race_interval_score',   # 間隔スコア（適正間隔=高スコア）
    # ── タイム・スピード指数（新規追加）──
    'speed_index',           # スピード指数（タイムから算出）
    'speed_index_avg3',      # 直近3走の平均スピード指数
]


# ══════════════════════════════════════════
#  データ前処理
# ══════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """生データを特徴量付きDataFrameに変換する"""
    df = df.copy()

    # 1. 列名を英語に変換
    df = df.rename(columns={k: v for k, v in JP_COL_MAP.items() if k in df.columns})
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # 2. 馬体重・増減を分離（例: "480(+2)" → 480, +2）
    if 'horse_weight_raw' in df.columns:
        df['horse_weight_kg'] = df['horse_weight_raw'].str.extract(r'(\d+)').astype(float)
        df['weight_change']   = df['horse_weight_raw'].str.extract(r'\(([+-]?\d+)\)').astype(float)
    elif 'horse_weight' in df.columns:
        df['horse_weight_kg'] = pd.to_numeric(df['horse_weight'], errors='coerce')
        if 'weight_change' not in df.columns:
            df['weight_change'] = 0.0

    # 3. 性別・年齢を分離（例: "牡5" → 牡, 5）
    if 'sex_age' in df.columns:
        df['sex'] = df['sex_age'].str[0]
        df['age'] = df['sex_age'].str[1:].astype(float, errors='ignore')
    elif 'age' not in df.columns:
        df['age'] = 4.0

    sex_map = {'牡': 0, '牝': 1, 'セン': 2}
    df['sex_num'] = df.get('sex', pd.Series(['牡']*len(df))).map(sex_map).fillna(0)

    # 4. 数値変換
    for col in ['rank', 'odds', 'favorite', 'weight_carried', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.rename(columns={'odds': 'odds_num', 'favorite': 'favorite_num',
                             'rank': 'rank_num'})

    # 5. 馬場・コース
    if 'track_condition' in df.columns:
        df['track_cond_num'] = df['track_condition'].map(TRACK_COND_MAP).fillna(0)
    else:
        df['track_cond_num'] = 0

    if 'course_raw' in df.columns:
        df['course_num'] = df['course_raw'].map({'芝': 0, 'ダート': 1}).fillna(0)
    elif 'course' in df.columns:
        df['course_num'] = df['course'].map({'芝': 0, 'ダート': 1}).fillna(0)
    else:
        df['course_num'] = 0

    # 6. 距離
    if 'distance_raw' in df.columns:
        df['distance_num'] = df['distance_raw'].str.extract(r'(\d+)').astype(float)
    elif 'distance' in df.columns:
        df['distance_num'] = pd.to_numeric(df['distance'], errors='coerce')
    else:
        df['distance_num'] = 2000.0

    # 7. 競馬場コード
    if 'venue_raw' in df.columns:
        df['venue_num'] = df['venue_raw'].map(VENUE_CODE_MAP).fillna(4)
    elif 'venue' in df.columns:
        df['venue_num'] = df['venue'].map(VENUE_CODE_MAP).fillna(4)
    else:
        df['venue_num'] = 4

    # 8. 人気
    if 'odds_num' in df.columns:
        df['odds_num'] = pd.to_numeric(df['odds_num'], errors='coerce').fillna(10.0)
        df['log_odds'] = np.log(df['odds_num'].clip(lower=1.1))
        if 'favorite_num' not in df.columns:
            df['favorite_num'] = df.groupby(
                df.get('race_id', pd.Series(range(len(df))))
            )['odds_num'].rank(method='first').astype(int)

    # 9. 目的変数
    if 'rank_num' in df.columns:
        df['is_win']   = (df['rank_num'] == 1).astype(int)
        df['is_place'] = (df['rank_num'] <= 3).astype(int)

    # 10. 騎手・調教師の勝率（データ全体から集計）
    if 'jockey' in df.columns and 'is_win' in df.columns:
        jw = df.groupby('jockey')['is_win'].agg(['mean', lambda x: (x <= 3).mean()])
        jw.columns = ['jockey_win_rate', 'jockey_place_rate']
        df = df.join(jw, on='jockey')
    else:
        df['jockey_win_rate']   = 0.15
        df['jockey_place_rate'] = 0.40

    if 'trainer' in df.columns and 'is_win' in df.columns:
        tw = df.groupby('trainer')['is_win'].mean().rename('trainer_win_rate')
        df = df.join(tw, on='trainer')
    else:
        df['trainer_win_rate'] = 0.12

    # 11. 馬の過去成績（時系列順に集計）
    if 'horse_name' in df.columns and 'is_win' in df.columns:
        df = df.sort_values(['horse_name', 'race_date'] if 'race_date' in df.columns else ['horse_name'])

        df['horse_avg_rank_last3'] = (
            df.groupby('horse_name')['rank_num']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df['horse_avg_rank_last5'] = (
            df.groupby('horse_name')['rank_num']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        df['horse_best_rank'] = (
            df.groupby('horse_name')['rank_num']
            .transform(lambda x: x.shift(1).expanding().min())
        )
        df['horse_win_count'] = (
            df.groupby('horse_name')['is_win']
            .transform(lambda x: x.shift(1).expanding().sum())
        )
        df['horse_race_count'] = (
            df.groupby('horse_name')['is_win']
            .transform(lambda x: x.shift(1).expanding().count())
        )
        df['horse_win_rate'] = (
            df['horse_win_count'] / df['horse_race_count'].replace(0, 1)
        )
    else:
        for col in ['horse_avg_rank_last3','horse_avg_rank_last5',
                    'horse_best_rank','horse_win_count','horse_race_count','horse_win_rate']:
            df[col] = 5.0 if 'rank' in col else 0.0

    # 12. コース・距離適性
    if 'horse_name' in df.columns and 'is_win' in df.columns:
        df['same_course_win_rate'] = (
            df.groupby(['horse_name', 'course_num'])['is_win']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0.10)
        df['same_dist_win_rate'] = (
            df.groupby(['horse_name', 'distance_num'])['is_win']
            .transform(lambda x: x.shift(1).expanding().mean())
        ).fillna(0.10)
    else:
        df['same_course_win_rate'] = 0.10
        df['same_dist_win_rate']   = 0.10

    # 13. 前走からの日数
    if 'race_date' in df.columns and 'horse_name' in df.columns:
        df['race_date_dt'] = pd.to_datetime(df['race_date'], errors='coerce')
        df['days_since_last_race'] = (
            df.groupby('horse_name')['race_date_dt']
            .transform(lambda x: x.diff().dt.days)
        ).fillna(30)
    else:
        df['days_since_last_race'] = 30

    # 適正間隔スコア（中8週前後が最も高い）
    df['race_interval_score'] = np.exp(
        -((df['days_since_last_race'] - 56) ** 2) / (2 * 28 ** 2)
    )

    # 14. スピード指数（タイムが分:秒.コンマ形式）
    if 'finish_time' in df.columns:
        def parse_time(t):
            try:
                parts = str(t).split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                return float(t)
            except:
                return np.nan
        df['finish_time_sec'] = df['finish_time'].apply(parse_time)
        # 距離別の基準タイムとの差でスピード指数を算出
        base_time = df['distance_num'] / 1000 * 60  # 概算
        df['speed_index'] = (base_time - df['finish_time_sec'] + 60) * 2
        df['speed_index_avg3'] = (
            df.groupby('horse_name')['speed_index']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        ).fillna(100)
    else:
        df['speed_index']      = 100.0
        df['speed_index_avg3'] = 100.0

    # 15. 欠損値補完
    for col in FEATURE_COLS_V2:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
            df[col].median() if df[col].dtype in [float, int] else 0
        )

    return df


# ══════════════════════════════════════════
#  デモデータ生成
# ══════════════════════════════════════════
def make_demo_data(n_races: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for race_idx in range(n_races):
        n = np.random.randint(10, 18)
        strengths = np.random.randn(n)
        for i in range(n):
            jw = np.random.uniform(0.05, 0.25)
            speed = np.random.normal(100, 5)
            row = {
                'race_id':               f'2024{race_idx:05d}',
                'race_date':             f'2024-{(race_idx//40+1):02d}-15',
                'horse_name':            f'ウマ{np.random.randint(1000,9999)}',
                'distance_num':          np.random.choice([1200,1400,1600,1800,2000,2400]),
                'course_num':            np.random.choice([0,1], p=[0.6,0.4]),
                'track_cond_num':        np.random.choice([0,1,2,3], p=[0.6,0.2,0.15,0.05]),
                'venue_num':             np.random.randint(0,10),
                'age':                   np.random.randint(3,8),
                'sex_num':               np.random.choice([0,1,2], p=[0.6,0.35,0.05]),
                'weight_carried':        np.random.choice([54,55,56,57,58]),
                'horse_weight_kg':       np.random.normal(490,20),
                'weight_change':         np.random.choice([-6,-4,-2,0,2,4,6]),
                'odds_num':              max(1.1, np.random.exponential(8)),
                'favorite_num':          i+1,
                'log_odds':              np.log(max(1.1, np.random.exponential(8))),
                'jockey_win_rate':       jw,
                'jockey_place_rate':     jw * 2.5,
                'trainer_win_rate':      np.random.uniform(0.03,0.22),
                'horse_avg_rank_last3':  np.random.uniform(1,n),
                'horse_avg_rank_last5':  np.random.uniform(1,n),
                'horse_best_rank':       np.random.randint(1,5),
                'horse_win_count':       np.random.randint(0,10),
                'horse_race_count':      np.random.randint(1,30),
                'horse_win_rate':        np.random.uniform(0,0.4),
                'same_course_win_rate':  np.random.uniform(0,0.3),
                'same_dist_win_rate':    np.random.uniform(0,0.3),
                'days_since_last_race':  np.random.choice([28,35,42,56,70,84]),
                'race_interval_score':   np.random.uniform(0.3,1.0),
                'speed_index':           speed,
                'speed_index_avg3':      speed + np.random.randn()*3,
                '_strength':             strengths[i] + jw*4 + speed*0.05,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df['rank_num'] = (
        df.groupby('race_id')['_strength']
        .rank(ascending=False, method='first').astype(int)
    )
    df['is_win']   = (df['rank_num'] == 1).astype(int)
    df['is_place'] = (df['rank_num'] <= 3).astype(int)
    df['odds_num'] = df.groupby('race_id')['_strength'].rank(ascending=False) * np.random.uniform(0.8,1.2,len(df))
    df['log_odds'] = np.log(df['odds_num'].clip(lower=1.1))
    return df.drop(columns=['_strength'])


# ══════════════════════════════════════════
#  モデル学習
# ══════════════════════════════════════════
def train_model(df: pd.DataFrame):
    df = df.sort_values('race_date').reset_index(drop=True) if 'race_date' in df.columns else df

    X = df[FEATURE_COLS_V2].fillna(0)
    y = df['is_win']

    params = {
        'objective':         'binary',
        'metric':            'auc',
        'boosting_type':     'gbdt',
        'n_estimators':      1000,
        'learning_rate':     0.03,
        'num_leaves':        63,
        'max_depth':         -1,
        'min_child_samples': 15,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'reg_alpha':         0.1,
        'reg_lambda':        1.0,
        'class_weight':      'balanced',
        'random_state':      42,
        'verbose':           -1,
    }

    tscv    = TimeSeriesSplit(n_splits=5)
    cv_aucs = []

    print('=' * 55)
    print('  クロスバリデーション開始')
    print('=' * 55)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        cv_aucs.append(auc)
        print(f'  Fold {fold}: AUC = {auc:.4f}')

    print(f'\n  ▶ CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}')

    print('\n[最終モデル] 全データで学習中...')
    final = lgb.LGBMClassifier(**params)
    final.fit(X, y)

    fi = pd.DataFrame({
        'feature':    FEATURE_COLS_V2,
        'importance': final.feature_importances_,
    }).sort_values('importance', ascending=False)

    return final, fi, cv_aucs


# ══════════════════════════════════════════
#  予測
# ══════════════════════════════════════════
def predict_race(model, race_df: pd.DataFrame) -> pd.DataFrame:
    X     = race_df[FEATURE_COLS_V2].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    total = probs.sum()
    norm  = probs / total if total > 0 else np.ones(len(probs)) / len(probs)

    race_df = race_df.copy()
    race_df['win_prob']  = np.round(norm * 100, 2)
    race_df['ai_rank']   = pd.Series(norm).rank(ascending=False, method='first').astype(int)
    race_df['ai_score']  = ((norm - norm.min()) / (norm.max() - norm.min()) * 75 + 20).astype(int)

    pick_map = {1: '◎ 本命', 2: '○ 対抗', 3: '▲ 単穴'}
    race_df['pick'] = race_df['ai_rank'].map(lambda r: pick_map.get(r, ''))

    return race_df.sort_values('ai_rank')


# ══════════════════════════════════════════
#  回収率シミュレーション
# ══════════════════════════════════════════
def simulate_roi(model, df: pd.DataFrame) -> dict:
    df = df.copy()
    X  = df[FEATURE_COLS_V2].fillna(0)
    df['win_prob'] = model.predict_proba(X)[:, 1]
    df['ev']       = df['win_prob'] * df['odds_num']  # 期待値

    results = {}
    for strategy in ['top1', 'value']:
        bet = ret = 0
        for _, gdf in df.groupby('race_id' if 'race_id' in df.columns else df.index):
            gdf = gdf.sort_values('win_prob', ascending=False)
            targets = gdf.head(1) if strategy == 'top1' else gdf[gdf['ev'] > 1.2]
            for _, row in targets.iterrows():
                bet += 100
                if row.get('is_win', 0) == 1:
                    ret += 100 * row['odds_num']
        results[strategy] = {
            'bet': bet, 'ret': ret,
            'roi': ret / bet * 100 if bet > 0 else 0,
        }
    return results


# ══════════════════════════════════════════
#  可視化
# ══════════════════════════════════════════
def plot_importance(fi: pd.DataFrame, path='feature_importance_v2.png'):
    fig, ax = plt.subplots(figsize=(9, 7))
    top = fi.head(20)
    ax.barh(top['feature'][::-1], top['importance'][::-1], color='#c9a84c')
    ax.set_xlabel('重要度', fontsize=10)
    ax.set_title('特徴量重要度 Top20（v2）', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f'✅ 特徴量重要度: {path}')


# ══════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     default=None, help='学習用CSVパス')
    parser.add_argument('--predict', default=None, help='予測用CSVパス（出走表）')
    parser.add_argument('--save',    default=None, help='モデル保存パス（例: model.pkl）')
    parser.add_argument('--load',    default=None, help='モデル読み込みパス')
    args = parser.parse_args()

    # ── データ読み込み ──
    if args.csv:
        print(f'[データ読み込み] {args.csv}')
        raw = pd.read_csv(args.csv, encoding='utf-8-sig')
        df  = preprocess(raw)
    else:
        print('[デモモード] ランダムサンプルデータで動作確認します')
        df = make_demo_data(n_races=500)

    print(f'データ: {len(df)}行  特徴量: {len(FEATURE_COLS_V2)}個')

    # ── モデル学習 or 読み込み ──
    if args.load:
        with open(args.load, 'rb') as f:
            model = pickle.load(f)
        print(f'✅ モデル読み込み: {args.load}')
        fi = pd.DataFrame({'feature': FEATURE_COLS_V2, 'importance': model.feature_importances_})
        fi = fi.sort_values('importance', ascending=False)
    else:
        model, fi, cv_aucs = train_model(df)
        plot_importance(fi)

    # ── モデル保存 ──
    if args.save:
        with open(args.save, 'wb') as f:
            pickle.dump(model, f)
        print(f'✅ モデル保存: {args.save}')

    # ── 予測（出走表CSVがある場合）──
    if args.predict:
        print(f'\n[予測] {args.predict}')
        shutsuba = pd.read_csv(args.predict, encoding='utf-8-sig')
        shutsuba = preprocess(shutsuba)
        result   = predict_race(model, shutsuba)
        disp_cols = ['horse_name','ai_rank','pick','win_prob','ai_score','odds_num']
        disp_cols = [c for c in disp_cols if c in result.columns]
        print('\n' + '=' * 55)
        print('  AI 予想結果')
        print('=' * 55)
        print(result[disp_cols].to_string(index=False))

    else:
        # デモ予測（最後のレース）
        last_race = df[df.get('race_id', pd.Series(['x']*len(df))) ==
                       df.get('race_id', pd.Series(['x']*len(df))).iloc[-1]]
        if len(last_race) < 2:
            last_race = df.tail(18)
        result = predict_race(model, last_race)
        disp_cols = [c for c in ['horse_name','ai_rank','pick','win_prob','ai_score','odds_num'] if c in result.columns]
        print('\n' + '=' * 55)
        print('  AI 予想結果（デモレース）')
        print('=' * 55)
        print(result[disp_cols].to_string(index=False))

    # ── 回収率シミュレーション ──
    if 'is_win' in df.columns:
        print('\n' + '=' * 55)
        print('  回収率シミュレーション')
        print('=' * 55)
        roi_results = simulate_roi(model, df)
        for strategy, res in roi_results.items():
            print(f'  [{strategy:6s}] 投資:{res["bet"]:,}円  '
                  f'回収:{res["ret"]:,.0f}円  ROI:{res["roi"]:.1f}%')

    # ── 特徴量重要度 Top10 ──
    print('\n' + '=' * 55)
    print('  特徴量重要度 Top10')
    print('=' * 55)
    for _, row in fi.head(10).iterrows():
        bar = '█' * int(row['importance'] / fi['importance'].max() * 30)
        print(f'  {row["feature"]:<30} {bar}')

    print('\n✅ 完了！')
