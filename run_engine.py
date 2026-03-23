"""
run_engine.py — سكربت التشغيل الآلي لمحرك مهووس v9.0
======================================================
يعمل في بيئة GitHub Actions بشكل تلقائي كامل.
- يقرأ ملفات المتجر من: input/store/
- يقرأ ملفات المنافسين من: input/competitors/
- يقرأ ملف الماركات من: input/brands/
- يحفظ النتائج في: output/
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path

# ── إعداد السجل ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mahwous-runner")

# ── المسارات ─────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
INPUT_STORE     = BASE_DIR / "input" / "store"
INPUT_COMP      = BASE_DIR / "input" / "competitors"
INPUT_BRANDS    = BASE_DIR / "input" / "brands"
OUTPUT_DIR      = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── متغيرات البيئة ───────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_LLM    = os.environ.get("USE_LLM", "true").lower() == "true"

# ── استيراد المحرك ───────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from logic import (
    FeatureParser, GeminiOracle, MahwousEngine,
    MatchResult, SemanticIndex,
    export_brands_csv, export_salla_csv,
    load_brands, load_competitor_products, load_store_products,
)

def _load_csv_files(folder: Path) -> list:
    """تحميل كل ملفات CSV من مجلد معين."""
    files = list(folder.glob("*.csv"))
    if not files:
        log.warning(f"⚠️ لا توجد ملفات CSV في: {folder}")
        return []
    log.info(f"📂 وجدت {len(files)} ملف في {folder.name}/")
    return files

def _progress_cb(i: int, total: int, name: str) -> None:
    """شريط التقدم في السجل."""
    if i % 100 == 0 or i < 5:
        pct = i / max(total, 1) * 100
        log.info(f"  ⚙️  [{pct:5.1f}%] {i}/{total} — {name[:50]}")

def _log_cb(msg: str) -> None:
    """تسجيل رسائل المحرك."""
    log.info(msg)

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("🚀 محرك مهووس للحسم v9.0 — بدء التشغيل")
    log.info("=" * 60)

    # ── 1. تحميل ملفات المتجر ────────────────────────────────────────────────
    store_files = _load_csv_files(INPUT_STORE)
    if not store_files:
        log.error("❌ لا توجد ملفات متجر! ضع ملفات CSV في مجلد input/store/")
        sys.exit(1)

    log.info("📥 تحميل بيانات متجر مهووس...")
    store_df = load_store_products(store_files)
    if store_df.empty:
        log.error("❌ ملفات المتجر فارغة أو لا تحتوي على بيانات صالحة!")
        sys.exit(1)
    log.info(f"✅ {len(store_df):,} منتج في الجدار الواقي")

    # ── 2. تحميل ملف الماركات (اختياري) ─────────────────────────────────────
    brand_files = list(INPUT_BRANDS.glob("*.csv"))
    existing_brands = []
    if brand_files:
        existing_brands = load_brands(brand_files[0])
        log.info(f"✅ {len(existing_brands):,} ماركة محملة")
    else:
        log.info("ℹ️ لا يوجد ملف ماركات — سيتم الاستخراج تلقائياً")

    # ── 3. تحميل ملفات المنافسين ─────────────────────────────────────────────
    comp_files = _load_csv_files(INPUT_COMP)
    if not comp_files:
        log.error("❌ لا توجد ملفات منافسين! ضع ملفات CSV في مجلد input/competitors/")
        sys.exit(1)

    log.info("📦 تحميل بيانات المنافسين...")
    comp_df = load_competitor_products(comp_files)
    if comp_df.empty:
        log.error("❌ ملفات المنافسين فارغة!")
        sys.exit(1)
    log.info(f"✅ {len(comp_df):,} منتج من {len(comp_files)} منافس")
    for src in comp_df["source_file"].unique():
        count = len(comp_df[comp_df["source_file"] == src])
        log.info(f"   └── {Path(src).name}: {count:,} منتج")

    # ── 4. بناء فهرس FAISS ───────────────────────────────────────────────────
    log.info("🧠 تحميل نموذج اللغة وبناء فهرس FAISS...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        semantic_idx = SemanticIndex(model)
        semantic_idx.build(store_df, progress_cb=_log_cb)
        log.info(f"✅ FAISS جاهز: {len(store_df):,} متجه دلالي")
    except Exception as e:
        log.error(f"❌ فشل بناء FAISS: {e}")
        sys.exit(1)

    # ── 5. تهيئة الذكاء الاصطناعي (اختياري) ─────────────────────────────────
    oracle = None
    if USE_LLM and (GEMINI_KEY or OPENAI_KEY):
        oracle = GeminiOracle(GEMINI_KEY or OPENAI_KEY)
        log.info("🤖 الذكاء الاصطناعي نشط للمنطقة الرمادية")
    else:
        log.info("ℹ️ الذكاء الاصطناعي غير مفعّل — المنطقة الرمادية → مراجعة يدوية")

    # ── 6. تشغيل المحرك ──────────────────────────────────────────────────────
    log.info(f"⚖️ بدء التحليل الهجين على {len(comp_df):,} منتج...")
    engine = MahwousEngine(
        semantic_index=semantic_idx,
        brands_list=existing_brands,
        gemini_oracle=oracle,
    )

    new_opps, duplicates, reviews = engine.run(
        store_df    = store_df,
        comp_df     = comp_df,
        use_llm     = USE_LLM and oracle is not None,
        progress_cb = _progress_cb,
        log_cb      = _log_cb,
    )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"🎉 اكتمل التحليل في {elapsed:.1f} ثانية")
    log.info(f"   🌟 فرص جديدة:    {len(new_opps):,}")
    log.info(f"   🚫 مكررات:       {len(duplicates):,}")
    log.info(f"   🔍 مراجعة يدوية: {len(reviews):,}")
    log.info("=" * 60)

    # ── 7. حفظ النتائج ───────────────────────────────────────────────────────
    import pandas as pd
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")

    # ملف سلة الجاهز للرفع المباشر
    if new_opps:
        salla_bytes = export_salla_csv(new_opps)
        salla_path  = OUTPUT_DIR / f"سلة_فرص_جديدة_{date_str}.csv"
        salla_path.write_bytes(salla_bytes)
        log.info(f"💾 ملف سلة: {salla_path.name}")

    # ملف الفرص الجديدة التفصيلي
    def _results_to_df(results: list[MatchResult]) -> pd.DataFrame:
        return pd.DataFrame([{
            "اسم منتج المنافس":  r.comp_name,
            "صورة المنافس":      r.comp_image,
            "سعر المنافس":       r.comp_price,
            "مصدر الملف":        r.comp_source,
            "أقرب منتج لدينا":   r.store_name,
            "نسبة التطابق %":    f"{r.confidence*100:.1f}",
            "الطبقة المستخدمة":  r.layer_used,
            "الماركة":           r.brand,
            "القرار":            r.verdict,
        } for r in results])

    if new_opps:
        df_new = _results_to_df(new_opps)
        df_new.to_csv(OUTPUT_DIR / f"فرص_جديدة_{date_str}.csv", index=False, encoding="utf-8-sig")
        log.info(f"💾 فرص جديدة: {len(new_opps):,} منتج")

    if duplicates:
        df_dup = _results_to_df(duplicates)
        df_dup.to_csv(OUTPUT_DIR / f"مكررات_محظورة_{date_str}.csv", index=False, encoding="utf-8-sig")
        log.info(f"💾 مكررات: {len(duplicates):,} منتج")

    if reviews:
        df_rev = _results_to_df(reviews)
        df_rev.to_csv(OUTPUT_DIR / f"مراجعة_يدوية_{date_str}.csv", index=False, encoding="utf-8-sig")
        log.info(f"💾 مراجعة يدوية: {len(reviews):,} منتج")

    # ملف الملخص لواجهة GitHub
    summary_lines = [
        f"| المقياس | القيمة |",
        f"|:---|:---:|",
        f"| 📅 تاريخ التشغيل | {date_str} |",
        f"| 🏪 منتجات متجرنا | {len(store_df):,} |",
        f"| 🔍 منتجات المنافسين | {len(comp_df):,} |",
        f"| 🌟 **فرص جديدة** | **{len(new_opps):,}** |",
        f"| 🚫 مكررات محظورة | {len(duplicates):,} |",
        f"| 🔍 مراجعة يدوية | {len(reviews):,} |",
        f"| ⏱️ وقت التشغيل | {elapsed:.1f} ثانية |",
        f"| 🤖 الذكاء الاصطناعي | {'✅ نشط' if oracle else '⚠️ غير مفعّل'} |",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    log.info("✅ تم حفظ كل النتائج في مجلد output/")
    log.info(f"📥 قم بتحميل النتائج من: Actions → Run #{os.environ.get('GITHUB_RUN_NUMBER','?')} → Artifacts")

if __name__ == "__main__":
    main()
