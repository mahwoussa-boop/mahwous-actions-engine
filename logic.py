import pandas as pd
import io
import re
from dataclasses import dataclass
from rapidfuzz import process, fuzz

# ─── 1. محرك الاستخراج والتفكيك (Parsing Engine) ─────────────────────────

def extract_attributes(name: str):
    """تفكيك اسم المنتج إلى أبعاده الأساسية في عالم العطور والتجميل"""
    name_lower = str(name).lower()
    
    # 1. استخراج الحجم (Size)
    size_match = re.search(r'(\d+)\s*(مل|ml|جرام|g|oz|ملم)', name_lower, re.IGNORECASE)
    size = int(size_match.group(1)) if size_match else 0
    
    # 2. استخراج النوع (Product Type)
    product_type = "عطر تجاري" # الافتراضي
    if any(w in name_lower for w in ['تستر', 'tester', 'بدون كرتون', 'ديمو']):
        product_type = "تستر"
    elif any(w in name_lower for w in ['طقم', 'مجموعة', 'set', 'gift']):
        product_type = "طقم هدايا"
    elif any(w in name_lower for w in ['عطر شعر', 'للشعر', 'hair mist']):
        product_type = "عطر شعر"
    elif any(w in name_lower for w in ['لوشن', 'كريم', 'lotion', 'cream']):
        product_type = "عناية (لوشن/كريم)"
    elif any(w in name_lower for w in ['جل استحمام', 'شاور', 'shower gel']):
        product_type = "شاور جل"
    elif any(w in name_lower for w in ['معطر جسم', 'بدي مست', 'body mist', 'spray']):
        product_type = "معطر جسم"
    elif any(w in name_lower for w in ['مزيل عرق', 'ديودرنت', 'deodorant', 'stick']):
        product_type = "مزيل عرق"
        
    # 3. استخراج التركيز (Concentration)
    concentration = "غير محدد"
    if any(w in name_lower for w in ['اكستريت', 'extrait']):
        concentration = "Extrait"
    elif any(w in name_lower for w in ['او دي بارفيوم', 'او دو بارفيوم', 'edp', 'eau de parfum', 'بارفيوم']):
        concentration = "EDP"
    elif any(w in name_lower for w in ['او دي تواليت', 'او دو تواليت', 'edt', 'eau de toilette', 'تواليت']):
        concentration = "EDT"
    elif any(w in name_lower for w in ['بارفان', 'parfum', 'pure parfum']):
        concentration = "Parfum"
    elif any(w in name_lower for w in ['كولونيا', 'cologne', 'edc']):
        concentration = "EDC"
        
    # إضافة خصائص إضافية (Intense, Absolu)
    if 'انتنس' in name_lower or 'intense' in name_lower: concentration += " Intense"
    if 'ابسولو' in name_lower or 'absolu' in name_lower: concentration += " Absolu"

    # 4. تنظيف الاسم لاستخراج "الاسم النقي" للمطابقة (Core Name)
    clean_name = re.sub(r'\d+\s*(مل|ml|جرام|g|oz|ملم|x)', '', name_lower)
    words_to_remove = [
        'عطر', 'او دي بارفيوم', 'او دو بارفيوم', 'او دي تواليت', 'او دو تواليت', 
        'بارفيوم', 'برفيوم', 'تواليت', 'اكستريت', 'بارفان', 'كولونيا',
        'تستر', 'طقم', 'مجموعة', 'للرجال', 'للنساء', 'نسائي', 'رجالي', 'مركز',
        'عطر شعر', 'لوشن', 'شاور جل', 'edp', 'edt', 'tester', 'set', 'hair mist'
    ]
    for w in words_to_remove:
        clean_name = clean_name.replace(w, '')
        
    clean_name = ' '.join(clean_name.split()) # إزالة المسافات الزائدة
    
    return {
        'size': size,
        'type': product_type,
        'concentration': concentration,
        'clean_name': clean_name
    }


# ─── 2. كلاسات تجهيز البيانات ───────────────────────────────────────────────

class FeatureParser:
    @staticmethod
    def extract_features(df: pd.DataFrame, source_type="store") -> pd.DataFrame:
        df_cleaned = pd.DataFrame()
        
        # استخراج الأعمدة الأساسية باختلاف مصادرها
        if source_type == "store":
            name_col = next((c for c in df.columns if 'اسم' in str(c) or 'أسم' in str(c)), df.columns[2] if len(df.columns) > 2 else None)
            price_col = next((c for c in df.columns if 'سعر' in str(c)), df.columns[7] if len(df.columns) > 7 else None)
            image_col = next((c for c in df.columns if 'صورة' in str(c)), df.columns[4] if len(df.columns) > 4 else None)
        else:
            name_col = next((c for c in df.columns if 'name' in str(c).lower() or 'اسم' in str(c)), df.columns[2] if len(df.columns) > 2 else None)
            price_col = next((c for c in df.columns if 'price' in str(c).lower() or 'سعر' in str(c) or 'sm' in str(c).lower()), df.columns[3] if len(df.columns) > 3 else None)
            image_col = next((c for c in df.columns if 'src' in str(c).lower() or 'صورة' in str(c)), df.columns[1] if len(df.columns) > 1 else None)
            
        df_cleaned['orig_name'] = df[name_col].astype(str) if name_col else ""
        df_cleaned['price'] = df[price_col].astype(str) if price_col else "0"
        df_cleaned['image'] = df[image_col].astype(str) if image_col else ""
        if 'source_file' in df.columns:
            df_cleaned['source_file'] = df['source_file']
            
        # تطبيق التفكيك الجزيئي على كل منتج
        parsed = df_cleaned['orig_name'].apply(extract_attributes)
        df_cleaned['size'] = [p['size'] for p in parsed]
        df_cleaned['type'] = [p['type'] for p in parsed]
        df_cleaned['concentration'] = [p['concentration'] for p in parsed]
        df_cleaned['clean_name'] = [p['clean_name'] for p in parsed]
        
        return df_cleaned


class GeminiOracle:
    def __init__(self, api_key: str): self.api_key = api_key

class SemanticIndex:
    def __init__(self, model):
        self.model = model
        self.store_features = pd.DataFrame()
        
    def build(self, df: pd.DataFrame, progress_cb=None):
        if progress_cb: progress_cb("جاري تفكيك منتجات المتجر واستخراج (الأنواع، الأحجام، التراكيز)...")
        self.store_features = FeatureParser.extract_features(df, "store")


@dataclass
class MatchResult:
    comp_name: str = ""
    comp_image: str = ""
    comp_price: str = ""
    comp_source: str = ""
    store_name: str = ""
    confidence: float = 0.0
    layer_used: str = ""
    brand: str = ""
    verdict: str = ""
    generated_product_description: str = ""  # نستخدمه هنا لشرح سبب القرار
    generated_brand_description: str = ""

# ─── 3. المحرك الأساسي (العقل المدبر) ────────────────────────────────────────

class MahwousEngine:
    def __init__(self, semantic_index, brands_list, gemini_oracle=None):
        self.semantic_index = semantic_index
        self.brands_list = brands_list
        self.gemini_oracle = gemini_oracle

    def run(self, store_df, comp_df, use_llm=False, progress_cb=None, log_cb=None):
        if log_cb: log_cb("بدء المطابقة خماسية الأبعاد (الاسم، الحجم، التركيز، النوع)...")
            
        new_opps, duplicates, reviews, new_brands = [], [], [], []
        
        comp_features = FeatureParser.extract_features(comp_df, "comp")
        store_feats = self.semantic_index.store_features
        
        store_clean_dict = {i: row['clean_name'] for i, row in store_feats.iterrows()}
        total_comps = len(comp_features)
        
        for i, row in comp_features.iterrows():
            comp_orig = str(row['orig_name']).strip()
            if comp_orig == "nan" or not comp_orig: continue
                
            if progress_cb: progress_cb(i, total_comps, comp_orig[:40])
            
            comp_clean = row['clean_name']
            comp_size = row['size']
            comp_type = row['type']
            comp_conc = row['concentration']
            
            # 1. مطابقة الاسم النقي
            best_match = process.extractOne(comp_clean, store_clean_dict, scorer=fuzz.token_set_ratio)
            
            store_match_name = ""
            score = 0.0
            verdict = "فرصة جديدة"
            reason = "منتج جديد تماماً"
            
            if best_match:
                _, match_score, match_idx = best_match
                score = match_score / 100.0
                
                store_row = store_feats.iloc[match_idx]
                store_match_name = store_row['orig_name']
                
                if score >= 0.88:
                    # الاسم متطابق، ندخل في الفحص العميق!
                    store_size = store_row['size']
                    store_type = store_row['type']
                    store_conc = store_row['concentration']
                    
                    if comp_type != store_type:
                        verdict = "فرصة جديدة"
                        reason = f"اختلاف النوع: لدينا ({store_type}) والمنافس يبيع ({comp_type})"
                    elif comp_size != store_size and comp_size != 0 and store_size != 0:
                        verdict = "فرصة جديدة"
                        reason = f"اختلاف الحجم: لدينا ({store_size}مل) والمنافس يبيع ({comp_size}مل)"
                    elif comp_conc != store_conc and comp_conc != "غير محدد" and store_conc != "غير محدد":
                        verdict = "فرصة جديدة"
                        reason = f"اختلاف التركيز: لدينا ({store_conc}) والمنافس يبيع ({comp_conc})"
                    else:
                        verdict = "مكرر"
                        reason = "تطابق تام في (الاسم، الحجم، التركيز، النوع)"
                        
                elif score >= 0.60:
                    verdict = "مراجعة يدوية"
                    reason = "تشابه في الاسم، يرجى التأكد من الإصدار"
            
            result = MatchResult(
                comp_name=comp_orig,
                comp_price=str(row['price']),
                comp_image=str(row['image']),
                comp_source=row.get('source_file', 'مجهول'),
                store_name=store_match_name,
                confidence=score,
                layer_used="5D-Analyzer",
                verdict=verdict,
                generated_product_description=reason # تدوين السبب في الملف
            )
            
            if verdict == "مكرر": duplicates.append(result)
            elif verdict == "فرصة جديدة": new_opps.append(result)
            else: reviews.append(result)
        
        return new_opps, duplicates, reviews, new_brands


# ─── 4. دوال التصدير والتحميل ──────────────────────────────────────────────

def load_store_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            if str(f).endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8', skiprows=1) if 'متجرنا' in str(f) else pd.read_csv(f, encoding='utf-8')
            elif str(f).endswith(('.xlsx', '.xls')): df = pd.read_excel(f)
            else: continue
            frames.append(df)
        except: pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_competitor_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            if str(f).endswith('.csv'): df = pd.read_csv(f, encoding='utf-8')
            elif str(f).endswith(('.xlsx', '.xls')): df = pd.read_excel(f)
            else: continue
            df['source_file'] = str(f)
            frames.append(df)
        except: pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_brands(file: str) -> list: return []

def export_salla_csv(results: list) -> bytes:
    if not results: return b""
    rows = [{'product_name': getattr(r, 'comp_name', ''), 'price': getattr(r, 'comp_price', ''), 'image': getattr(r, 'comp_image', '')} for r in results]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig') 
    return buf.getvalue().encode('utf-8-sig')

def export_brands_csv(brands: list) -> bytes: return b""
