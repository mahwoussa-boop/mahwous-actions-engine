import pandas as pd
import io
from dataclasses import dataclass
from rapidfuzz import process, fuzz

# ─── الكلاسات الأساسية لتشغيل المحرك والمطابقة ───────────────────────────────

class FeatureParser:
    """كلاس لتحليل خصائص المنتجات والتعامل مع أسماء الأعمدة المتغيرة ديناميكياً"""
    @staticmethod
    def extract_features(df: pd.DataFrame, source_type="store") -> pd.DataFrame:
        df_cleaned = pd.DataFrame()
        
        if source_type == "store":
            # البحث عن أعمدة متجر مهووس ديناميكياً أو استخدام الترتيب كبديل
            name_col = next((c for c in df.columns if 'اسم' in str(c) or 'أسم' in str(c)), df.columns[2] if len(df.columns) > 2 else None)
            price_col = next((c for c in df.columns if 'سعر' in str(c)), df.columns[7] if len(df.columns) > 7 else None)
            image_col = next((c for c in df.columns if 'صورة' in str(c)), df.columns[4] if len(df.columns) > 4 else None)
            
            df_cleaned['name'] = df[name_col].astype(str) if name_col else "بدون اسم"
            df_cleaned['price'] = df[price_col].astype(str) if price_col else "0"
            df_cleaned['image'] = df[image_col].astype(str) if image_col else ""
            
        else:
            # البحث عن أعمدة المنافسين ديناميكياً (غالباً ملفات مسحوبة بأسماء برمجية)
            name_col = next((c for c in df.columns if 'name' in str(c).lower() or 'اسم' in str(c)), df.columns[2] if len(df.columns) > 2 else None)
            price_col = next((c for c in df.columns if 'price' in str(c).lower() or 'سعر' in str(c) or 'sm' in str(c).lower()), df.columns[3] if len(df.columns) > 3 else None)
            image_col = next((c for c in df.columns if 'src' in str(c).lower() or 'صورة' in str(c)), df.columns[1] if len(df.columns) > 1 else None)
            
            df_cleaned['name'] = df[name_col].astype(str) if name_col else "بدون اسم"
            df_cleaned['price'] = df[price_col].astype(str) if price_col else "0"
            df_cleaned['image'] = df[image_col].astype(str) if image_col else ""
            if 'source_file' in df.columns:
                df_cleaned['source_file'] = df['source_file']

        return df_cleaned

class GeminiOracle:
    """كلاس للتواصل مع نماذج الذكاء الاصطناعي"""
    def __init__(self, api_key: str):
        self.api_key = api_key

class SemanticIndex:
    """كلاس لبناء الفهرس الدلالي (FAISS)"""
    def __init__(self, model):
        self.model = model
        self.store_names = []
        
    def build(self, df: pd.DataFrame, progress_cb=None):
        if progress_cb:
            progress_cb("جاري تجهيز الفهرس الدلالي وكلمات المتجر...")
        # استخراج أسماء المنتجات من متجرك للاعتماد عليها في المطابقة
        features = FeatureParser.extract_features(df, "store")
        self.store_names = features['name'].tolist()

@dataclass
class MatchResult:
    """كلاس يمثل نتيجة مطابقة المنتج"""
    comp_name: str = ""
    comp_image: str = ""
    comp_price: str = ""
    comp_source: str = ""
    store_name: str = ""
    confidence: float = 0.0
    layer_used: str = ""
    brand: str = ""
    verdict: str = ""
    generated_product_description: str = ""
    generated_brand_description: str = ""

class MahwousEngine:
    """محرك مهووس الأساسي للمطابقة والتحليل"""
    def __init__(self, semantic_index, brands_list, gemini_oracle=None):
        self.semantic_index = semantic_index
        self.brands_list = brands_list
        self.gemini_oracle = gemini_oracle

    def run(self, store_df, comp_df, use_llm=False, progress_cb=None, log_cb=None):
        if log_cb:
            log_cb("بدء عملية تحليل البيانات الحقيقية ومطابقة المنتجات...")
            
        new_opps = []
        duplicates = []
        reviews = []
        new_brands = []
        
        # 1. تجهيز البيانات
        comp_features = FeatureParser.extract_features(comp_df, "comp")
        store_names = self.semantic_index.store_names
        total_comps = len(comp_features)
        
        # 2. المرور على منتجات المنافسين ومقارنتها بمتجرك
        for i, row in comp_features.iterrows():
            comp_name = str(row['name']).strip()
            if comp_name == "nan" or not comp_name:
                continue
                
            if progress_cb:
                progress_cb(i, total_comps, comp_name)
            
            # خوارزمية المطابقة النصية الذكية (Fuzzy Matching)
            best_match = process.extractOne(comp_name, store_names, scorer=fuzz.token_sort_ratio)
            
            store_match_name = ""
            score = 0.0
            if best_match:
                store_match_name = best_match[0]
                score = best_match[1] / 100.0  # تحويل النسبة لتكون بين 0 و 1
            
            # 3. اتخاذ القرار بناءً على نسبة التطابق
            result = MatchResult(
                comp_name=comp_name,
                comp_price=str(row['price']),
                comp_image=str(row['image']),
                comp_source=row.get('source_file', 'مجهول'),
                store_name=store_match_name,
                confidence=score,
                layer_used="FuzzyText",
                verdict=""
            )
            
            if score >= 0.85:
                # تشابه كبير جداً = المنتج موجود مسبقاً
                result.verdict = "مكرر"
                duplicates.append(result)
            elif score >= 0.55:
                # تشابه متوسط = يحتاج لمراجعة بشرية
                result.verdict = "مراجعة يدوية"
                reviews.append(result)
            else:
                # لا يوجد تشابه = منتج جديد للمنافس فرصة لمتجرك
                result.verdict = "فرصة جديدة"
                new_opps.append(result)
        
        return new_opps, duplicates, reviews, new_brands


# ─── الدوال الأصلية لتحميل وتصدير البيانات ──────────────────────────────

def load_store_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            if str(f).endswith('.csv'):
                # تخطي السطر الأول في ملف سلة إذا كان يحتوي على معلومات عامة
                df = pd.read_csv(f, encoding='utf-8', skiprows=1) if 'متجرنا' in str(f) else pd.read_csv(f, encoding='utf-8')
            elif str(f).endswith(('.xlsx', '.xls')):
                df = pd.read_excel(f)
            else:
                continue
            frames.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_competitor_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            if str(f).endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8')
            elif str(f).endswith(('.xlsx', '.xls')):
                df = pd.read_excel(f)
            else:
                continue
                
            df['source_file'] = str(f)
            frames.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_brands(file: str) -> list:
    try:
        if str(file).endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif str(file).endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return []
        
        brand_col = next((c for c in df.columns if 'brand' in c.lower()), df.columns[0])
        return df[brand_col].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error loading brands: {e}")
        return []


def export_salla_csv(results: list) -> bytes:
    if not results:
        return b""
    rows = []
    for r in results:
        rows.append({
            'product_name': getattr(r, 'comp_name', ''),
            'price': getattr(r, 'comp_price', ''),
            'image': getattr(r, 'comp_image', ''),
        })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue().encode('utf-8')


def export_brands_csv(brands: list) -> bytes:
    if not brands:
        return b""
    rows = [{'brand': b} for b in brands]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue().encode('utf-8')
