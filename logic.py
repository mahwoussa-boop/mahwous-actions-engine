import pandas as pd
import io
from dataclasses import dataclass

# ─── الكلاسات المفقودة التي يطلبها المحرك ───────────────────────────────

class FeatureParser:
    """كلاس لتحليل خصائص المنتجات"""
    pass

class GeminiOracle:
    """كلاس للتواصل مع نماذج الذكاء الاصطناعي"""
    def __init__(self, api_key: str):
        self.api_key = api_key

class SemanticIndex:
    """كلاس لبناء الفهرس الدلالي (FAISS)"""
    def __init__(self, model):
        self.model = model
        
    def build(self, df: pd.DataFrame, progress_cb=None):
        # بناء الفهرس
        if progress_cb:
            progress_cb("جاري بناء الفهرس الدلالي...")
        pass

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
        """
        تشغيل المحرك وإرجاع 4 قوائم:
        (فرص جديدة, مكررات, مراجعة يدوية, ماركات جديدة)
        """
        if log_cb:
            log_cb("بدء عملية تحليل البيانات في المحرك...")
            
        new_opps = []
        duplicates = []
        reviews = []
        new_brands = []
        
        return new_opps, duplicates, reviews, new_brands


# ─── الدوال الأصلية لتحميل وتصدير البيانات ──────────────────────────────

def load_store_products(files: list) -> pd.DataFrame:
    """Load store products from CSV/Excel files."""
    frames = []
    for f in files:
        try:
            if str(f).endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8')
            elif str(f).endswith(('.xlsx', '.xls')):
                df = pd.read_excel(f)
            else:
                continue
            frames.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_competitor_products(files: list) -> pd.DataFrame:
    """Load competitor products from CSV/Excel files."""
    frames = []
    for f in files:
        try:
            if str(f).endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8')
            elif str(f).endswith(('.xlsx', '.xls')):
                df = pd.read_excel(f)
            else:
                continue
            frames.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_brands(file: str) -> list:
    """Load brands from CSV/Excel file."""
    try:
        if str(file).endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif str(file).endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return []
        
        # Get brand column
        brand_col = next((c for c in df.columns if 'brand' in c.lower()), df.columns[0])
        return df[brand_col].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error loading brands: {e}")
        return []


def export_salla_csv(results: list) -> bytes:
    """Export results as Salla-compatible CSV."""
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
    """Export brands as CSV."""
    if not brands:
        return b""
    
    rows = [{'brand': b} for b in brands]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue().encode('utf-8')
