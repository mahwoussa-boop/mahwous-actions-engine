import pandas as pd
import io

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
