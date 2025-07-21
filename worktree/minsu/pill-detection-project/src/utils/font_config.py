# src/utils/font_config.py
"""한글 폰트 설정"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # NanumGothic 폰트 설정
        fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
        plt.rcParams['mathtext.default'] = 'regular'
        print("✅ NanumGothic 폰트 적용 완료")
        return True
    except Exception as e:
        print(f"⚠️ NanumGothic 폰트 로드 실패: {e}")
        
        # 대안 폰트들 시도
        fallback_fonts = [
            'Malgun Gothic',  # Windows
            'AppleGothic',    # macOS
            'Noto Sans CJK KR',  # Linux
            'DejaVu Sans'     # 최후의 대안
        ]
        
        for font in fallback_fonts:
            try:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ {font} 폰트로 대체")
                return True
            except:
                continue
        
        print("⚠️ 한글 폰트 설정 실패 - 기본 폰트 사용")
        # 한글 폰트 경고 억제
        warnings.filterwarnings('ignore', category=UserWarning, 
                              message='.*Glyph.*missing from font.*')
        return False

# 자동 설정
setup_korean_font()