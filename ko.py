import FinanceDataReader as fdr

# KOSPI 종목 불러오기
kospi_df = fdr.StockListing('KOSPI')

# 종목명과 종목코드만 추출
kospi_names = kospi_df[['Name', 'Code']]

# CSV 저장
kospi_names.to_csv('kospi_names.csv', index=False, encoding='utf-8-sig')

print("KOSPI 종목 이름과 코드가 'kospi_names.csv'로 저장되었습니다.")