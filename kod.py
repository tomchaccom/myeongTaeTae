import FinanceDataReader as fdr

# KOSDAQ 종목 불러오기
kosdaq_df = fdr.StockListing('KOSDAQ')

# 종목명과 종목코드 저장
kosdaq_names = kosdaq_df[['Name', 'Code']]
kosdaq_names.to_csv('kosdaq_names.csv', index=False, encoding='utf-8-sig')

print("KOSDAQ 종목 이름과 코드도 저장 완료!")