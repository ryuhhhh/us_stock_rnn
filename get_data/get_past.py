import traceback
import pandas_datareader.data as pdr
import pandas as pd

def get_close_price_series(code,start = '2012-01-01',end = '2021-02-12'):
    """
    終値・取引高のseriesを取得する
    args:
        code(str):銘柄コーtド
        data_num_length(int):何日前のモノを取得するか
    returns:
        close_price_and_volume_df(series):終値と出来高のdf
    """
    # 終値のシリーズを取得する
    try:
        close_price_series = pdr.DataReader(code, 'yahoo',start,end)
    except Exception as ex:
        print(f'終値取得時にエラー')
        print(traceback.format_exc())
    return close_price_series

if __name__ == "__main__":
    CODE = 'コード'
    NAME = '名称'
    us_stock_list_df = pd.read_csv('./got_data/us_stocks_list.csv',encoding='utf-8')
    for index, row in us_stock_list_df.iterrows():
        if row['業種'] != '情報技術':
            continue
        print(f'{index}番目 {row[CODE]} {row[NAME]}')
        try:
            # 指定期間の終値シリーズの取得(日付で昇順)
            close_price_series = get_close_price_series(row[CODE])
            close_price_series.to_csv(f'./got_data/past_stock_data/{row[CODE]}.csv')
        except Exception as e:
            print(f'終値取得時にエラー発生。スキップします。')
            print(traceback.format_exc())
            continue
    print(f'終了しました /got_data/past_stock_data をご確認ください。')
