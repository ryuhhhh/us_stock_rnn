"""
取得した銘柄の過去株価を結合し1ファイルにまとめる
"""
import pandas as pd
import traceback

if __name__ == '__main__':
    CODE = 'コード'
    NAME = '名称'

    us_stock_list_df = pd.read_csv('./got_data/us_stocks_list.csv',encoding='utf-8')
    concatenated_df = pd.DataFrame()
    for index, row in us_stock_list_df.iterrows():
        print(f'{index}番目 {row[CODE]} {row[NAME]}')
        if row['業種'] != '情報技術':
            continue
        # if index>=50:
        #     break
        try:
            next_df = pd.read_csv(f'./got_data/past_stock_data/{row[CODE]}.csv',index_col=0,usecols=['Date','Close'])
            next_df = next_df.rename(columns={'Close': row[CODE]})
            concatenated_df = pd.concat([concatenated_df, next_df],axis=1,sort=True)
            print(concatenated_df)
            # exit()
        except Exception as e:
            print(f'終値取得時にエラー発生。スキップします。')
            print(traceback.format_exc())
            continue
    concatenated_df.to_csv('./got_data/concat_close.csv')
    print(f'終了しました ./got_data/concat_close.csv をご確認ください。')