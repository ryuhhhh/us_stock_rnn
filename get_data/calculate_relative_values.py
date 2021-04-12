"""
株価の相対的な値上がり率を取得
"""
import pandas as pd
import numpy as np
import traceback

def get_up_ratio_and_isup(concat_close_df):
    close_ratio_df_1day = pd.DataFrame(columns=['ratio_1','ratio_2','ratio_3',
                                           'ratio_4','ratio_5','ratio_6',
                                           'ratio_7','ratio_8','ratio_9','ratio_10',
                                           '10per_up','5per_up'])
    close_ratio_df_1day.index.name = 'id'
    close_ratio_df_5day = pd.DataFrame(columns=['0~5slope_1','5~10slope_2','10~15slope_3','15~20slope_4',
                                    '10per_up','5per_up'])
    close_ratio_df_5day.index.name = 'id'
    for i in range(10,len(concat_close_df),10):
        # <--値上がり率--><-上昇率->
        # 10日前 ~ 対象日 ~ 10日後
        data = concat_close_df.iloc[i-10:i+11].dropna(axis=1)
        if len(data) < 20:
            break

        # データ用意
        past_11_data = data.iloc[0:-10]
        target_date = concat_close_df.iloc[i].name
        target_data = concat_close_df.iloc[i]
        future_10_data = data.iloc[-10:]

        print(target_date)

        # 上昇率
        up_ratio = ((past_11_data / past_11_data.shift(1)-1)*100).dropna(axis=0)
        print(up_ratio)
        # 銘柄同士で正規化する場合
        # up_ratio_standard = up_ratio.apply(lambda x: (x-x.mean())/x.std(), axis=1).dropna(axis=0)

        # 教師データ作成(10日後に10%up or 5%up)
        up_10_series = (future_10_data.max()/target_data > 1.1).astype(np.int32)
        up_10_series.name = '10per_up'
        up_5_series = (future_10_data.max()/target_data > 1.05).astype(np.int32)
        up_5_series.name = '5per_up'

        # Date+銘柄名,up_ratio_standard(10日分),5日up,10日up
        # for name, item in up_ratio_standard.iteritems():
        for name, item in up_ratio.iteritems():
            close_ratio_df_1day.loc[target_date + name] = [
                item[0],item[1],item[2],
                item[3],item[4],item[5],
                item[6],item[7],item[8],
                item[9],up_10_series[name],up_5_series[name]
            ]
        # 20以上の場合は5日区切りの傾きも
        if i>=20:
            data = concat_close_df.iloc[i-20:i+11].dropna(axis=1)
            past_20_data = data.iloc[0:-10]
            tmp_df = pd.DataFrame(columns=['0~5slope_1','5~10slope_2','10~15slope_3','15~20slope_4',
                                           '10per_up','5per_up'])
            for name, item in past_20_data.iteritems():
                id = target_date + name
                slope_list = get_trendline_list_4_quarter(item)
                tmp_df.loc[id] = [
                    slope_list[3],slope_list[2],slope_list[1],slope_list[0],
                    up_10_series[name],up_5_series[name]
                ]

            # 銘柄で正規化する場合
            # tmp_df_2 = tmp_df[['0~5slope_1','5~10slope_2','10~15slope_3','15~20slope_4']]\
            #     .apply(lambda x: (x-x.mean())/x.std(), axis=1)
            # tmp_df_2 = pd.concat([tmp_df_2,tmp_df[['10per_up','5per_up']]],axis=1)
            # close_ratio_df_5day = pd.concat([close_ratio_df_5day,tmp_df_2],axis=0)
            close_ratio_df_5day = pd.concat([close_ratio_df_5day,tmp_df],axis=0)

    close_ratio_df_1day.to_csv('./got_data/1day_data.csv')
    close_ratio_df_5day.to_csv('./got_data/5day_data.csv')

def get_trendline_list_4_quarter(close_df):
    """
    日付昇順の終値リストに指定されている期間で株価を取得し1時近似を取得する
    args:
        n日分の日付で昇順のリスト
    return:
        日付昇順の傾きリスト
    """
    QUARTER_NUM = 4
    quarter_length = 5
    slope_list = []
    # 平均
    close_price_avg = close_df.mean()
    # 指定期間の4等分の上昇量を取得する
    for i in range(0,len(close_df)-1,quarter_length):
        try:
            index = i
            if i>0:
                index = i-1
            slope,intercept =\
                 get_slope(close_df[index:i+quarter_length])
            # 傾きと切片を割合で示す
            slope_list.append(round(slope*100/close_price_avg,3))
        except Exception:
            print(f'1次近似取得時にエラー発生。スキップします。')
            print(traceback.format_exc())
            continue
    return slope_list

def get_slope(close_price_list,digits=2):
    """
    終値のリストより1次近似を求めます
    """
    close_price_list_num = len(close_price_list)
    try:
        slope,intercept = np.polyfit(list(range(close_price_list_num)), close_price_list, 1)
        slope = round(slope,digits)
        intercept = round(intercept,digits)
    except:
        print('1時近似を求めるのに失敗しました。スキップします。')
        return 0
    return slope,intercept

if __name__ == '__main__':
    """
    ・特徴量
      値上がり率(%) - 1日毎 x 10
      値上がり率(%) - 1週間ごと x 4
      1次近似の傾き - 1週間ごと x 4
      出来高 - 1日毎 x 10
      出来高 - 1週間ごと x 4
    ・教師
      10日以内に10%あがったか
      10日以内に5%あがったか
    """
    concat_close_df = pd.read_csv('./got_data/concat_close.csv',encoding='utf-8',index_col=0)
    # 1日x10 値上がり率
    # 1日x10 値上がり率の10日以内 5%up
    # 1日x10 値上がり率の10日以内 10%up
    get_up_ratio_and_isup(concat_close_df)

    # 5日x4(slope) 値上がり率
    # 5日x4(slope) 値上がり率の10日以内 5%up
    # 5日x4(slope) 値上がり率の10日以内 10%up

