#预处理数据，将数据更改为含有临近指标的数据，并分为测试集和训练集

import pandas as pd
import numpy as np


def main():
    """
    #读入数据（作为csv文件的本底和信号数据）

    background = pd.read_csv('background.csv')
    signal = pd.read_csv('signal.csv')
    frames = [background, signal]

    #合并与排序
    total = pd.concat(frames)
    newtotal = total.reset_index(drop=True)
    newtotal.rename(columns={'0': 'EventID', '1': 'Layer', '2': 'Wire', '3': 'Rawtime', '4': 'ADC', '5': 't0',
                             '6': 'isSignal'}, inplace=True)
    newtotal.to_csv('debug.csv')
    """
    # 读入数据（作为txt文件的混合数据）
    txt = np.loadtxt('data.txt')
    newtotal = pd.DataFrame(txt).reset_index(drop=True)
    newtotal.to_csv('../data/debug.csv')
    newtotal.rename(columns={0: 'isSignal', 1: 'EventID', 2: 'Layer', 3: 'Wire', 4: 'Rawtime', 5: 'ADC',
                             6: 't0'}, inplace=True)

    # 构建neighbor集
    ext = pd.DataFrame(columns=['nRt', 'nRA', 'nuRt', 'nuRA', 'ndRt', 'ndRA', 'nLt', 'nLA', 'nuLt', 'nuLA', 'ndLt',
                                'ndLA'])
    for i, row in enumerate(newtotal.itertuples()):
        for row2 in newtotal.itertuples():
            if row2.Layer == row.Layer:
                if row2.Wire == row.Wire+1:
                    ext.loc[i, 'nRt'] = row2.Rawtime
                    ext.loc[i, 'nRA'] = row2.ADC
                if row2.Wire == row.Wire-1:
                    ext.loc[i, 'nLt'] = row2.Rawtime
                    ext.loc[i, 'nLA'] = row2.ADC
            if row2.Layer == row.Layer+1:
                if row2.Wire == row.Wire+1:
                    ext.loc[i, 'nuRt'] = row2.Rawtime
                    ext.loc[i, 'nuRA'] = row2.ADC
                if row2.Wire == row.Wire-1:
                    ext.loc[i, 'nuLt'] = row2.Rawtime
                    ext.loc[i, 'nuLA'] = row2.ADC
            if row2.Layer == row.Layer-1:
                if row2.Wire == row.Wire+1:
                    ext.loc[i, 'ndRt'] = row2.Rawtime
                    ext.loc[i, 'ndRA'] = row2.ADC
                if row2.Wire == row.Wire-1:
                    ext.loc[i, 'ndLt'] = row2.Rawtime
                    ext.loc[i, 'ndLA'] = row2.ADC
    result = pd.concat([newtotal, ext], axis=1)
    result.fillna(-1).to_csv('../data/result.csv')


if __name__ == '__main__':
    main()
