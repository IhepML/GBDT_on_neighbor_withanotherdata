import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


def selection_feature() -> None:
    #用直观图像显示是否选取
    data = pd.read_csv('../data/result.csv')
    for i, x in enumerate(['EventID', 'Layer', 'Wire', 'Rawtime', 't0', 'ADC', 'nRt', 'nRA', 'nuRt', 'nuRA', 'ndRt',
                           'ndRA', 'nLt', 'nLA', 'nuLt', 'nuLA', 'ndLt', 'ndLA']):
        if x in ['EventID', 'Layer', 'Wire', 't0', 'ADC', 'nRA', 'nuRA', 'ndRA', 'nLA', 'nuLA', 'ndLA']:
            log_Signal = data.loc[data['isSignal'] == 1].loc[:, x].apply(lambda y: np.log10(y + 2))
            log_Background = data.loc[data['isSignal'] == 0].loc[:, x].apply(lambda y: np.log10(y + 2))
            hist_Signal = plt.hist(log_Signal, bins=50, density=True, facecolor="blue", edgecolor="black",
                                   alpha=0.5, label='Signal')
            hist_Background = plt.hist(log_Background, bins=50, density=True, facecolor="red", edgecolor="black",
                                       alpha=0.5, label='Background')
            plt.xlabel(x + ' Interval')
            plt.ylabel('fpd')
            plt.title('log_' + x + '  FDH')
            plt.savefig('../imagine/' + x + '.png', format='png')
            plt.cla()
        else:
            _Signal = data.loc[data['isSignal'] == 1].loc[:, x]
            _Background = data.loc[data['isSignal'] == 0].loc[:, x]
            hist_Signal = plt.hist(_Signal, bins=50, density=True, facecolor="blue", edgecolor="black",
                                   alpha=0.5, label='Signal')
            hist_Background = plt.hist(_Background, bins=50, density=True, facecolor="red", edgecolor="black",
                                       alpha=0.5, label='Background')
            plt.xlabel(x + ' Interval')
            plt.ylabel('fpd')
            plt.title(x + '  FDH')
            plt.savefig('../imagine/' + x + '.png', format='png')
            plt.cla()


def main():
    selection_feature()
    # 随机分为训练集和测试集
    data = pd.read_csv('../data/result.csv')
    feature = ['Layer', 'Rawtime', 'ADC', 'nRA', 'nLA', 'nuRA', 'ndRA', 'nuLA', 'ndLA']
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, feature], data.loc[:, 'isSignal'],
        test_size=0.33, random_state=42)
    clf = GradientBoostingClassifier(random_state=10, learning_rate=0.20, n_estimators=82, min_samples_leaf=20,
                                     max_features='sqrt', subsample=0.90, max_depth=15, min_samples_split=504)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    y_pre_prob = clf.predict_proba(X_test)[:, 1]

    # 画图
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pre_prob)
    roc_auc = auc(fpr, tpr)
    result = pd.DataFrame({'tpr': tpr, 'tnr': 1-fpr})
    print(result.loc[0.9895 <= result['tpr']].loc[0.9910 > result['tpr']])
    print(roc_auc)
    plt.plot(tpr, 1-fpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([1, 0], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Background Rejection Efficiency')
    plt.xlabel('Signal Retention Efficiency')
    plt.savefig('../imagine/RUC_AUC ')
    # plt.show()


if __name__ == '__main__':
    main()
