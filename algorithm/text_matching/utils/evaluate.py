def cal_precision_recall_F1(count, y, prediction):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(count):       # 验证集、测试集要填集合的大小
        if y[i] == 1:
            if prediction[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction[i] == 1:
                FP += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('accuracy：', round(accuracy * 100, 2), '%\tprecision：', round(precision * 100, 2), '%\trecall：',
          round(recall * 100, 2), '%\tF1：', round(F1 * 100, 2), '%')