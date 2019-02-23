import matplotlib.pyplot as plt

def plot_acc_history(history, legend=['acc']):
    # print(history.history.keys())

    # 精度の履歴をプロット
    for h in history:
        plt.plot(h.history['acc'])
        
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(legend, loc='lower right')
    #plt.show()
    plt.savefig('acc_history.png')

def plot_loss_history(history, legend=['loss']):
    # 損失の履歴をプロット

    for h in history:
        plt.plot(h.history['loss'])

    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(legend, loc='lower right')
    #plt.show()
    plt.savefig('loss_history.png')

def outputfile_evalute(score,name):
    #結果のファイル出力
    path = 'result.txt'
    with open(path,mode='a') as f:
        f.write('===' + name + '===\n')
        f.write('Test loss:' + str(score[0]) + '\n')
        f.write('Test accuracy:' + str(score[1])+'\n')