#代码上传时间:2019/5/12
#需要学习的数据,movielens100k
import numpy as np
import matplotlib.pyplot as plt

def init(tmp , ratio):
    np.random.shuffle(tmp) #对tmp重新洗牌
    train = tmp[0:int(len(tmp)*ratio)]
    test =  tmp[int(len(tmp) * ratio):]
    return  train , test

def Show(maes , rmses , alpha = 0.6):
    x_aixs = []
    for i in range(len(maes)):
        x_aixs.append(i)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.set_title("MAE")
    ax2.set_title("RMSE")
    ax1.set_xlabel("time")
    ax2.set_xlabel("time")
    ax1.set_ylabel("rate")
    ax2.set_ylabel("rate")
    ax1.scatter(x_aixs, maes, alpha=0.6 , color = 'm', s = 5)
    ax2.scatter(x_aixs, rmses, alpha=0.6 , s = 5)
    plt.xlim(right=200, left=0)
    plt.ylim(top=rmses[0] + 0.5 , bottom = 0)
    plt.show()

def predict(test, u , bu , bi , X , Y):
    mae = 0
    rmse = 0
    N = len(test)
    for k in range(N):  # 计算测试数据误差
        i = test[k][0]
        j = test[k][1]
        rate = test[k][2]
        difference = rate - u - bu[i] - bi[j] + (X[i] - Y[j]).dot(X[i] - Y[j])
        rmse += np.square(difference)
        mae += np.absolute(difference)
    rmse = np.sqrt(rmse / N)
    mae /= N
    return  rmse , mae

def EuclideanE(R , bi , bu , X , Y , u ,test, steps = 1500 , alpha = 0.01 , lbda = 0.05):
    maes = []
    rmses = []
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0 :
                    eui = R[i][j] - u - bu[i] - bi[j] + (X[i] - Y[j]).dot(X[i] - Y[j])
                    bu[i] = bu[i] + alpha * (eui - lbda * bu[i])
                    bi[j] = bi[j] + alpha * (eui - lbda * bi[j])
                    X[i] = X[i] - alpha * (X[i] - Y[j]) * (eui + lbda)
                    Y[j] = Y[j] + alpha * (X[i] - Y[j]) * (eui + lbda)
        loss = 0
        for i in range(len(R)):
            for j in range(len(R[0])):
                if R[i][j] > 0:
                    tmp = R[i][j] - u + bu[i] + bi[j] - (X[i] - Y[j]).dot(X[i] - Y[j])
                    loss += tmp * tmp
                    loss += lbda * (bu[i] * bu[i] + bi[j] * bi[j] + (X[i] - Y[j]).dot(X[i] - Y[j]))
        Delta = 9999
        if step == 0:
            lastloss = loss
        else:
            Delta = abs(lastloss - loss)
            if Delta < 1:
                break
            lastloss = loss
        rmse , mae = predict(test, u , bu , bi , X , Y)
        print("step:", step)
        print("Delta", Delta)
        print("RMSE值:",rmse)
        print("MAE值:",mae)
        maes.append(mae)
        rmses.append(rmse)
    Show(maes, rmses)

if __name__ == "__main__":
    ratio = 0.8
    K = 10 #隐藏因素维度
    M = 944
    N = 1683
    tmp = np.loadtxt("u.data" , int)
    tmp = np.delete(tmp , 3 , 1)
    u = np.mean(tmp[:,2])
    train , test = init(tmp , ratio)
    R = np.zeros((M,N))
    for i in range(len(train)):
        R[train[i][0] ][train[i][1] ] = train[i][2]
    X = np.random.rand(len(R),K)
    Y = np.random.rand(len(R[0]),K)
    bu = np.random.rand(len(R))
    bi = np.random.rand(len(R[0]))
    # for i in range(len(bi)):
    #     print(bi[i])
    EuclideanE(R , bi , bu , X , Y , u , test)
