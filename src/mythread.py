from threading import Thread

class MyThread(Thread):
    def __init__(self, func, args, semaphore):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)   # 不要忘记调用Thread的初始化方法
        self.func = func
        self.args = args
        self.semaphore = semaphore
        self.result = None

    def get_result(self):
        return self.result

    def run(self):
        
        self.semaphore.acquire()  # 获取信号 -1
        #print(threading.currentThread().getName() + ' is running.')
        self.result = self.func(*self.args)
        #print(threading.currentThread().getName() + ' is done.')
        self.semaphore.release()  # 释放信号 +1