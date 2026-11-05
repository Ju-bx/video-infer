#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

class IntQueue {
public:
    IntQueue(size_t max_size) {
        MAX_QUEUE = max_size;
    }

    void push(int x) {
        std::lock_guard<std::mutex> lock(m);
        if (q.size() >= MAX_QUEUE) {
            q.pop();                    // 丟掉最前面
            dropped_++;
        }
        q.push(x);
        cv.notify_one();
    }

    bool pop(int& out) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return !q.empty() || done; });

        if (q.empty() && done) return false;   // 全部處理完了
        out = q.front();
        q.pop();             // 從 queue 移除
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(m);
        done = true;
        cv.notify_one();
    }

    int dropped() {
        std::lock_guard<std::mutex> lock(m);
        return dropped_;
    }

private:
    std::queue<int> q;              // 生產者和消費者共用的queue
    std::mutex m;                   // 保護 q 的鎖
    std::condition_variable cv;     // 用來叫醒消費者
    bool done = false;              // 生產者是否已經送完
    size_t MAX_QUEUE = 3;
    int dropped_ = 0;
};

void producer(IntQueue& q) {
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        q.push(i);                     // 空格 A：把 i 放進 queue
    }
    q.close();                         // 空格 B：送完了，通知 queue
}

void consumer(IntQueue& q) {
    int x;
    while (q.pop(x)) {                // 空格 C：拿得到東西就繼續
        std::cout << "got " << x << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

int main() {
    IntQueue q(3);                  // 建立一個上限 3 的 queue

    std::thread p(producer, std::ref(q));   // 空格 D
    std::thread c(consumer, std::ref(q));   // 空格 E

    p.join();
    c.join();
    std::cout << "dropped " << q.dropped() << "\n";
    return 0;
}