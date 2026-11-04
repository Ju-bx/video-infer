#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

std::queue<int> q;              // 生產者和消費者共用的queue
std::mutex m;                   // 保護 q 的鎖
std::condition_variable cv;     // 用來叫醒消費者
bool done = false;              // 生產者是否已經送完
const size_t MAX_QUEUE = 3;
int dropped = 0;

void producer() {
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));   // 每 50 ms 一格

        {
            std::lock_guard<std::mutex> lock(m);
            if (q.size() >= MAX_QUEUE) {
                std::cout << "[producer] queue full, drop " << q.front() << "\n";
                q.pop();                    // 丟掉最前面
                dropped++;
            }
            q.push(i);
            std::cout << "[producer] push " << i << "   queue size = " << q.size() << "\n";
        }
        cv.notify_one();        // 叫醒C
    }

    {
        std::lock_guard<std::mutex> lock(m);
        done = true;
    }
    cv.notify_one();
}

void consumer() {
    while (true) {
        int frame;
        {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [] { return !q.empty() || done; });   // 沒東西就等

            if (q.empty() && done) break;   // 全部處理完了

            frame = q.front();   // 拿出最前面的
            q.pop();             // 從 queue 移除
            std::cout << "    [consumer] got " << frame << "   queue size = " << q.size() << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));   // 模擬推論，很慢
    }
}

int main() {
    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();
    std::cout << "dropped " << dropped << " frames\n";
    std::cout << "all done\n";
    return 0;
}