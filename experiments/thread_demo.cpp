#include <iostream>
#include <thread>
#include <mutex>

std::mutex cout_mutex;   // 一把鎖，所有 thread 共用

void work(int id) {
    for (int i = 0; i < 1000; ++i) {
        std::lock_guard<std::mutex> lock(cout_mutex);   // 鎖門
        std::cout << "thread " << id << " step " << i << "\n";
    }   // ← 離開這個大括號時，自動開鎖
}
int main() {
    std::thread t1(work, 1);   // 開一個 thread，執行 work(1)
    std::thread t2(work, 2);   // 再開一個，執行 work(2)

    t1.join();                 // 等 t1 做完
    t2.join();                 // 等 t2 做完

    std::cout << "all done\n";
    return 0;
}