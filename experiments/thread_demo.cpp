#include <iostream>
#include <thread>
#include <mutex>

std::mutex cout_mutex;

void work(int id) {
    for (int i = 0; i < 1000; ++i) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "thread " << id << " step " << i << "\n";
    }
}
int main() {
    std::thread t1(work, 1);   // 開一個 thread，執行 work(1)
    std::thread t2(work, 2);   // 再開一個，執行 work(2)

    t1.join();                 // 等 t1 做完
    t2.join();                 // 等 t2 做完

    std::cout << "all done\n";
    return 0;
}