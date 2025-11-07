/*
* FIXED SIZE RING BUFFER WITH SEMAPHORE BASED FULL/EMPTY EVENT SIGNALLING
methods required:
* pop
* push
* drain
* construct(LENGTH)
data required:
* sem_is_empty
* sem_is_full
* capacity
* tail_idx
* head_idx
* data[] -> array (fixed-size at compile time, no heap) of accel samples 
notes:
- works for single producer/single consumer only (no mutual exclusion guarantee for multithread access)
- implemented for compile-time control of capacity (not runtime)
- it is blocking, meaning when full, it waits for pop before push is allowed
*/
#pragma once
#include <semaphore>
#include <cstddef> // std::size_t
#include <vector>
#include <utility>
#include <atomic>
#include "types.hpp"
#include "configs.hpp"

// semaphore template parameter type is ptrdiff_t (type for the update param in the release() function)
// setting its max to 250 means we're saying, 250 is the max we can release at once (really we'll keep it at 1 for this)
static constexpr std::ptrdiff_t SEM_BUFFER_CAPACITY = 250;

// "T" will be LSM9DS1 burst reads for this application (defined in types.h)
template<typename T>
class ringBuffer_C {
public:

    // a semaphore's count gives current number of allowed allocated 'slots' that can be 'taken'
    std::counting_semaphore<SEM_BUFFER_CAPACITY> sem_buffer_slots_available;
    std::counting_semaphore<SEM_BUFFER_CAPACITY> sem_data_items_available;

    // Constructor
    explicit ringBuffer_C(size_t capacity);

    // ring buffer methods
    bool pop(T *dest);
    bool push(const T& data);
    size_t drain(T *dest);
    void close();
    size_t get_count() const { return count_.load(std::memory_order_acquire); }; 

private:
    size_t const capacity_;
    size_t tailIdx_ = 0;
    size_t headIdx_ = 0;
    std::atomic<size_t> count_ = 0;
    std::atomic<bool> isClosed_ = 0; // open upon init; atomic because both threads use it
    // full/empty conditions based on semaphore logic
    // do not use isFull() for now --> can lead to data race since both consumer/producer read and can write by changing head/tail  
    bool isFull() const { return 1 ? count_.load(std::memory_order_acquire) == capacity_ : 0; }; // full if the next write index (tail+1, wrapped) would collide with head
    // data array
    std::vector<T> ringBufferArr;
};

template<typename T>
ringBuffer_C<T>::ringBuffer_C(size_t capacity)
: capacity_(capacity), sem_buffer_slots_available(static_cast<std::ptrdiff_t>(capacity)), sem_data_items_available(0) {
    ringBufferArr.resize(capacity_);
}

template<typename T>
bool ringBuffer_C<T>::push (const T& data) { 
    if(isClosed_.load(std::memory_order_relaxed)) {
        return 0;
    }
    // each time we push, we 'take' a semaphore slot (from the sem_full)
    sem_buffer_slots_available.acquire(); // BLOCKING. producer waits here.
    // should i use try_acquire_until instead? when sem_full is 0 it's time to drain.
    // sem full should only be 'acquirable' again once thing has drained 
    
    // if we get unblocked here from close() call, need to return and release
    if(isClosed_.load((std::memory_order_relaxed))) {
        // must give slot back because we didn't really push anything from here
        sem_buffer_slots_available.release();
        return 0;
    }

    // push to tail
    ringBufferArr[tailIdx_] = data;
    // sems guarantee there's space to add when push happens
    count_++;
    
    // always finish push with tail increment & wrap-around if this increment causes tailIdx_ >= capacity
    tailIdx_ ++; 
    if(tailIdx_ >= capacity_) {
        // wrap-around
        tailIdx_ = 0;
    }

    sem_data_items_available.release(); // "we've pushed something that can be emptied"

    return 1;
}

// pops until an item exists; returns false if closed (BLOCKING)
template<typename T>
bool ringBuffer_C<T>::pop(T *dest) {
    if(isClosed_.load(std::memory_order_relaxed)) {
        return 0;
    }
    sem_data_items_available.acquire(); // need to make sure there's something we can pop
    if(isClosed_.load((std::memory_order_relaxed))) {
        return 0;
    }
    // pop from head, then increment (& consider wrap-around)
    *dest = std::move(ringBufferArr[headIdx_]);
    headIdx_++;
    if(headIdx_ >= capacity_) {
        // wraparound
        headIdx_ = 0;
    }
    sem_buffer_slots_available.release(); // added a slot
    count_--; 
    return 1;
}

// drain: pops as many as currently available items (non-blocking -> skips if can't acquire sem)
// returns num of items
template<typename T>
size_t ringBuffer_C<T>::drain(T* dest) {
    if(isClosed_.load(std::memory_order_acquire)){
        return 0;
    }
    // pop everything from head to tail - continue checking try_acquire() on sem until it fails
    size_t i = 0;
    while(sem_data_items_available.try_acquire()){
        if(isClosed_.load(std::memory_order_relaxed)) {
            break; // if closed, break and return how many you've added before close
        }
        *(dest+i) = std::move(ringBufferArr[headIdx_]);
        headIdx_++;
        count_--;
        // handle wrap around
        if(headIdx_>= capacity_){
            headIdx_ = 0;
        }
        i++;
        sem_buffer_slots_available.release(); // we've got another spot we could write to in the meantime w producer thread
    }

    return i;
}

template<typename T>
void ringBuffer_C<T>::close() {
    isClosed_.store(true, std::memory_order_release);
    // free the semaphores incase they were waiting for an acq
    sem_buffer_slots_available.release();
    sem_data_items_available.release();
}




