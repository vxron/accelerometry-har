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
#include <array>
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

    // constructor for sems with appropriate count inits is required
    ringBuffer_C()
        : sem_buffer_slots_available(SEM_BUFFER_CAPACITY),
        sem_data_items_available(0) {}

    size_t const capacity_ = RING_BUFFER_CAPACITY;
    // a semaphore's count gives current number of allowed allocated 'slots' that can be 'taken'
    std::counting_semaphore<SEM_BUFFER_CAPACITY> sem_buffer_slots_available;
    std::counting_semaphore<SEM_BUFFER_CAPACITY> sem_data_items_available;

    // ring buffer methods
    bool pop(T *dest);
    bool push(const T& data);
    size_t drain(T *dest);
    void close();

private:
    size_t tailIdx_ = 0;
    size_t headIdx_ = 0;
    std::atomic<bool> isClosed_ = 0; // open upon init; atomic because both threads use it
    // full/empty conditions based on semaphore logic
    // do not use isFull() for now --> can lead to data race since both consumer/producer read and can write by changing head/tail  
    bool isFull() const { return 1 ? ((tailIdx_+1) % RING_BUFFER_CAPACITY) == headIdx_ : 0; }; // full if the next write index (tail+1, wrapped) would collide with head
    // data array
    std::array<T,RING_BUFFER_CAPACITY> ringBufferArr;
};


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

/* // this drop-oldest clause is dead code with sems (acquire()) that only let u push when there's slots available...
// u will never reach here unless there are slots available
// HOWEVER -> u can change this logic down the line if u want and use try_acquire() then go here if acquire fails...
    else {
        // trying to write to full queue -> use drop oldest approach
        // head and tail should be at the same index in this case
        ASSERT(headIdx_ == tailIdx_);
        ringBufferArr[tailIdx_] = std::move(data);
        headIdx_ ++; // overwrite, shift head 
    }
*/
    // always finish push with tail increment & wrap-around if this increment causes tailIdx_ >= capacity
    tailIdx_ ++; 
    if(tailIdx_ >= capacity_) {
        // wrap-around
        tailIdx_ = 0;
    }

    sem_data_items_available.release(); // "we've pushed something that can be emptied"

    return 1;
}

// pops until an item exists; returns false if closed
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
    return 1;
}

// drain: pops as many as currently available items (non-blocking -> skips if can't acquire sem)
// returns num of items
template<typename T>
size_t ringBuffer_C<T>::drain(T* dest) {
    if(isClosed_){
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
    isClosed_ = 1;
    // free the semaphores incase they were waiting for an acq
    sem_buffer_slots_available.release();
    sem_data_items_available.release();
}




