#include "ledmatrix.hpp"
#include "logger.hpp"

#include <fcntl.h>    // open
#include <unistd.h>   // write, close, lseek
#include <cerrno>
#include <cstring>
#include <cstddef>
#include <thread>
#include <chrono>

struct Rgb {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

namespace {

// ======= PALETTE =======

// Sitting
constexpr Rgb SIT_PALETTE[3] = {
    { 10,  0,  30},
    {255,105, 180},
    {120,200, 255},
};

// Standing
constexpr Rgb STAND_PALETTE[3] = {
    {  0,  0,  25},
    {120,220, 255},
    {190,140, 255},
};

// Walking
constexpr Rgb WALK_PALETTE[3] = {
    {  5,  0,  35},
    { 80,230, 210},
    {255, 90, 200},
};

// Turning
constexpr Rgb TURN_PALETTE[3] = {
    {  5,  0,  35},
    {255, 80, 210},
    {120,180, 255},  
};

// ======= 8x8 PATTERNS =======

// "SIT": sort of an S shape in the middle
constexpr uint8_t SIT_PATTERN[8][8] = {
    {0,0,1,1,1,0,0,0},
    {0,1,0,0,0,1,0,0},
    {0,1,1,1,1,1,0,0},
    {0,1,0,0,0,0,0,0},
    {0,1,1,1,1,0,0,0},
    {0,0,0,0,1,0,0,0},
    {0,0,0,0,1,0,0,0},
    {0,0,2,0,0,2,0,0}, // little pink dots under like a chair
};

// "STAND": tall vertical bar with a little head
constexpr uint8_t STAND_PATTERN[8][8] = {
    {0,0,0,1,1,0,0,0},
    {0,0,0,1,1,0,0,0},
    {0,0,0,1,1,0,0,0},
    {0,0,0,1,1,0,0,0},
    {0,0,0,1,1,0,0,0},
    {0,0,0,1,1,0,0,0},
    {0,0,2,0,0,2,0,0},
    {0,2,0,0,0,0,2,0},
};

// "WALK": right-pointing arrow with a trail
constexpr uint8_t WALK_PATTERN[8][8] = {
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,2,0,0},
    {0,0,0,0,2,2,0,0},
    {1,1,1,2,2,2,2,0},
    {1,1,1,2,2,2,2,0},
    {0,0,0,0,2,2,0,0},
    {0,0,0,0,0,2,0,0},
    {0,0,0,0,0,0,0,0},
};

// "TURN": circular-ish arrow indicating rotation
constexpr uint8_t TURN_PATTERN[8][8] = {
    {0,0,0,2,2,0,0,0},
    {0,0,2,1,1,2,0,0},
    {0,2,1,0,0,1,2,0},
    {2,1,0,0,0,0,1,2},
    {2,1,0,0,0,0,1,2},
    {0,2,1,0,0,1,2,0},
    {0,0,2,1,1,2,0,0},
    {0,0,0,2,0,0,0,0},
};

// Idle calibration: blue/purple gradient with a "C" icon
constexpr Rgb CALIB_IDLE_PALETTE[3] = {
    {  5,  0,  25},
    {120,180,255},
    {200,120,255},
};

constexpr uint8_t CALIB_IDLE_PATTERN[8][8] = {
    {0,0,1,1,1,1,0,0},
    {0,1,0,0,0,0,1,0},
    {1,0,0,0,0,0,0,1},
    {1,0,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,1},
    {0,1,0,0,0,0,1,0},
    {0,0,1,1,1,1,0,0},
};

// Active recording: pink/purple with a "dot + bars" record symbol
constexpr Rgb CALIB_REC_PALETTE[3] = {
    { 10,  0,  25},
    {255, 80, 180},
    {180,120,255},
};

constexpr uint8_t CALIB_REC_PATTERN[8][8] = {
    {0,0,1,1,1,1,0,0},
    {0,1,2,2,2,2,1,0},
    {1,2,2,2,2,2,2,1},
    {1,2,2,2,2,2,2,1},
    {1,2,2,2,2,2,2,1},
    {0,1,2,2,2,2,1,0},
    {0,0,1,1,1,1,0,0},
    {0,0,0,0,0,0,0,0},
};

// WELCOME MSG SYMBOLS
static constexpr uint8_t GLYPH_W[7][5] = {
    {1,0,0,0,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
    {1,1,0,1,1},
    {1,0,0,0,1},
};

static constexpr uint8_t GLYPH_E[7][5] = {
    {1,1,1,1,1},
    {1,0,0,0,0},
    {1,1,1,1,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,1,1,1,1},
};

static constexpr uint8_t GLYPH_L[7][5] = {
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,1,1,1,1},
};

static constexpr uint8_t GLYPH_C[7][5] = {
    {0,1,1,1,0},
    {1,0,0,0,1},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,0},
    {1,0,0,0,1},
    {0,1,1,1,0},
};

static constexpr uint8_t GLYPH_O[7][5] = {
    {0,1,1,1,0},
    {1,0,0,0,1},
    {1,0,0,0,1},
    {1,0,0,0,1},
    {1,0,0,0,1},
    {1,0,0,0,1},
    {0,1,1,1,0},
};

static constexpr uint8_t GLYPH_M[7][5] = {
    {1,0,0,0,1},
    {1,1,0,1,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
    {1,0,1,0,1},
};

} // namespace

// =================== START DRAWING HELPERS =========================
static uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    uint16_t r5 = (r >> 3) & 0x1F;  // 8 bits → 5 bits
    uint16_t g6 = (g >> 2) & 0x3F;  // 8 bits → 6 bits
    uint16_t b5 = (b >> 3) & 0x1F;  // 8 bits → 5 bits

    return (r5 << 11) | (g6 << 5) | b5;
}


static void draw_pattern_to_fb(int fb_fd,
                               const uint8_t pattern[8][8],
                               const Rgb* palette,
                               std::size_t palette_size)
{
#if !I2C_MOCK && defined(__linux__)
    if (fb_fd < 0) return;

    uint16_t buffer[64];
    int idx = 0;

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            uint8_t p = pattern[y][x];
            if (p >= palette_size) p = 0; // safety clamp

            const Rgb& c = palette[p];
            buffer[idx++] = rgb888_to_rgb565(c.r, c.g, c.b);
        }
    }

    if (::lseek(fb_fd, 0, SEEK_SET) < 0) {
        LOG_ALWAYS("LedMatrixDriver: lseek failed: " << std::strerror(errno));
    }

    const ssize_t bytes_to_write = sizeof(buffer);
    ssize_t n = ::write(fb_fd, buffer, bytes_to_write);
    if (n != bytes_to_write) {
        LOG_ALWAYS("LedMatrixDriver: write() short or failed (n="
                   << n << "): " << std::strerror(errno));
    }
#else
    (void)fb_fd;
    (void)pattern;
    (void)palette;
    (void)palette_size;
    LOG_ALWAYS("LedMatrixDriver (mock): draw_pattern_to_fb");
#endif
}
// =================== END DRAWING HELPERS ===========================

LedMatrixDriver::LedMatrixDriver()
    : fb_fd_(-1)
{
#if !I2C_MOCK && defined(__linux__)
    open_dev_kernel();
#else
    // mock / non-Linux: nothing to open
#endif
}

LedMatrixDriver::~LedMatrixDriver()
{
#if !I2C_MOCK && defined(__linux__)
    close_dev_kernel();
#endif
}

void LedMatrixDriver::open_dev_kernel()
{
#if !I2C_MOCK && defined(__linux__)
    if (fb_fd_ != -1) return; // already open

    fb_fd_ = ::open("/dev/fb0", O_RDWR | O_CLOEXEC);
    if (fb_fd_ < 0) {
        LOG_ALWAYS("LedMatrixDriver: open(/dev/fb0) failed: " << std::strerror(errno));
    } else {
        LOG_ALWAYS("LedMatrixDriver: opened /dev/fb0");
    }
#else
    // mock / non-Linux: just log
    LOG_ALWAYS("LedMatrixDriver: open_dev_kernel (mock / non-Linux)");
#endif
}

void LedMatrixDriver::close_dev_kernel()
{
#if !I2C_MOCK && defined(__linux__)
    if (fb_fd_ >= 0) {
        ::close(fb_fd_);
        fb_fd_ = -1;
        LOG_ALWAYS("LedMatrixDriver: closed /dev/fb0");
    }
#endif
}

void LedMatrixDriver::close_and_reset()
{
#if !I2C_MOCK && defined(__linux__)
    if (fb_fd_ < 0) {
        open_dev_kernel();
        if (fb_fd_ < 0) return;
    }

    // 8x8 = 64 pixels, all black
    uint16_t buffer[64];
    std::fill(std::begin(buffer), std::end(buffer),
              rgb888_to_rgb565(0, 0, 0));

    if (::lseek(fb_fd_, 0, SEEK_SET) < 0) {
        LOG_ALWAYS("LedMatrixDriver: clear_matrix lseek failed: " << std::strerror(errno));
        return;
    }

    const ssize_t bytes_to_write = sizeof(buffer);
    ssize_t n = ::write(fb_fd_, buffer, bytes_to_write);
    close_dev_kernel();
#endif
}

void LedMatrixDriver::display_class_on_matrix(classes_e class_label)
{
    // Choose palette + pattern based on class
    const Rgb* palette = nullptr;
    const uint8_t (*pattern)[8] = nullptr;
    std::size_t palette_size = 0;

    switch (class_label) {
    case CLASS_SITTING:
        palette = SIT_PALETTE;
        palette_size = 3;
        pattern = SIT_PATTERN;
        break;
    case CLASS_STANDING:
        palette = STAND_PALETTE;
        palette_size = 3;
        pattern = STAND_PATTERN;
        break;
    case CLASS_WALKING:
        palette = WALK_PALETTE;
        palette_size = 3;
        pattern = WALK_PATTERN;
        break;
    case CLASS_TURNING_ON_SPOT:
        palette = TURN_PALETTE;
        palette_size = 3;
        pattern = TURN_PATTERN;
        break;
    default:
        // Unknown class: simple "X" on dark background
        static constexpr Rgb UNKNOWN_PALETTE[3] = {
            {  5,  0,  20},
            {255, 80, 210},
            {120,180, 255},
        };
        static constexpr uint8_t UNKNOWN_PATTERN[8][8] = {
            {1,0,0,0,0,0,0,1},
            {0,1,0,0,0,0,1,0},
            {0,0,1,0,0,1,0,0},
            {0,0,0,1,1,0,0,0},
            {0,0,0,1,1,0,0,0},
            {0,0,1,0,0,1,0,0},
            {0,1,0,0,0,0,1,0},
            {1,0,0,0,0,0,0,1},
        };
        palette = UNKNOWN_PALETTE;
        palette_size = 3;
        pattern = UNKNOWN_PATTERN;
        break;
    }

    if (!palette || !pattern) {
        return;
    }

#if !I2C_MOCK && defined(__linux__)
    if (fb_fd_ < 0) {
        open_dev_kernel();
        if (fb_fd_ < 0) return;
    }
    draw_pattern_to_fb(fb_fd_, pattern, palette, palette_size);
#else
    // mock / non-Linux: just log what class we would draw
    LOG_ALWAYS("LedMatrixDriver (mock): display_class_on_matrix class="
               << static_cast<int>(class_label));
#endif
}

void LedMatrixDriver::display_calibration_on_matrix(classes_e active_recording)
{
    const Rgb* palette = nullptr;
    const uint8_t (*pattern)[8] = nullptr;
    std::size_t palette_size = 0;

    if (active_recording == CLASS_UNKNOWN || active_recording == CLASS_REST) {
        palette = CALIB_IDLE_PALETTE;
        palette_size = 3;
        pattern = CALIB_IDLE_PATTERN;
    } else { // active recording
        palette = CALIB_REC_PALETTE;
        palette_size = 3;
        pattern = CALIB_REC_PATTERN;
    }

    if (!palette || !pattern) return;

#if !I2C_MOCK && defined(__linux__)
    if (fb_fd_ < 0) {
        open_dev_kernel();
        if (fb_fd_ < 0) return;
    }
    draw_pattern_to_fb(fb_fd_, pattern, palette, palette_size);
#endif
}

void LedMatrixDriver::display_welcome_message()
{
#if !I2C_MOCK && defined(__linux__)
    if (fb_fd_ < 0) {
        open_dev_kernel();
        if (fb_fd_ < 0) return;
    }

    using namespace std::chrono_literals;

    constexpr Rgb WELC_PALETTE[3] = {
        {  5,  0,  25},
        { 40, 10,  60},
        {255,105, 180},
    };

    // Message buffer: 8 rows x N cols (N > 8 so we can scroll).
    // Letters: "WELCOME" = 7 letters, each 5 cols + 1 col spacing.
    constexpr int LETTER_W = 5;
    constexpr int LETTER_H = 7;
    constexpr int LETTER_SP = 1;
    constexpr int NUM_LETTERS = 7;
    constexpr int LEFT_MARGIN = 8;
    constexpr int RIGHT_MARGIN = 8;
    constexpr int MSG_COLS =
        LEFT_MARGIN + NUM_LETTERS * (LETTER_W + LETTER_SP) + RIGHT_MARGIN;

    uint8_t msg[8][MSG_COLS] = {}; // all zeros initially

    auto blit_letter = [&](const uint8_t glyph[7][5], int& cursor_x) {
        for (int y = 0; y < LETTER_H; ++y) {
            for (int x = 0; x < LETTER_W; ++x) {
                if (glyph[y][x]) {
                    int col = cursor_x + x;
                    if (col >= 0 && col < MSG_COLS) {
                        msg[y][col] = 2;
                    }
                }
            }
        }
        cursor_x += LETTER_W + LETTER_SP;
    };

    int cursor_x = LEFT_MARGIN;
    blit_letter(GLYPH_W, cursor_x);
    blit_letter(GLYPH_E, cursor_x);
    blit_letter(GLYPH_L, cursor_x);
    blit_letter(GLYPH_C, cursor_x);
    blit_letter(GLYPH_O, cursor_x);
    blit_letter(GLYPH_M, cursor_x);
    blit_letter(GLYPH_E, cursor_x);

    // Scroll from off-screen right to off-screen left
    for (int offset = -8; offset < MSG_COLS; ++offset) {
        uint8_t frame[8][8];

        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                int src_x = offset + x;
                uint8_t idx = 0;

                if (src_x >= 0 && src_x < MSG_COLS) {
                    uint8_t v = msg[y][src_x];
                    if (v == 0) {
                        // background stripe: alternate rows between palette 0 and 1
                        idx = (y % 2 == 0) ? 0 : 1;
                    } else {
                        idx = v;
                    }
                } else {
                    // off-screen, background only
                    idx = (y % 2 == 0) ? 0 : 1;
                }

                frame[y][x] = idx;
            }
        }

        draw_pattern_to_fb(fb_fd_, frame, WELC_PALETTE, 3);
        std::this_thread::sleep_for(80ms); // adjust speed
    }
#endif
}

