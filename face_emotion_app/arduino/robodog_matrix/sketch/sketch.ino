// RoboDog conversational status display for the UNO Q 8x13 LED matrix.
// Animation is entirely MCU-side and non-blocking; Linux only sends state changes.

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <zephyr/kernel.h>

Arduino_LED_Matrix matrix;
K_MUTEX_DEFINE(display_mutex);

enum RobotState {
  STATE_OFF,
  STATE_STARTING,
  STATE_WAITING,
  STATE_LISTENING,
  STATE_HEARING,
  STATE_THINKING,
  STATE_SPEAKING,
  STATE_ERROR
};

static RobotState current_state = STATE_STARTING;
static uint8_t phase = 0;
static unsigned long next_frame_at = 0;
static uint8_t frame[8 * 13];

void clear_frame() {
  memset(frame, 0, sizeof(frame));
}

void pixel(int x, int y, uint8_t brightness = 7) {
  if (x >= 0 && x < 13 && y >= 0 && y < 8) {
    frame[y * 13 + x] = brightness;
  }
}

void rows(const char *art[8], uint8_t brightness = 7) {
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 13; x++) {
      if (art[y][x] != '.') pixel(x, y, brightness);
    }
  }
}

void render_starting() {
  static const uint8_t ring[12][2] = {
    {4, 1}, {6, 0}, {8, 1}, {10, 2}, {11, 4}, {10, 6},
    {8, 7}, {6, 7}, {4, 7}, {2, 6}, {1, 4}, {2, 2}
  };
  for (int tail = 0; tail < 4; tail++) {
    int p = (phase + 12 - tail) % 12;
    pixel(ring[p][0], ring[p][1], 7 - tail);
  }
}

void render_waiting() {
  static const char *plug[8] = {
    ".....#.#.....",
    ".....#.#.....",
    "....#####....",
    "....#####....",
    ".....###.....",
    ".....###.....",
    "......#......",
    "......#......"
  };
  rows(plug, phase % 2 ? 7 : 3);
}

void render_listening() {
  static const char *ear[8] = {
    "....####.....",
    "...##..##....",
    "...#....#....",
    "...#..###....",
    "...#..#......",
    "....###......",
    ".....##......",
    "............."
  };
  rows(ear);
  // A soft pulse at the ear canal shows that capture is alive.
  pixel(7, 3, 3 + (phase % 3) * 2);
}

void render_hearing() {
  // Live-looking waveform: phase shifts a deterministic amplitude pattern.
  static const uint8_t amplitude[13] = {1, 2, 3, 2, 1, 3, 4, 2, 3, 1, 2, 3, 1};
  for (int x = 0; x < 13; x++) {
    int a = amplitude[(x + phase) % 13];
    for (int y = 4 - a; y <= 3 + a; y++) pixel(x, y);
  }
}

void render_thinking() {
  // Three animated dots, deliberately distinct from listening and speaking.
  for (int dot = 0; dot < 3; dot++) {
    int x0 = 2 + dot * 4;
    uint8_t level = ((phase / 2) % 3 == dot) ? 7 : 2;
    pixel(x0, 3, level); pixel(x0 + 1, 3, level);
    pixel(x0, 4, level); pixel(x0 + 1, 4, level);
  }
}

void render_speaking() {
  static const char *closed_mouth[8] = {
    ".............",
    ".............",
    "...#######...",
    "..##.....##..",
    "..#########..",
    "...#######...",
    ".............",
    "............."
  };
  static const char *open_mouth[8] = {
    ".............",
    "...#######...",
    "..##.....##..",
    "..#.......#..",
    "..#.......#..",
    "..##.....##..",
    "...#######...",
    "............."
  };
  rows(phase % 2 ? open_mouth : closed_mouth);
}

void render_error() {
  static const char *cross[8] = {
    ".##.......##.",
    "..##.....##..",
    "...##...##...",
    "....##.##....",
    "....##.##....",
    "...##...##...",
    "..##.....##..",
    ".##.......##."
  };
  rows(cross, phase % 2 ? 7 : 3);
}

void draw_state() {
  clear_frame();
  switch (current_state) {
    case STATE_OFF: break;
    case STATE_STARTING: render_starting(); break;
    case STATE_WAITING: render_waiting(); break;
    case STATE_LISTENING: render_listening(); break;
    case STATE_HEARING: render_hearing(); break;
    case STATE_THINKING: render_thinking(); break;
    case STATE_SPEAKING: render_speaking(); break;
    case STATE_ERROR: render_error(); break;
  }
  matrix.draw(frame);
}

void set_robodog_status(String value) {
  RobotState requested = STATE_ERROR;
  if (value == "off") requested = STATE_OFF;
  else if (value == "starting") requested = STATE_STARTING;
  else if (value == "waiting") requested = STATE_WAITING;
  else if (value == "listening") requested = STATE_LISTENING;
  else if (value == "hearing") requested = STATE_HEARING;
  else if (value == "thinking") requested = STATE_THINKING;
  else if (value == "speaking") requested = STATE_SPEAKING;
  else if (value == "error") requested = STATE_ERROR;

  k_mutex_lock(&display_mutex, K_FOREVER);
  if (requested != current_state) {
    current_state = requested;
    phase = 0;
    next_frame_at = 0;
  }
  k_mutex_unlock(&display_mutex);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("set_robodog_status", set_robodog_status);
}

void loop() {
  unsigned long now = millis();
  if (now < next_frame_at) {
    delay(5);
    return;
  }
  k_mutex_lock(&display_mutex, K_FOREVER);
  draw_state();
  phase++;
  next_frame_at = now + 140;
  k_mutex_unlock(&display_mutex);
}
